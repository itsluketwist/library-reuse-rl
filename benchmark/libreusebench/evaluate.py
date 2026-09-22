"""Evaluate model responses against LibReuseBench: parse, install, run, and aggregate.

The flow is: parse every response and work out the union of third-party packages they import;
create the venv, install those packages, and run each response against its task's stdin/stdout
tests. Reporting gives correctness (pass rate) and library reuse on the same problems - the
signal this benchmark exists to provide - tracked at two strengths: how often a solution imported
an external library, and the stricter how often it actually called or accessed one. Alongside
these, radon measures each solution's size (logical lines of code) and maintainability index, to
see whether reuse makes code shorter or easier to maintain. The installed package plan is saved
with the results for debugging and reproducibility.

Since responses can name arbitrary third-party packages, installation is confirmed on stdin by
default (`confirm_packages`) - pass `confirm=None` to install automatically instead, for unattended
runs. For control and reproducibility the environment can instead be supplied as a package-per-line
`.txt` file (`env_file`), which skips both collection and confirmation. When confirmation declines,
or a package fails to install, an environment file is written for correction and reuse.
"""

import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any

from libreusebench.environment import _DEFAULT_TIMEOUT_SECONDS, Environment
from libreusebench.libraries import (
    extract_code,
    imported_libraries,
    pip_package,
    required_import_names,
    required_pip_packages,
    used_libraries,
)
from libreusebench.load import load_dataset, load_records
from llm_cgr import save_json
from radon.metrics import mi_visit
from radon.raw import analyze


def _benchmark_by_id() -> dict[str, dict]:
    """Load the benchmark splits flattened into a dict keyed by task id.

    Returns the {task_id: record} map.
    """
    return {
        record["task_id"]: record
        for records in load_dataset().values()
        for record in records
    }


def load_responses(
    file_path: str,
) -> dict[str, list[str]]:
    """Load model responses, keyed by task id, from a .json or .jsonl file (see load_records).

    Each record is either {"task_id": ..., "responses": [...]} or {"task_id": ...,
    "response": "..."}; single responses are collated by id.

    Returns the {task_id: [response, ...]} map.
    """
    responses: dict[str, list[str]] = defaultdict(list)
    for record in load_records(file_path):
        task_id = record["task_id"]
        if "responses" in record:
            responses[task_id].extend(record["responses"])
        elif "response" in record:
            responses[task_id].append(record["response"])
        else:
            raise ValueError(f"record {task_id} has neither 'responses' nor 'response'")
    return dict(responses)


def plan_installs(
    responses: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Scan every response and collect the pip packages needed to run them all.

    Each required pip distribution is mapped back to the import name(s) it came from, so the user
    can see exactly what will be installed - and any import->distribution remaps - before agreeing.

    Returns a {pip package: {import names}} map.
    """
    origins: defaultdict[str, set[str]] = defaultdict(set)
    for response_list in responses.values():
        for response in response_list:
            for name in required_import_names(response):
                origins[pip_package(name)].add(name)
    return dict(origins)


def confirm_packages(
    plan: dict[str, set[str]],
) -> bool:
    """Show the third-party packages that will be pip-installed and ask for confirmation on stdin.

    Returns whether the user approved installing them and running the evaluation.
    """
    if not plan:
        print("No third-party packages to install.")
        return True
    print(
        f"\nThe responses import {len(plan)} third-party package(s) to install into the eval venv:"
    )
    for package in sorted(plan):
        imports = sorted(plan[package])
        # note the import name only when it differs from the distribution name (sklearn -> scikit-learn)
        note = "" if imports == [package] else f"  (import: {', '.join(imports)})"
        print(f"  - {package}{note}")
    reply = input("\nInstall these and run the evaluation? [y/N] ").strip().lower()
    return reply in ("y", "yes")


# the editable environment file the user fixes up and passes back with --env-file
_ENV_FILENAME = "libreusebench_env.txt"


def read_env_file(
    path: str,
) -> set[str]:
    """Read a package-per-line environment file, ignoring blank lines and # comments.

    Each remaining line is a pip spec (a name, or a name with a version pin such as numpy==2.5.1).

    Returns the set of pip package specs to install.
    """
    packages: set[str] = set()
    for line in Path(path).read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec:
            packages.add(spec)
    return packages


def write_env_file(
    directory: str,
    packages: set[str],
    origins: dict[str, set[str]] | None = None,
    failed: set[str] | None = None,
) -> Path:
    """Write an editable environment file listing the packages, for the user to correct and reuse.

    Annotates each package with the import it came from (when they differ) and flags any that
    failed to install, so the user can fix names/versions or drop libraries they do not want.

    Returns the path written.
    """
    origins = origins or {}
    failed = failed or set()
    lines = [
        "# LibReuseBench eval environment",
        "# one pip package per line; blank lines and '#' comments are ignored.",
        "# edit as needed (remove unwanted libraries, fix names, pin versions), then re-run:",
        "#   libreuseeval <responses.jsonl> --env-file <this file>",
        "",
    ]
    for package in sorted(packages):
        comments = []
        imports = sorted(origins.get(package, set()))
        if imports and imports != [package]:
            comments.append(f"import: {', '.join(imports)}")
        if package in failed:
            comments.append("FAILED TO INSTALL - fix or remove")
        suffix = f"  # {'; '.join(comments)}" if comments else ""
        lines.append(f"{package}{suffix}")

    path = Path(directory) / _ENV_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _code_metrics(
    code: str,
) -> tuple[int | None, float | None]:
    """Measure a program's size and maintainability with radon.

    Size is logical lines of code (lloc): one per statement, so blank lines, comments, and
    statements split over several lines don't change it. The maintainability index is radon's
    0-100 score (higher is more maintainable), built from the code's halstead volume, cyclomatic
    complexity, size, and share of comment lines (docstrings count as comments here).

    Returns a (lloc, maintainability_index) tuple, both None if radon cannot parse the code.
    """
    # model code isn't trusted to be valid python, so a parse failure means "not measured",
    # and the SyntaxWarnings from non-raw regex strings are silenced as in libraries.py
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return analyze(code).lloc, mi_visit(code, multi=True)
    except Exception:
        return None, None


def _score_response(
    response: str,
    record: dict,
    env: Environment,
    timeout_seconds: float,
    install_failed: set[str],
) -> dict[str, Any]:
    """Run one response against its task's tests and score it (packages already installed).

    Tracks library reuse at two strengths: `external_libs` is every third-party library the
    response imports, `used_external_libs` is the subset it actually calls or accesses - an
    unused import inflates the former without showing real reuse. Also records the code's
    size and maintainability (see _code_metrics).

    Returns a per-response result dict (passed, external_lib, external_lib_used, lloc, ...).
    """
    code = extract_code(response)
    if code is None:
        return {
            "passed": False,
            "external_libs": [],
            "external_lib": False,
            "used_external_libs": [],
            "external_lib_used": False,
            "lloc": None,
            "maintainability_index": None,
            "reason": "no code",
        }

    external = imported_libraries(response)
    used = used_libraries(response)
    lloc, maintainability_index = _code_metrics(code)
    # any of this response's own packages that failed the up-front install (explains a failure)
    missing = sorted(required_pip_packages(response) & install_failed)
    passed, message = env.evaluate(
        program=code,
        inputs=record["test_inputs"],
        expected_outputs=record["expected_outputs"],
        timeout_seconds=timeout_seconds,
    )
    return {
        "passed": passed,
        "external_libs": sorted(external),
        "external_lib": bool(external),
        "used_external_libs": sorted(used),
        "external_lib_used": bool(used),
        "lloc": lloc,
        "maintainability_index": maintainability_index,
        "install_failed": missing,
        "reason": "" if passed else message,
    }


def _mean(
    values: Iterable[float],
) -> float:
    """The arithmetic mean of some values, or 0.0 when there are none.

    Returns the mean.
    """
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """The unbiased pass@k estimate for one task: P(at least 1 of k samples correct).

    Given `n` samples of which `c` passed, this is 1 - C(n-c, k) / C(n, k) (the HumanEval estimator).

    Returns the estimate in [0, 1].
    """
    k = min(k, n)
    if (
        n - c < k
    ):  # too few failures to fill k slots, so at least one is guaranteed correct
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _lib_stats(
    lib_sets: list[set],
) -> dict:
    """Library-use statistics over external-library sets (one per task, or one per response).

    Returns {using_lib_rate, avg_extlibs, library_counts}: the fraction that used any library, the
    mean number of distinct libraries, and a per-library count of how many used it.
    """
    counts: Counter = Counter()
    for libs in lib_sets:
        counts.update(libs)
    return {
        "using_lib_rate": _mean(1.0 if libs else 0.0 for libs in lib_sets),
        "avg_extlibs": _mean(len(libs) for libs in lib_sets),
        "library_counts": dict(counts.most_common()),
    }


def _code_stats(
    rows: list[dict],
) -> dict:
    """Average code size and maintainability over per-response results.

    Only responses radon could measure are averaged (no-code and unparseable responses are
    skipped), and `n_measured` says how many that was. The averages are None, not 0.0, when
    nothing was measured - a 0.0 average size would wrongly read as very short code.

    Returns {n_measured, avg_lloc, avg_maintainability_index}.
    """
    measured = [row for row in rows if row.get("lloc") is not None]
    if not measured:
        return {
            "n_measured": 0,
            "avg_lloc": None,
            "avg_maintainability_index": None,
        }
    return {
        "n_measured": len(measured),
        "avg_lloc": _mean(row["lloc"] for row in measured),
        "avg_maintainability_index": _mean(
            row["maintainability_index"] for row in measured
        ),
    }


def _aggregate(
    per_response: list[dict],
    k: int | None = None,
) -> dict:
    """Aggregate per-response results into pass@1 / pass@k and library-use statistics.

    Responses are grouped by task. Pass rates use the unbiased pass@k estimator (k defaults to the
    smallest number of samples any task has, so it is well-defined everywhere). Library-use is
    tracked at two strengths - "imported" (any third-party library named in an import) and "used"
    (the subset actually called or accessed) - each reported both per task (true if any of its
    responses hit the bar) and per response, with a using-rate, an average, and a per-library count.
    Code size and maintainability are averaged per response, over all responses and over just
    the passing ones - failing code can be short simply because it is broken or unfinished.

    Returns the results dict (without the per-response detail, which the caller attaches).
    """
    # group by task: sample count, correct count, and the union of libraries imported/used
    tasks: dict[str, dict] = {}
    for row in per_response:
        task = tasks.setdefault(
            row["task_id"],
            {
                "n": 0,
                "correct": 0,
                "imported_libs": set(),
                "used_libs": set(),
            },
        )
        task["n"] += 1
        task["correct"] += int(row["passed"])
        task["imported_libs"].update(row.get("external_libs", []))
        task["used_libs"].update(row.get("used_external_libs", []))

    values = list(tasks.values())
    samples_min = min((t["n"] for t in values), default=0)
    k = samples_min if k is None else min(k, samples_min)

    return {
        "n_tasks": len(values),
        "n_responses": len(per_response),
        "k": k,
        "pass@1": _mean(t["correct"] / t["n"] for t in values),
        "pass@k": _mean(_pass_at_k(t["n"], t["correct"], k) for t in values),
        "library_use": {
            "imported": {
                "per_task": _lib_stats([t["imported_libs"] for t in values]),
                "per_response": _lib_stats(
                    [set(row.get("external_libs", [])) for row in per_response]
                ),
            },
            "used": {
                "per_task": _lib_stats([t["used_libs"] for t in values]),
                "per_response": _lib_stats(
                    [set(row.get("used_external_libs", [])) for row in per_response]
                ),
            },
        },
        "code_metrics": {
            "all": _code_stats(per_response),
            "passed": _code_stats([row for row in per_response if row["passed"]]),
        },
    }


def score_responses(
    responses: dict[str, list[str]],
    benchmark: dict[str, dict],
    env: Environment,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    install_failed: set[str] | None = None,
    k: int | None = None,
) -> dict:
    """Score in-memory responses with the official benchmark execution and statistics.

    This is the reusable scoring half of :func:`evaluate_responses`. The caller owns
    environment setup, which lets experiment code use a prebuilt, pinned evaluation
    environment while the standalone CLI keeps its install-planning workflow.

    Returns the aggregate statistics together with per-response results.
    """
    unknown = set(responses) - set(benchmark)
    if unknown:
        raise ValueError(f"responses reference unknown task ids: {sorted(unknown)}")

    failed = install_failed or set()
    per_response: list[dict] = []
    for task_id, task_responses in responses.items():
        record = benchmark[task_id]
        for idx, response in enumerate(task_responses):
            result = _score_response(
                response=response,
                record=record,
                env=env,
                timeout_seconds=timeout_seconds,
                install_failed=failed,
            )
            per_response.append({"task_id": task_id, "idx": idx, **result})

    return _aggregate(per_response, k) | {"results": per_response}


def evaluate_responses(
    responses: dict[str, list[str]],
    output_dir: str = "output",
    env_dir: str | None = None,
    env_root: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    env_file: str | None = None,
    k: int | None = None,
    confirm: Callable[[dict[str, set[str]]], bool] | None = confirm_packages,
) -> dict:
    """Evaluate in-memory responses to LibReuseBench and save the results.

    This is the complete workflow (unlike score_responses' scoring-only half): it plans,
    confirms, and installs the venv itself before scoring. For responses kept in a .json
    or .jsonl file rather than already in memory, see evaluate_responses_file.

    The venv is disposable by default - built fresh and deleted afterwards, so evaluations
    running at the same time can't corrupt each other's. Name an `env_dir` to reuse one across
    runs instead, or `env_root` to keep the disposable one somewhere specific (see Environment).

    The environment is decided one of two ways. With `env_file`, that file is the definitive list
    of packages to install. Otherwise the responses are parsed and the collected packages are
    confirmed (by default, on stdin via `confirm_packages`) before installing. Pass `confirm=None`
    to install automatically without asking - only do this in unattended contexts, since responses
    can name arbitrary third-party packages. If confirmation declines, or a package fails to
    install, an editable environment file is written for the caller to fix up and pass back with
    `env_file` - and the run stops.

    On success, reports pass@1 / pass@k and per-problem library-reuse statistics, split into
    "imported" (fraction of problems naming a library, average external libraries per problem, and
    a per-library task count) and the stricter "used" (the same statistics but only counting
    libraries actually called or accessed), and the average code size (lloc) and maintainability
    index, over all and over passing responses. The installed package plan and per-response
    details are saved with them to a timestamped JSON. `k` defaults to the smallest number of
    samples any task has.

    Returns the results dict (empty if the run stopped for the environment to be corrected).
    """
    n_responses = sum(len(task_responses) for task_responses in responses.values())
    print(
        f"Evaluating {n_responses} LibReuseBench response(s) across {len(responses)} task(s)"
    )
    benchmark = _benchmark_by_id()

    unknown = set(responses) - set(benchmark)
    if unknown:
        raise ValueError(f"responses reference unknown task ids: {sorted(unknown)}")

    plan = plan_installs(responses)

    if env_file is not None:
        # the supplied file is the validated environment - use it as-is
        packages = read_env_file(env_file)
        print(f"using environment from {env_file}: {len(packages)} package(s)")
    else:
        # install automatically unless the caller explicitly supplies an approval gate
        if confirm is not None and not confirm(plan):
            path = write_env_file(output_dir, set(plan), origins=plan)
            print(
                f"\nwrote {path} - edit it (remove/rename packages) and re-run with "
                f"env_file={path} (or --env-file {path} via the libreuseeval CLI)"
            )
            return {}
        packages = set(plan)

    # only now create the venv and install the chosen packages in one batch. the context
    # manager deletes it on the way out unless the caller named an env_dir to reuse.
    with Environment(cache_dir=env_dir, scratch_root=env_root) as env:
        # worth saying out loud: a disposable environment is randomly named and deleted on
        # the way out, so this is the only chance to see where a failing run was running
        print(f"environment: {env.cache_dir}{'' if env_dir else ' (disposable)'}")
        install_failed = env.ensure_packages(packages)
        if install_failed:
            # emit a fixable environment file and stop, so the user can correct it and re-run
            path = write_env_file(
                output_dir, packages, origins=plan, failed=install_failed
            )
            print(
                f"\ncould not install {sorted(install_failed)}."
                f"\nwrote {path} - fix the offending package(s) and re-run with "
                f"env_file={path} (or --env-file {path} via the libreuseeval CLI)"
            )
            return {}

        # correctness (pass@1 / pass@k) and per-problem library-reuse statistics
        results = score_responses(
            responses=responses,
            benchmark=benchmark,
            env=env,
            timeout_seconds=timeout_seconds,
            install_failed=install_failed,
            k=k,
        )
        # pop and re-add the per-response detail so "environment" is saved above it, keeping
        # the summary stats together before the detailed breakdown
        per_response_results = results.pop("results")
        results["environment"] = {
            "directory": str(env.cache_dir),
            "source": str(env_file) if env_file is not None else "response_imports",
            "installed_packages": sorted(packages),
            "installed_package_versions": env.installed_distributions(),
            "package_imports": {
                package: sorted(plan.get(package, set()))
                for package in sorted(packages)
            },
        }
        results["results"] = per_response_results

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"libreusebench_eval_{datetime.now().isoformat()}.json"
    # save the file's own name last, so the saved json is self-describing
    results["results_file"] = str(file_path)
    save_json(data=results, file_path=str(file_path))

    summary = f"pass@1 {results['pass@1']:.3f}"
    if results["k"] > 1:
        summary += f" | pass@{results['k']} {results['pass@k']:.3f}"
    lib_imported = results["library_use"]["imported"]["per_task"]["using_lib_rate"]
    lib_used = results["library_use"]["used"]["per_task"]["using_lib_rate"]
    summary += f" | lib-imported {lib_imported:.3f} | lib-used {lib_used:.3f}"
    code_all = results["code_metrics"]["all"]
    if code_all["n_measured"]:
        summary += (
            f" | lloc {code_all['avg_lloc']:.1f}"
            f" | mi {code_all['avg_maintainability_index']:.1f}"
        )
    print(f"{summary} | saved {file_path}")
    return results


def evaluate_responses_file(
    responses_file: str,
    output_dir: str = "output",
    env_dir: str | None = None,
    env_root: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    env_file: str | None = None,
    k: int | None = None,
    confirm: Callable[[dict[str, set[str]]], bool] | None = confirm_packages,
) -> dict:
    """Evaluate responses loaded from a .json or .jsonl file (see load_responses).

    A thin file-loading wrapper around evaluate_responses, for callers that keep
    responses on disk (e.g. the libreuseeval CLI) rather than already holding them in
    memory - experiment code generating responses in-process should call
    evaluate_responses directly instead, to skip the round trip through a file.

    Returns the results dict (see evaluate_responses).
    """
    return evaluate_responses(
        responses=load_responses(responses_file),
        output_dir=output_dir,
        env_dir=env_dir,
        env_root=env_root,
        timeout_seconds=timeout_seconds,
        env_file=env_file,
        k=k,
        confirm=confirm,
    )
