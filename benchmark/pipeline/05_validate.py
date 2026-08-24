"""Stage 05: confirm dual-solvability with an external-library gold solution.

Stage 04 produced a standard-library gold and its test cases. Here we generate an independent
solution that reaches for a third-party library (the natural, reuse way) and require it to
reproduce exactly the same outputs. If it does, the problem is genuinely solvable both ways - a
real, measurable library-reuse choice - and the two independent solutions agreeing cross-validates
the ground truth. We also reject any prompt that still names a specific library.

The model declares the PyPI packages its solution needs (import names often differ from pip
names, e.g. bs4 -> beautifulsoup4); we install those into the isolated build venv, run the tests,
and uninstall them again, so each problem is evaluated with exactly its own libraries.

Apparent failures are dominated by sampling variance, not model capability, so we draw best-of-N
(--attempts): a task is kept as soon as any sample reproduces the outputs. The run is resumable
(only non-kept tasks are retried) and supports --limit / --model. Point LRB_BUILD_PYTHON at the
seeded build venv (benchmark/pipeline/setup_build_venv.sh).

Usage: python benchmark/pipeline/05_validate.py [--limit N] [--model M] [--attempts N]
"""

import argparse

from build import BUILD_PYTHON, artefact_path, succeeded_ids
from generation import DEFAULT_GEN_MODEL, complete_json
from libreusebench.environment import evaluate_record, temporary_packages
from libreusebench.libraries import analyse_source, find_named_libraries

from src.utils.io import append_jsonl, load_jsonl


_LIB_SYSTEM = (
    "You are an expert Python programmer. You solve stdin/stdout problems the idiomatic way, "
    "reaching for a suitable third-party library. Reply with JSON only."
)

_LIB_PROMPT = """\
Solve this task in Python: read all input from standard input and print the result to
standard output, exactly as the task specifies. Solve it the idiomatic way a developer would,
using available libraries - you MUST use at least one third-party/pip library (for example
numpy, pandas, dateutil, beautifulsoup4, ...) where it helps.

{prompt}

Reply with JSON only:
{{"solution": "<python source>",
  "pip_packages": ["<exact PyPI package name to install for each third-party library you import,
  e.g. beautifulsoup4 for bs4, scikit-learn for sklearn, python-dateutil for dateutil>"]}}
"""

_LIB_SCHEMA = {
    "type": "object",
    "properties": {
        "solution": {"type": "string"},
        "pip_packages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["solution", "pip_packages"],
    "additionalProperties": False,
}


def build_lib_prompt(
    prompt: str,
) -> str:
    """Build the library-solution prompt for one task.

    Returns the prompt string.
    """
    return _LIB_PROMPT.format(prompt=prompt.strip())


def classify_failure(
    message: str,
) -> str:
    """Bucket an execution failure so install / logic / crash / timeout are told apart.

    The point is diagnosis without reruns: an install failure or a timeout is transient/fixable,
    an output mismatch is a genuine behavioural disagreement (maybe not dual-solvable), and a
    runtime error is usually a bug in the generated solution worth retrying with a stronger model.

    Returns a short, stable reason category.
    """
    if message.startswith("install failed"):
        return "install failed"
    if message.startswith("timeout"):
        return "timeout"
    # our test harness asserts on stdout, so an AssertionError means it ran but disagreed
    if "AssertionError" in message and "test " in message:
        return "output mismatch"
    return "runtime error"


def failure_detail(
    message: str,
) -> str:
    """Pull the informative tail out of a failure message (the assertion diff or exception line).

    Returns a trimmed detail string kept on the record for later inspection.
    """
    lines = [line for line in message.strip().splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


def library_solution(
    prompt: str,
    model: str = DEFAULT_GEN_MODEL,
) -> tuple[str, list[str], str] | None:
    """Generate a library solution and its PyPI packages.

    A single sample - the caller draws several (best-of-N) since re-sampling recovers most
    apparent failures, which are dominated by sampling variance rather than model capability.

    Returns a (solution, pip_packages, resolved model id) tuple, or None if nothing usable.
    """
    data, resolved_model = complete_json(
        prompt=build_lib_prompt(prompt),
        schema=_LIB_SCHEMA,
        system=_LIB_SYSTEM,
        model=model,
    )
    solution = str(data.get("solution", "")).strip()
    packages = [
        str(p) for p in data.get("pip_packages", []) if isinstance(p, str) and p.strip()
    ]
    if not solution:
        return None
    return solution, packages, resolved_model


def validate_once(
    row: dict,
    model: str,
) -> dict:
    """Draw one library solution for a task, install it, and run it against the stdlib gold tests.

    Returns a result dict: always carries 'passed'; on success the gold_extlib solution and its
    version-pinned packages; on failure a drop_reason, a diagnostic detail, and the attempted
    solution so the drop can be inspected without regenerating it.
    """
    result = library_solution(row["prompt"], model=model)
    if not result:
        return {"passed": False, "drop_reason": "no library solution"}

    solution, pip_packages, library_model = result
    library_libs = analyse_source(solution)[1]
    # keep the attempt on the record (success or failure) so a drop is inspectable later; on a drop
    # the packages are the model's raw declared names (no versions - nothing was validated)
    attempt = {
        "gold_extlib": solution,
        "gold_extlib_model": library_model,
        "gold_extlib_packages": pip_packages,
    }
    if not library_libs:
        return {"passed": False, "drop_reason": "solution used no library"} | attempt

    # install the declared packages into the isolated build venv, run the tests, then fully
    # uninstall - recording the exact versions the library solution was validated against
    with temporary_packages(BUILD_PYTHON, pip_packages) as install:
        versions = install.versions
        if install.failed:
            passed, message = False, f"install failed: {sorted(install.failed)}"
        else:
            passed, message = evaluate_record(
                program=solution,
                inputs=row["inputs"],
                expected_outputs=row["expected_outputs"],
                python_executable=BUILD_PYTHON,
            )

    if passed:
        # pin each installed package to the exact version it was validated against
        pinned = sorted(f"{name}=={version}" for name, version in versions.items())
        return {"passed": True} | attempt | {"gold_extlib_packages": pinned}
    # a granular failure kind (not a generic "disagrees") so reruns target only what's worth retrying
    return {
        "passed": False,
        "drop_reason": classify_failure(message),
        "detail": failure_detail(message),
    } | attempt


def main() -> None:
    """Keep only problems whose prompt is agnostic and that a library solution also solves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="only validate the first N"
    )
    parser.add_argument(
        "--model", default=DEFAULT_GEN_MODEL, help="generation model id"
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="best-of-N: draw up to N solutions, keep the task on the first that passes",
    )
    args = parser.parse_args()

    rows = [
        row for row in load_jsonl(artefact_path("04_tests.jsonl")) if row.get("kept")
    ]
    out_path = artefact_path("05_validated.jsonl")
    done = succeeded_ids(out_path)
    print(
        f"{len(rows)} problems to validate"
        + (f", {len(done)} already done" if done else "")
        + (f", best-of-{args.attempts}" if args.attempts > 1 else "")
    )

    processed = kept = 0
    for row in rows:
        seed_id = row["seed_id"]
        if seed_id in done:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        # gate 1: the prompt must not leak a specific library name (deterministic - no point retrying)
        leaked = find_named_libraries(row["prompt"])
        if leaked:
            append_jsonl(
                {
                    "seed_id": seed_id,
                    "kept": False,
                    "drop_reason": f"prompt names {leaked}",
                },
                out_path,
            )
            continue

        # gate 2/3: draw up to N solutions, stopping at the first that reproduces the gold outputs
        outcome: dict = {}
        for _ in range(max(1, args.attempts)):
            outcome = validate_once(row, model=args.model)
            if outcome["passed"]:
                break

        fields = {key: value for key, value in outcome.items() if key != "passed"}
        if not outcome["passed"]:
            append_jsonl({"seed_id": seed_id, "kept": False} | fields, out_path)
            continue

        kept += 1
        append_jsonl(row | {"kept": True} | fields, out_path)
        print(
            f"validated {processed} | kept {kept} | pkgs {fields['gold_extlib_packages']} | last {seed_id}"
        )

    print(f"\ndone: processed {processed}, kept {kept}")


if __name__ == "__main__":
    main()
