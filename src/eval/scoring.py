"""Score generated eval responses per task type (no model runtime, so unit-testable).

The eval loop (src/eval/evaluate.py) generates one response per prompt; this module turns
those responses into per-row results and an analysis summary using the evaluator explicitly
configured beside the dataset. Tested-implementation datasets execute code, product-choice
and library-usage use the deterministic checks their rewards use, and LibReuseBench
delegates to its official scorer.
"""

import dataclasses
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from libreusebench.evaluate import evaluate_responses as evaluate_lrb_responses

from src.grpo.rewards.library_usage import analyse_library_usage
from src.response.parser import parse_response
from src.response.pick import extract_primary_pick
from src.response.pipeline import process_response
from src.utils.io import save_jsonl


def _rate(
    count: int,
    total: int,
) -> float:
    """Fraction count/total, or 0.0 when there is nothing to divide.

    Returns the rate.
    """
    return count / total if total else 0.0


def _average(
    values: list[float],
) -> float | None:
    """Average a list of numbers.

    Returns the mean, or None if there are no values to average.
    """
    return sum(values) / len(values) if values else None


def _per_phrasing_mode_rate(
    results: list[dict],
    flag: str,
) -> dict[str, float]:
    """Rate of `flag` being true, broken down by bare/repo phrasing mode.

    Returns a {phrasing_mode: rate} mapping.
    """
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for result in results:
        mode = result.get("phrasing_mode") or "?"
        totals[mode] += 1
        if result.get(flag):
            hits[mode] += 1
    return {mode: hits[mode] / totals[mode] for mode in sorted(totals)}


def _library_statistics(
    library_sets: list[set[str]],
) -> dict[str, Any]:
    """Summarise distinct external libraries used by each response.

    Returns reuse rate, average library count, and per-library response counts.
    """
    counts: Counter[str] = Counter()
    for libraries in library_sets:
        counts.update(libraries)

    return {
        "using_lib_rate": _rate(
            sum(bool(libs) for libs in library_sets),
            len(library_sets),
        ),
        "avg_extlibs": _average([len(libs) for libs in library_sets]),
        "library_counts": dict(counts.most_common()),
    }


def _score_tested(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score tested-implementation rows by executing their code (the original eval path).

    Returns the (analysis, results) pair.
    """
    del environment_root, environment_cache_dir
    # each result is a heterogeneous record (strs, bools, lists, ...), so type as Any
    results: list[dict[str, Any]] = []
    n_passed = 0
    for row, (response, stop_reason) in zip(dataset, responses):
        processed = process_response(
            response=response,
            record={
                "inputs": row["inputs"],
                "expected_outputs": row["expected_outputs"],
            },
            python_executable=python_executable,
            stop_reason=stop_reason,
        )
        n_passed += processed.passed
        processed_data = dataclasses.asdict(processed)
        # candidate properties are derived from the marked block and therefore are not
        # dataclass fields; retain the established flat result schema alongside blocks.
        processed_data.update(
            {
                "code": processed.code,
                "valid": processed.valid,
                "error": processed.error,
                "std_libs": processed.std_libs,
                "ext_libs": processed.ext_libs,
                "lib_usage": processed.lib_usage,
                "logical_lines": processed.logical_lines,
                "maintainability_index": processed.maintainability_index,
            }
        )
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                **processed_data,
            }
        )

    n_total = len(results)
    # average code metrics over valid-code responses only, so missing/unparseable code
    # doesn't skew them (matches the original inline eval analysis)
    valid = [r for r in results if r.get("valid") is True]
    analysis = {
        "task": "implement_tested",
        "n_total": n_total,
        "n_passed": n_passed,
        "accuracy": _rate(n_passed, n_total),
        "n_reasoning": sum(1 for r in results if r.get("reasoning")),
        "n_answer": sum(1 for r in results if r.get("answer")),
        "n_overflow": sum(1 for r in results if r.get("stop_reason") == "length"),
        "n_valid_code": len(valid),
        "avg_std_libs": _average([len(r["std_libs"]) for r in valid]),
        "avg_ext_libs": _average([len(r["ext_libs"]) for r in valid]),
        "avg_ll": _average(
            [r["logical_lines"] for r in valid if r["logical_lines"] is not None]
        ),
        "avg_mi": _average(
            [
                r["maintainability_index"]
                for r in valid
                if r["maintainability_index"] is not None
            ]
        ),
    }
    return analysis, results


def _score_product_choice(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score product-choice rows by the prose pick (external tool vs DIY vs none).

    Returns the (analysis, results) pair.
    """
    del python_executable, environment_root, environment_cache_dir
    results = []
    for row, (response, stop_reason) in zip(dataset, responses):
        parsed = parse_response(response=response, stop_reason=stop_reason)
        pick = extract_primary_pick(
            response=parsed.answer or "",
            allowlist=row.get("allowlist") or [],
            aliases=row.get("aliases") or {},
            diy_terms=row.get("diy_terms") or [],
        )
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "stop_reason": parsed.stop_reason,
                "category_id": row.get("category_id"),
                "primary": pick.primary,
                "external_pick": pick.is_external,
                "diy_pick": pick.is_diy,
                "mentioned_external": pick.mentioned_external,
            }
        )

    n_total = len(results)
    n_external = sum(1 for r in results if r["external_pick"])
    n_diy = sum(1 for r in results if r["diy_pick"])
    analysis = {
        "task": "product_choice",
        "n_total": n_total,
        "external_pick_rate": _rate(n_external, n_total),
        "diy_rate": _rate(n_diy, n_total),
        "no_pick_rate": _rate(n_total - n_external - n_diy, n_total),
    }
    return analysis, results


def _task_library_use(
    task_results: list[list[dict]],
) -> dict[str, Any]:
    """Per-task library-use stats: a task counts as using a library if any of its
    sampled responses did (see MULTI_SAMPLE_EVALUATORS).

    Returns the same shape as _library_statistics, plus appropriate-use rates.
    """
    used_sets = [
        set().union(*(r["used_external_libs"] for r in task)) if task else set()
        for task in task_results
    ]
    appropriate_hits = [
        any(r["appropriate_lib_used"] for r in task) for task in task_results
    ]
    stats = _library_statistics(used_sets)
    stats["appropriate_lib_rate"] = _rate(sum(appropriate_hits), len(task_results))
    stats["appropriate_given_use_rate"] = _rate(
        sum(appropriate_hits),
        sum(bool(used) for used in used_sets),
    )
    return stats


def _score_library_usage(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score library-usage rows by the imports (external reuse vs stdlib/local).

    Each row may carry multiple sampled responses (see eval_decoding_profiles'
    n_samples); every sample is scored and kept as its own result row, tagged with
    `sample_idx`. The `library_use`/rate fields in `analysis` are per response (the
    existing behaviour, now spanning every sample); `library_use_per_task` adds the
    per-task view (a task "uses a library" if any of its samples did).

    Returns the (analysis, results) pair.
    """
    del python_executable, environment_root, environment_cache_dir
    # each result is a heterogeneous record (strs, bools, lists, ...), so type as Any
    results: list[dict[str, Any]] = []
    task_results: list[list[dict]] = []
    for row, samples in zip(dataset, responses):
        row_results = []
        for sample_idx, (response, stop_reason) in enumerate(samples):
            parsed = parse_response(response=response, stop_reason=stop_reason)
            usage = analyse_library_usage(parsed, row.get("local_modules"))
            external = usage.external_libs
            used = usage.used_external_libs
            useful = set(row.get("useful_libs") or [])
            appropriate = sorted(lib for lib in used if lib in useful)
            all_blocks_valid = (
                bool(usage.n_code_blocks)
                and usage.n_valid_code_blocks == usage.n_code_blocks
            )
            # one working code block is enough to count the response as valid
            any_block_valid = usage.n_valid_code_blocks > 0
            result = {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "sample_idx": sample_idx,
                "stop_reason": stop_reason,
                "category_id": row.get("category_id"),
                "valid": any_block_valid,
                "code_blocks": [
                    dataclasses.asdict(block) for block in parsed.code_blocks
                ],
                "n_code_blocks": usage.n_code_blocks,
                "n_valid_code_blocks": usage.n_valid_code_blocks,
                "all_code_blocks_valid": all_blocks_valid,
                "external_libs": external,
                "external_lib": bool(external),
                "used_external_libs": used,
                "external_lib_used": bool(used),
                "appropriate_libs": appropriate,
                "appropriate_lib_used": bool(appropriate),
                "phrasing_mode": row.get("phrasing_mode"),
            }
            results.append(result)
            row_results.append(result)
        task_results.append(row_results)

    n_total = len(results)
    n_used = sum(r["external_lib_used"] for r in results)
    n_appropriate = sum(r["appropriate_lib_used"] for r in results)
    library_use = _library_statistics(
        [set(result["used_external_libs"]) for result in results]
    )
    library_use["appropriate_lib_rate"] = _rate(n_appropriate, n_total)
    library_use["appropriate_given_use_rate"] = _rate(n_appropriate, n_used)
    analysis = {
        "task": "library_usage",
        "n_total": n_total,
        "n_valid_code": sum(r["valid"] for r in results),
        "n_external_lib_used": n_used,
        "n_appropriate_lib_used": n_appropriate,
        "valid_code_rate": _rate(sum(r["valid"] for r in results), n_total),
        "external_lib_rate": _rate(sum(r["external_lib"] for r in results), n_total),
        "external_lib_used_rate": _rate(n_used, n_total),
        "appropriate_lib_rate": _rate(n_appropriate, n_total),
        "library_use": library_use,
        "library_use_per_task": _task_library_use(task_results),
        "external_lib_used_rate_by_phrasing_mode": _per_phrasing_mode_rate(
            results, "external_lib_used"
        ),
        "appropriate_lib_rate_by_phrasing_mode": _per_phrasing_mode_rate(
            results, "appropriate_lib_used"
        ),
    }
    return analysis, results


def _score_libreusebench(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Run the complete official LibReuseBench evaluation workflow.

    Each row may carry multiple sampled responses (see eval_decoding_profiles'
    n_samples); every sample is submitted for the task, so the official evaluator's
    pass@k (see benchmark/libreusebench/evaluate.py's _pass_at_k) reflects them.

    The official evaluator discovers packages from the responses, installs them into
    its cached environment, executes the selected code, and aggregates the results.
    Eval runs are unattended (including on HPC/Slurm), so this passes `confirm=None`
    to install the discovered package plan automatically rather than prompting on stdin.

    Returns the official analysis and per-response results.
    """
    del python_executable
    rows = list(dataset)
    response_records = [
        {"task_id": row["task_id"], "response": response}
        for row, samples in zip(rows, responses)
        for response, _stop_reason in samples
    ]
    stop_reasons = {
        (row["task_id"], sample_idx): stop_reason
        for row, samples in zip(rows, responses)
        for sample_idx, (_response, stop_reason) in enumerate(samples)
    }

    with tempfile.TemporaryDirectory(prefix="libreusebench-eval-") as temp_dir:
        responses_file = Path(temp_dir) / "responses.jsonl"
        save_jsonl(response_records, responses_file)
        official = evaluate_lrb_responses(
            responses_file=str(responses_file),
            output_dir=temp_dir,
            # without a cache dir the evaluator builds a disposable venv per run and
            # deletes it, so concurrently running eval jobs can never corrupt a shared one
            env_dir=environment_cache_dir,
            env_root=environment_root,
            confirm=None,
        )

    if not official:
        raise RuntimeError(
            "LibReuseBench could not prepare the generated responses' package "
            "environment; see the evaluator log above for failed packages."
        )

    results = official.pop("results")
    for result in results:
        result["stop_reason"] = stop_reasons[(result["task_id"], result["idx"])]

    return {
        "task": "libreusebench",
        "n_total": official["n_responses"],
        **official,
    }, results


# split by responses shape so each dict stays internally homogeneous (a type checker can't
# verify a single dict mixing both signatures against a single call site)
_SINGLE_SAMPLE_SCORERS = {
    "implement_tested": _score_tested,
    "product_choice": _score_product_choice,
}
_MULTI_SAMPLE_SCORERS = {
    "library_usage": _score_library_usage,
    "libreusebench": _score_libreusebench,
}
_SCORERS = {**_SINGLE_SAMPLE_SCORERS, **_MULTI_SAMPLE_SCORERS}

# evaluators that score every sampled response per row (see eval_decoding_profiles'
# n_samples); the rest only ever score the first sample - score_dataset takes care of
# that split itself, so callers never need to know which evaluators are multi-sample
MULTI_SAMPLE_EVALUATORS = set(_MULTI_SAMPLE_SCORERS)


def score_dataset(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    evaluator: str,
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score a whole eval dataset with its configured evaluator.

    `responses` holds every sampled response per row (see generate_vllm_responses'
    n_samples support). MULTI_SAMPLE_EVALUATORS score every sample; the rest only ever
    look at the first, which this takes care of internally.

    Datasets that execute code build a venv to run it in: `environment_root` is where a
    disposable one goes, and `environment_cache_dir` names one to reuse across runs
    instead (see libreusebench.environment.Environment).

    Returns the (analysis, results) pair for the dataset's task.
    """
    rows = list(dataset)
    if not rows:
        return {"task": evaluator, "n_total": 0}, []

    if evaluator in _MULTI_SAMPLE_SCORERS:
        return _MULTI_SAMPLE_SCORERS[evaluator](
            rows, responses, python_executable, environment_root, environment_cache_dir
        )
    if evaluator in _SINGLE_SAMPLE_SCORERS:
        single_responses = [samples[0] for samples in responses]
        return _SINGLE_SAMPLE_SCORERS[evaluator](
            rows,
            single_responses,
            python_executable,
            environment_root,
            environment_cache_dir,
        )
    raise ValueError(f"Unknown evaluator '{evaluator}'. Available: {sorted(_SCORERS)}")
