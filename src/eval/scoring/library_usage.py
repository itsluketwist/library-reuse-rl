"""Score library-usage eval responses: the imports (external reuse vs stdlib/local)."""

from collections import Counter
from typing import Any, Iterable

from lure import is_appropriate_library, parse_response

from src.eval.scoring.base import _library_statistics, _rate


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


def score_library_usage(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str, int | None]]],
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
        for sample_idx, (response, stop_reason, completion_tokens) in enumerate(
            samples
        ):
            # the same parsing and library analysis (lure) as the training reward
            parsed = parse_response(
                response,
                stop_reason=stop_reason,
                local_modules=row.get("local_modules") or (),
            )
            external = sorted(parsed.external_imports)
            used = sorted(parsed.external_used)
            appropriate = [
                lib
                for lib in used
                if is_appropriate_library(lib, row.get("useful_libs") or ())
            ]
            n_valid_blocks = sum(block.valid for block in parsed.code_blocks)
            result = {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "sample_idx": sample_idx,
                "stop_reason": stop_reason,
                "completion_tokens": completion_tokens,
                "response": response,
                "category_id": row.get("category_id"),
                # valid means the answer has code and all of it parses, matching the
                # training reward's passes_syntax_check
                "valid": parsed.valid,
                # the code blocks themselves are not stored - parse_response(response)
                # re-extracts them - only what was found in them
                "n_code_blocks": len(parsed.code_blocks),
                "n_valid_code_blocks": n_valid_blocks,
                "any_code_block_valid": n_valid_blocks > 0,
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
