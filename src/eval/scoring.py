"""Score generated eval responses per task type (no model runtime, so unit-testable).

The eval loop (src/eval/evaluate.py) generates one response per prompt; this module turns
those responses into per-row results and an analysis summary, dispatching on the dataset's
task. The tested-implementation datasets are scored by executing the code (as before); the
new product-choice and library-usage datasets are scored by the same deterministic checks
their rewards use - the prose pick and the import classification - so eval metrics line up
with what training optimised.
"""

import dataclasses
from collections import Counter
from typing import Any, Iterable

from src.grpo.rewards.base import CodeReward
from src.response.parser import parse_response, strip_reasoning
from src.response.pick import extract_primary_pick
from src.response.pipeline import process_response


def detect_task(
    column_names: list[str],
    first_row: dict[str, Any],
) -> str:
    """Infer a dataset's task from its columns/first row.

    The tested-implementation datasets carry stdin/stdout test columns; the new datasets
    carry an explicit task_type.

    Returns "implement_tested", "product_choice", or "library_usage".
    """
    if "inputs" in column_names and "expected_outputs" in column_names:
        return "implement_tested"
    return first_row.get("task_type", "implement_tested")


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


def _per_category_rate(
    results: list[dict],
    flag: str,
) -> dict[str, float]:
    """Rate of `flag` being true, broken down by category_id.

    Returns a {category_id: rate} mapping.
    """
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for result in results:
        category = result.get("category_id", "?")
        totals[category] += 1
        if result.get(flag):
            hits[category] += 1
    return {category: hits[category] / totals[category] for category in sorted(totals)}


def _score_tested(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
) -> tuple[dict, list[dict]]:
    """Score tested-implementation rows by executing their code (the original eval path).

    Returns the (analysis, results) pair.
    """
    results = []
    n_passed = 0
    for row, (response, stop_reason) in zip(dataset, responses):
        processed = process_response(
            response=response,
            record={
                "inputs": row["inputs"],
                "expected_outputs": row["expected_outputs"],
            },
        )
        n_passed += processed.passed
        results.append(
            {
                "idx": row["idx"],
                "stop_reason": stop_reason,
                **dataclasses.asdict(processed),
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
) -> tuple[dict, list[dict]]:
    """Score product-choice rows by the prose pick (external tool vs DIY vs none).

    Returns the (analysis, results) pair.
    """
    results = []
    for row, (response, stop_reason) in zip(dataset, responses):
        _, answer = strip_reasoning(response)
        pick = extract_primary_pick(
            response=answer,
            allowlist=row.get("allowlist") or [],
            diy_terms=row.get("diy_terms") or [],
        )
        results.append(
            {
                "idx": row["idx"],
                "stop_reason": stop_reason,
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
        "external_pick_rate_by_category": _per_category_rate(results, "external_pick"),
    }
    return analysis, results


def _score_library_usage(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
) -> tuple[dict, list[dict]]:
    """Score library-usage rows by the imports (external reuse vs stdlib/local).

    Returns the (analysis, results) pair.
    """
    results = []
    for row, (response, stop_reason) in zip(dataset, responses):
        parsed = parse_response(response)
        external = CodeReward.external_libraries(parsed, row.get("local_modules"))
        used = [lib for lib in external if lib in parsed.lib_usage]
        useful = set(row.get("useful_libs") or [])
        results.append(
            {
                "idx": row["idx"],
                "stop_reason": stop_reason,
                "category_id": row.get("category_id"),
                "valid": parsed.valid,
                "external_libs": external,
                "external_lib": bool(external),
                "external_lib_used": bool(used),
                "appropriate_lib_used": any(lib in useful for lib in used),
            }
        )

    n_total = len(results)
    analysis = {
        "task": "library_usage",
        "n_total": n_total,
        "external_lib_rate": _rate(sum(r["external_lib"] for r in results), n_total),
        "external_lib_used_rate": _rate(
            sum(r["external_lib_used"] for r in results), n_total
        ),
        "appropriate_lib_rate": _rate(
            sum(r["appropriate_lib_used"] for r in results), n_total
        ),
        "external_lib_used_rate_by_category": _per_category_rate(
            results, "external_lib_used"
        ),
    }
    return analysis, results


_SCORERS = {
    "implement_tested": _score_tested,
    "product_choice": _score_product_choice,
    "library_usage": _score_library_usage,
}


def score_dataset(
    dataset: Iterable[dict],
    column_names: list[str],
    responses: list[tuple[str, str]],
) -> tuple[dict, list[dict]]:
    """Score a whole eval dataset, dispatching on its task type.

    Returns the (analysis, results) pair for the dataset's task.
    """
    rows = list(dataset)
    if not rows:
        return {"task": "empty", "n_total": 0}, []

    task = detect_task(column_names, rows[0])
    return _SCORERS[task](rows, responses)
