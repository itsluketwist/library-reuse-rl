"""Score generated eval responses per task type (no model runtime, so unit-testable).

The eval loop (src/eval/evaluate.py) generates one response per prompt; this module turns
those responses into per-row results and an analysis summary using the evaluator explicitly
configured beside the dataset. Tested-implementation datasets execute code, product-choice
and library-usage use the deterministic checks their rewards use, and BigCodeBenchReuse
delegates to its official scorer.
"""

import dataclasses
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bcbreuse.evaluate import evaluate_responses as evaluate_bcb_responses

from src.grpo.rewards.library_usage import analyse_library_usage
from src.response.parser import parse_response, strip_reasoning
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
    environment_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score tested-implementation rows by executing their code (the original eval path).

    Returns the (analysis, results) pair.
    """
    del environment_dir
    results = []
    n_passed = 0
    for row, (response, stop_reason) in zip(dataset, responses):
        processed = process_response(
            response=response,
            record={
                "inputs": row["inputs"],
                "expected_outputs": row["expected_outputs"],
            },
            python_executable=python_executable,
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
                "stop_reason": stop_reason,
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
    environment_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score product-choice rows by the prose pick (external tool vs DIY vs none).

    Returns the (analysis, results) pair.
    """
    del python_executable, environment_dir
    results = []
    for row, (response, stop_reason) in zip(dataset, responses):
        _, answer = strip_reasoning(response)
        pick = extract_primary_pick(
            response=answer,
            allowlist=row.get("allowlist") or [],
            aliases=row.get("aliases") or {},
            diy_terms=row.get("diy_terms") or [],
        )
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
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
    }
    return analysis, results


def _score_library_usage(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
    python_executable: str | None = None,
    environment_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score library-usage rows by the imports (external reuse vs stdlib/local).

    Returns the (analysis, results) pair.
    """
    del python_executable, environment_dir
    results = []
    for row, (response, stop_reason) in zip(dataset, responses):
        parsed = parse_response(response)
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
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
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
        )

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
        "external_lib_used_rate_by_phrasing_mode": _per_phrasing_mode_rate(
            results, "external_lib_used"
        ),
        "appropriate_lib_rate_by_phrasing_mode": _per_phrasing_mode_rate(
            results, "appropriate_lib_used"
        ),
    }
    return analysis, results


def _score_bcbreuse(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
    python_executable: str | None = None,
    environment_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Run the complete official BigCodeBenchReuse evaluation workflow.

    The official evaluator discovers packages from the responses, installs them into
    its cached environment, executes the selected code, and aggregates the results.
    The evaluator installs its discovered package plan automatically.

    Returns the official analysis and per-response results.
    """
    del python_executable
    rows = list(dataset)
    response_records = [
        {"task_id": row["task_id"], "response": response}
        for row, (response, _stop_reason) in zip(rows, responses)
    ]
    stop_reasons = {
        (row["task_id"], 0): stop_reason
        for row, (_response, stop_reason) in zip(rows, responses)
    }

    with tempfile.TemporaryDirectory(prefix="bcbreuse-eval-") as temp_dir:
        responses_file = Path(temp_dir) / "responses.jsonl"
        save_jsonl(response_records, responses_file)
        official = evaluate_bcb_responses(
            responses_file=str(responses_file),
            output_directory=temp_dir,
            env_dir=environment_dir or ".bcbeval-venv",
        )

    if not official:
        raise RuntimeError(
            "BigCodeBenchReuse could not prepare the generated responses' package "
            "environment; see the evaluator log above for failed packages."
        )

    results = official.pop("results")
    for result in results:
        result["stop_reason"] = stop_reasons[(result["task_id"], result["idx"])]

    return {
        "task": "bcbreuse",
        "n_total": official["n_responses"],
        **official,
    }, results


_SCORERS = {
    "implement_tested": _score_tested,
    "product_choice": _score_product_choice,
    "library_usage": _score_library_usage,
    "bcbreuse": _score_bcbreuse,
}


def score_dataset(
    dataset: Iterable[dict],
    responses: list[tuple[str, str]],
    evaluator: str,
    python_executable: str | None = None,
    environment_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score a whole eval dataset with its configured evaluator.

    Returns the (analysis, results) pair for the dataset's task.
    """
    rows = list(dataset)
    if not rows:
        return {"task": evaluator, "n_total": 0}, []

    if evaluator not in _SCORERS:
        raise ValueError(
            f"Unknown evaluator '{evaluator}'. Available: {sorted(_SCORERS)}"
        )
    return _SCORERS[evaluator](
        rows,
        responses,
        python_executable,
        environment_dir,
    )
