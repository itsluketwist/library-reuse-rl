"""Score tested-implementation eval responses: execute the code against its test cases."""

from typing import Any, Iterable

from src.eval.scoring.base import _average, _rate
from src.response.pipeline import process_response


def score_tested(
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
        # the response in full plus its scores. the extracted code is not stored, as
        # parse_response(response) re-extracts it; the properties below describe the block
        # marked for tests
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "stop_reason": processed.stop_reason,
                "response": response,
                "passed": processed.passed,
                "output": processed.output,
                "has_reasoning": bool(processed.reasoning),
                "has_answer": bool(processed.answer),
                "n_code_blocks": len(processed.code_blocks),
                "valid": processed.valid,
                "error": processed.error,
                "std_libs": processed.std_libs,
                "ext_libs": processed.ext_libs,
                "logical_lines": processed.logical_lines,
                "maintainability_index": processed.maintainability_index,
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
        "n_reasoning": sum(1 for r in results if r["has_reasoning"]),
        "n_answer": sum(1 for r in results if r["has_answer"]),
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
