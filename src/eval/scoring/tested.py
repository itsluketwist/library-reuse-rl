"""Score tested-implementation eval responses: execute the code against its test cases."""

from typing import Any, Iterable

from src.eval.scoring.base import _average, _rate
from src.response.metrics import code_metrics
from src.response.pipeline import process_response


def score_tested(
    dataset: Iterable[dict],
    responses: list[tuple[str, str, int | None]],
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
    for row, (response, stop_reason, completion_tokens) in zip(dataset, responses):
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
        parsed = processed.parsed
        block = processed.executed_block
        logical_lines, maintainability_index = (
            code_metrics(block.code) if block is not None else (None, None)
        )
        # the response in full plus its scores. the extracted code is not stored, as
        # lure.parse_response(response) re-extracts it; the libraries, error and metrics
        # below describe the block that was run (the longest one)
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "stop_reason": stop_reason,
                "completion_tokens": completion_tokens,
                "response": response,
                "passed": processed.passed,
                "output": processed.output,
                "has_reasoning": bool(parsed.reasoning),
                "has_answer": bool(parsed.answer),
                "n_code_blocks": len(parsed.code_blocks),
                # the answer has code and all of it parses, like passes_syntax_check
                "valid": parsed.valid,
                "error": block.error if block is not None else None,
                "std_libs": sorted(block.stdlib_imports) if block is not None else [],
                "ext_libs": sorted(block.external_imports) if block is not None else [],
                "logical_lines": logical_lines,
                "maintainability_index": maintainability_index,
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
