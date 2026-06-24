"""Combine parsing and evaluation for a single response."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.response.evaluate import evaluate_record
from src.response.parser import ParsedResponse, parse_response


@dataclass
class ProcessedResponse(ParsedResponse):
    """
    A parsed response plus the outcome of evaluating its code against a record.

    Inherits ParsedResponse's fields directly (e.g. `.code`, `.valid`) rather than
    nesting them under a `.parsed` attribute, since a ProcessedResponse is always a
    parsed response, just with the evaluation outcome attached too.
    """

    passed: bool
    output: str  # captured stdout, or an error message if evaluation failed


def process_response(
    response: str,
    record: dict[str, Any],
    python_executable: str | Path | None = None,
) -> ProcessedResponse:
    """
    Parse a response and evaluate its code against a record.

    Returns a ProcessedResponse with the parsed response's fields and the evaluation outcome.
    """
    parsed = parse_response(response=response)
    passed, output = evaluate_record(
        code=parsed.code,
        record=record,
        python_executable=python_executable,
    )

    return ProcessedResponse(
        **vars(parsed),
        passed=passed,
        output=output,
    )
