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

    Inherits ParsedResponse's complete block collection. Candidate properties such as
    `.code` and `.valid` resolve to the block marked for tests.
    """

    passed: bool
    output: (
        str | None
    )  # captured stdout or error message, None if there was no code to run


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

    # no code extracted means there is nothing to evaluate - the record fails
    # without an output, rather than running the empty string against the tests
    if parsed.code is None:
        passed: bool = False
        output: str | None = None
    else:
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
