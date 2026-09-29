"""Combine parsing and evaluation for a single response."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lure import CodeBlock, ParsedResponse, parse_response

from src.response.evaluate import evaluate_record


@dataclass(frozen=True)
class ProcessedResponse:
    """
    A parsed response plus the outcome of running its code against a record's tests.

    The program that is run is the answer's longest python code block (see
    `executed_block`), since a response may also contain short usage examples.
    """

    parsed: ParsedResponse
    passed: bool
    # captured stdout or error message, None if there was no code to run
    output: str | None

    @property
    def executed_block(
        self,
    ) -> CodeBlock | None:
        """
        The code block that was run against the tests.

        Returns the longest code block, or None when the answer has no code.
        """
        return self.parsed.longest_block


def process_response(
    response: str,
    record: dict[str, Any],
    python_executable: str | Path | None = None,
    *,
    stop_reason: str = "unknown",
) -> ProcessedResponse:
    """
    Parse a response and run its longest code block against a record's tests.

    Returns a ProcessedResponse with the parsed response and the evaluation outcome.
    """
    parsed = parse_response(response, stop_reason=stop_reason)
    block = parsed.longest_block

    # no code extracted means there is nothing to evaluate - the record fails
    # without an output, rather than running the empty string against the tests
    if block is None:
        return ProcessedResponse(parsed=parsed, passed=False, output=None)

    passed, output = evaluate_record(
        code=block.code,
        record=record,
        python_executable=python_executable,
    )
    return ProcessedResponse(parsed=parsed, passed=passed, output=output)
