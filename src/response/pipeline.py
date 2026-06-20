"""Combine parsing, evaluation, and reward computation for a single response."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.response.evaluate import evaluate_record
from src.response.parser import ParsedResponse, parse_response
from src.response.reward import ALL_REWARD_BITS, RewardBit, compute_reward


@dataclass
class ProcessedResponse:
    """
    The full result of processing a single LLM response against a record.
    """

    parsed: ParsedResponse
    passed: bool
    output: str  # captured stdout, or an error message if evaluation failed
    reward: float


def process_response(
    response: str,
    record: dict[str, Any],
    python_executable: str | Path | None = None,
    reward_bits: RewardBit = ALL_REWARD_BITS,
) -> ProcessedResponse:
    """
    Parse a response, evaluate its code against a record, and compute its reward.

    `reward_bits` is forwarded to compute_reward, so a caller (e.g. the training
    cli) can choose which reward components are switched on for a given run.

    Returns a ProcessedResponse with the parsed response, evaluation outcome, and reward.
    """
    parsed = parse_response(response=response)
    passed, output = evaluate_record(
        code=parsed.code,
        record=record,
        python_executable=python_executable,
    )
    reward = compute_reward(parsed=parsed, passed=passed, reward_bits=reward_bits)

    return ProcessedResponse(
        parsed=parsed,
        passed=passed,
        output=output,
        reward=reward,
    )
