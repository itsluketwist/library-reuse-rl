"""Tests for the combined parse/evaluate/reward pipeline."""

import sys

from src.response.pipeline import process_response
from src.response.reward import REWARD_WEIGHTS, RewardBit


PYTHON = sys.executable


class TestProcessResponse:
    """Tests for process_response."""

    def test_passing_response(self):
        """A response with correct code is parsed, passes, and gets the full reward."""
        response = """
```python
line = input()
print(line)
```
"""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response=response, record=record, python_executable=PYTHON
        )

        assert result.parsed.code.strip() == "line = input()\nprint(line)"
        assert result.passed is True
        assert result.reward == (
            REWARD_WEIGHTS[RewardBit.CORRECT_FORMAT]
            + REWARD_WEIGHTS[RewardBit.PASSED_TESTS]
        )

    def test_failing_response(self):
        """A response with incorrect, but well-formatted, code only gets the format reward."""
        response = """
```python
print("wrong")
```
"""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response=response, record=record, python_executable=PYTHON
        )

        assert result.passed is False
        assert result.reward == REWARD_WEIGHTS[RewardBit.CORRECT_FORMAT]

    def test_reward_bits_restricts_reward_to_correctness_only(self):
        """Passing `reward_bits` lets a caller select e.g. a correctness-only reward."""
        response = """
```python
line = input()
print(line)
```
"""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response=response,
            record=record,
            python_executable=PYTHON,
            reward_bits=RewardBit.PASSED_TESTS,
        )

        assert result.reward == REWARD_WEIGHTS[RewardBit.PASSED_TESTS]
