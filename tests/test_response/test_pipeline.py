"""Tests for the combined parse/evaluate pipeline."""

import sys

from src.response.pipeline import process_response


PYTHON = sys.executable


class TestProcessResponse:
    """Tests for process_response."""

    def test_passing_response(self):
        """A response with correct code is parsed and passes evaluation."""
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

        assert result.executed_block is not None
        assert result.executed_block.code == "line = input()\nprint(line)"
        assert result.passed is True

    def test_failing_response(self):
        """A response with incorrect, but well-formatted, code fails evaluation."""
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

    def test_no_code_response(self):
        """A response with no extracted code is not evaluated at all."""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response="<think>ran out of tokens",
            record=record,
            python_executable=PYTHON,
        )

        assert result.executed_block is None
        assert result.passed is False
        assert result.output is None

    def test_executes_only_the_largest_code_block(self):
        """Test-backed evaluation treats the largest block as the candidate program."""
        response = """
```python
print("wrong")
```
```python
line = input()
result = line.strip()
print(result)
```
"""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response=response,
            record=record,
            python_executable=PYTHON,
        )

        assert result.executed_block is not None
        assert "result = line.strip()" in result.executed_block.code
        assert result.passed is True

    def test_longer_non_python_block_is_not_executed(self):
        """Only python code is run, even when a bash block is longer."""
        response = """
```bash
pip install nothing
echo one
echo two
echo three
```
```python
print(input())
```
"""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response=response,
            record=record,
            python_executable=PYTHON,
        )

        assert result.passed is True

    def test_stop_reason_is_passed_through(self):
        """stop_reason flows into the parsed response, marking it as truncated."""
        record = {"inputs": ["hello\n"], "expected_outputs": ["hello\n"]}

        result = process_response(
            response="<think>ran out of tokens",
            record=record,
            python_executable=PYTHON,
            stop_reason="length",
        )

        assert result.parsed.stop_reason == "length"
        assert result.parsed.truncated
