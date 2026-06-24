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

        assert result.code.strip() == "line = input()\nprint(line)"
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
