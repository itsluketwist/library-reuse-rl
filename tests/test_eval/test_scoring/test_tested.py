"""Tests for scoring the tested-implementation split."""

import sys

from lure import parse_response

from src.eval.scoring import score_dataset


def _responses(*texts: str) -> list[list[tuple[str, str]]]:
    """One sampled response per row, each paired with a 'stop' stop-reason."""
    return [[(text, "stop")] for text in texts]


class TestScoreTestedImplementation:
    """Test-backed scoring exposes all blocks and its selected candidate."""

    def test_saves_blocks_and_executes_the_marked_largest_block(self):
        """The flat candidate fields and complete block analysis remain recoverable."""
        dataset = [
            {
                "idx": 0,
                "inputs": ["hello\n"],
                "expected_outputs": ["hello\n"],
            }
        ]
        response = """```python
print("wrong")
```
```python
line = input()
value = line.strip()
print(value)
```"""

        analysis, results = score_dataset(
            dataset,
            _responses(response),
            evaluator="implement_tested",
            python_executable=sys.executable,
        )

        assert analysis["accuracy"] == 1.0
        assert results[0]["n_code_blocks"] == 2
        # the code is not stored, but re-extracts from the stored response
        reparsed = parse_response(results[0]["response"])
        assert reparsed.longest_block is reparsed.code_blocks[1]
        assert "value = line.strip()" in reparsed.longest_block.code
