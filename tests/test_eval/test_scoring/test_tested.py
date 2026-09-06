"""Tests for scoring the tested-implementation split."""

import sys

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
        assert len(results[0]["code_blocks"]) == 2
        assert not results[0]["code_blocks"][0]["selected_for_tests"]
        assert results[0]["code_blocks"][1]["selected_for_tests"]
        assert "value = line.strip()" in results[0]["code"]
