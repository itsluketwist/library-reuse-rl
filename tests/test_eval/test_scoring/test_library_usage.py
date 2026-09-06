"""Tests for scoring the library-usage split."""

from src.eval.scoring import score_dataset


def _responses(*texts: str) -> list[list[tuple[str, str]]]:
    """One sampled response per row, each paired with a 'stop' stop-reason."""
    return [[(text, "stop")] for text in texts]


def _multi_responses(*rows: list[str]) -> list[list[tuple[str, str]]]:
    """Pair each row's sampled response texts with a 'stop' stop-reason.

    Each argument is one dataset row's list of sampled response texts (see
    MULTI_SAMPLE_EVALUATORS - these evaluators score every sample, not just one).
    """
    return [[(text, "stop") for text in row] for row in rows]


class TestScoreLibraryUsage:
    """Tests for scoring the library-usage split."""

    def test_external_use_vs_stdlib_and_local(self):
        """Only a used external library counts; stdlib and local imports do not."""
        base = {
            "task_type": "library_usage",
            "category_id": "pdf",
            "useful_libs": ["pypdf"],
        }
        dataset = [
            {**base, "idx": 0, "local_modules": []},
            {**base, "idx": 1, "local_modules": ["src"]},
            {**base, "idx": 2, "local_modules": []},
        ]
        responses = _responses(
            "r</think>\n```python\nimport pypdf\npypdf.PdfReader('x')\n```",
            "r</think>\n```python\nfrom src.models import Job\nJob()\n```",
            "r</think>\n```python\nimport math\nprint(math.pi)\n```",
        )

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="library_usage",
        )

        assert analysis["task"] == "library_usage"
        # only the first row reused an external library
        assert analysis["external_lib_used_rate"] == 1 / 3
        assert analysis["appropriate_lib_rate"] == 1 / 3
        assert analysis["library_use"] == {
            "using_lib_rate": 1 / 3,
            "avg_extlibs": 1 / 3,
            "library_counts": {"pypdf": 1},
            "appropriate_lib_rate": 1 / 3,
            "appropriate_given_use_rate": 1.0,
        }
        assert results[0]["external_lib_used"] and results[0]["appropriate_lib_used"]
        assert not results[1]["external_lib"]  # local import doesn't count
        assert not results[2]["external_lib"]  # stdlib doesn't count

    def test_aggregates_all_code_blocks(self):
        """Every block contributes imports, usage, and validity statistics."""
        dataset = [
            {
                "idx": 0,
                "task_type": "library_usage",
                "category_id": "http-requests",
                "useful_libs": ["requests"],
                "local_modules": ["app"],
            }
        ]
        response = """r</think>
```python
import requests
requests.get("https://example.com")
```
```python
import numpy as np
from app.models import Item
np.array([1])
Item()
```
"""

        analysis, results = score_dataset(
            dataset,
            _responses(response),
            evaluator="library_usage",
        )

        assert results[0]["external_libs"] == ["numpy", "requests"]
        assert results[0]["used_external_libs"] == ["numpy", "requests"]
        assert results[0]["n_code_blocks"] == 2
        assert results[0]["n_valid_code_blocks"] == 2
        assert results[0]["all_code_blocks_valid"]
        assert not results[0]["code_blocks"][0]["selected_for_tests"]
        assert results[0]["code_blocks"][1]["selected_for_tests"]
        assert analysis["library_use"]["library_counts"] == {
            "numpy": 1,
            "requests": 1,
        }

    def test_one_valid_block_is_enough_for_valid_code_rate(self):
        """A throwaway broken snippet alongside a working one still counts as valid."""
        dataset = [
            {
                "idx": 0,
                "task_type": "library_usage",
                "category_id": "http-requests",
                "useful_libs": [],
                "local_modules": [],
            }
        ]
        response = """r</think>
```python
import requests
requests.get("https://example.com")
```
```python
def broken(:
```
"""

        analysis, results = score_dataset(
            dataset,
            _responses(response),
            evaluator="library_usage",
        )

        assert results[0]["n_code_blocks"] == 2
        assert results[0]["n_valid_code_blocks"] == 1
        # one working block is enough, unlike all_code_blocks_valid below
        assert results[0]["valid"]
        assert not results[0]["all_code_blocks_valid"]
        assert analysis["valid_code_rate"] == 1.0
        # external_lib_used always comes from a block that parsed
        assert analysis["external_lib_used_rate"] <= analysis["valid_code_rate"]

    def test_multiple_samples_are_all_scored_and_aggregated_per_task(self):
        """With n_samples > 1, every sample is scored; per-task rates are any-of-k."""
        dataset = [
            {
                "idx": 0,
                "task_type": "library_usage",
                "category_id": "http-requests",
                "useful_libs": ["requests"],
                "local_modules": [],
            }
        ]
        # one of the two samples for this task reused the useful library
        responses = _multi_responses(
            [
                "r</think>\n```python\nimport requests\nrequests.get('x')\n```",
                "r</think>\n```python\nimport math\nprint(math.pi)\n```",
            ]
        )

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="library_usage",
        )

        assert len(results) == 2
        assert [r["sample_idx"] for r in results] == [0, 1]
        # per-response: only half the samples used the library
        assert analysis["external_lib_used_rate"] == 0.5
        assert analysis["library_use"]["using_lib_rate"] == 0.5
        # per-task: this task's samples used it at least once
        assert analysis["library_use_per_task"]["using_lib_rate"] == 1.0
        assert analysis["library_use_per_task"]["appropriate_lib_rate"] == 1.0


class TestScoreEmptyDataset:
    """Tests for the empty-dataset edge case."""

    def test_empty_dataset_scores_nothing(self):
        """An empty dataset returns a zero-total analysis and no results."""
        analysis, results = score_dataset([], [], evaluator="library_usage")

        assert analysis["n_total"] == 0
        assert results == []
