"""Tests for the task-aware eval scoring (product-choice + library-usage paths)."""

from src.eval.scoring import detect_task, score_dataset


def _responses(*texts: str) -> list[tuple[str, str]]:
    """Pair each response text with a 'stop' stop-reason."""
    return [(text, "stop") for text in texts]


class TestDetectTask:
    """Tests for detect_task."""

    def test_tested_from_test_columns(self):
        """Stdin/stdout test columns mark the tested-implementation task."""
        columns = ["prompt", "inputs", "expected_outputs", "idx"]
        assert detect_task(columns, {"idx": 0}) == "implement_tested"

    def test_new_tasks_from_task_type(self):
        """Without test columns, the row's task_type decides."""
        assert (
            detect_task(["prompt", "idx"], {"task_type": "product_choice"})
            == "product_choice"
        )
        assert (
            detect_task(["prompt", "idx"], {"task_type": "library_usage"})
            == "library_usage"
        )


class TestScoreProductChoice:
    """Tests for scoring the product-choice split."""

    def test_external_pick_and_diy_rates(self):
        """External picks and DIY picks are counted into their rates and per-category."""
        dataset = [
            {
                "idx": 0,
                "task_type": "product_choice",
                "category_id": "databases",
                "allowlist": ["postgresql"],
                "diy_terms": ["json file"],
            },
            {
                "idx": 1,
                "task_type": "product_choice",
                "category_id": "databases",
                "allowlist": ["postgresql"],
                "diy_terms": ["json file"],
            },
        ]
        responses = _responses(
            "reasoning</think>\nUse PostgreSQL.",
            "reasoning</think>\nJust use a JSON file.",
        )

        analysis, results = score_dataset(dataset, ["prompt", "idx"], responses)

        assert analysis["task"] == "product_choice"
        assert analysis["external_pick_rate"] == 0.5
        assert analysis["diy_rate"] == 0.5
        assert analysis["external_pick_rate_by_category"] == {"databases": 0.5}
        assert results[0]["external_pick"] and results[0]["primary"] == "postgresql"
        assert results[1]["diy_pick"]


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

        analysis, results = score_dataset(dataset, ["prompt", "idx"], responses)

        assert analysis["task"] == "library_usage"
        # only the first row reused an external library
        assert analysis["external_lib_used_rate"] == 1 / 3
        assert analysis["appropriate_lib_rate"] == 1 / 3
        assert results[0]["external_lib_used"] and results[0]["appropriate_lib_used"]
        assert not results[1]["external_lib"]  # local import doesn't count
        assert not results[2]["external_lib"]  # stdlib doesn't count


class TestScoreEmptyDataset:
    """Tests for the empty-dataset edge case."""

    def test_empty_dataset_scores_nothing(self):
        """An empty dataset returns a zero-total analysis and no results."""
        analysis, results = score_dataset([], ["prompt", "idx"], [])

        assert analysis["n_total"] == 0
        assert results == []
