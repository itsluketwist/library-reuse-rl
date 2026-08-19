"""Tests for the explicitly configured evaluation scorers."""

import sys

import pytest

from src.eval.scoring import score_dataset
from src.utils.io import load_jsonl


def _responses(*texts: str) -> list[tuple[str, str]]:
    """Pair each response text with a 'stop' stop-reason."""
    return [(text, "stop") for text in texts]


class TestScoreProductChoice:
    """Tests for scoring the product-choice split."""

    def test_external_pick_and_diy_rates(self):
        """External picks and DIY picks are counted into their overall rates."""
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

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="product_choice",
        )

        assert analysis["task"] == "product_choice"
        assert analysis["external_pick_rate"] == 0.5
        assert analysis["diy_rate"] == 0.5
        assert results[0]["external_pick"] and results[0]["primary"] == "postgresql"
        assert results[1]["diy_pick"]


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


class TestScoreEmptyDataset:
    """Tests for the empty-dataset edge case."""

    def test_empty_dataset_scores_nothing(self):
        """An empty dataset returns a zero-total analysis and no results."""
        analysis, results = score_dataset([], [], evaluator="library_usage")

        assert analysis["n_total"] == 0
        assert results == []


class TestScoreBigCodeBenchReuse:
    """The BCBReuse route uses its complete official evaluator."""

    def test_official_statistics_are_saved(self, monkeypatch, tmp_path):
        """Responses go through official install-and-score orchestration."""
        seen = {}

        def fake_evaluate_responses(
            responses_file,
            output_directory,
            env_dir,
        ):
            seen["records"] = load_jsonl(responses_file)
            seen["output_directory"] = output_directory
            seen["env_dir"] = env_dir
            return {
                "n_tasks": 1,
                "n_responses": 1,
                "k": 1,
                "pass@1": 1.0,
                "pass@k": 1.0,
                "library_use": {
                    "per_task": {"using_lib_rate": 1.0},
                    "per_response": {"using_lib_rate": 1.0},
                },
                "environment": {
                    "directory": env_dir,
                    "source": "response_imports",
                    "installed_packages": ["pandas"],
                    "installed_package_versions": {"pandas": "2.3.1"},
                    "package_imports": {"pandas": ["pandas"]},
                },
                "results": [
                    {
                        "task_id": "BCBR/T000",
                        "idx": 0,
                        "passed": True,
                        "external_libs": ["pandas"],
                    }
                ],
            }

        monkeypatch.setattr(
            "src.eval.scoring.evaluate_bcb_responses",
            fake_evaluate_responses,
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "BCBR/T000",
                "category": "tables",
                "test_inputs": ["1\n"],
                "expected_outputs": ["1"],
            }
        ]

        analysis, results = score_dataset(
            dataset,
            _responses("```python\nimport pandas as pd\nprint(1)\n```"),
            evaluator="bcbreuse",
            environment_dir=str(tmp_path / "env"),
        )

        assert seen["records"] == [
            {
                "task_id": "BCBR/T000",
                "response": "```python\nimport pandas as pd\nprint(1)\n```",
            }
        ]
        assert seen["env_dir"] == str(tmp_path / "env")
        assert analysis["task"] == "bcbreuse"
        assert analysis["pass@1"] == 1.0
        assert analysis["library_use"]["per_task"]["using_lib_rate"] == 1.0
        assert analysis["environment"]["installed_packages"] == ["pandas"]
        assert analysis["environment"]["installed_package_versions"] == {
            "pandas": "2.3.1"
        }
        assert results[0]["task_id"] == "BCBR/T000"
        assert results[0]["stop_reason"] == "stop"

    def test_environment_setup_failure_is_fatal(self, monkeypatch):
        """An official evaluator abort caused by failed installs fails the run."""
        monkeypatch.setattr(
            "src.eval.scoring.evaluate_bcb_responses",
            lambda **_kwargs: {},
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "BCBR/T000",
                "category": "tables",
                "test_inputs": ["1\n"],
                "expected_outputs": ["1"],
            }
        ]

        with pytest.raises(RuntimeError, match="package environment"):
            score_dataset(
                dataset,
                _responses("```python\nprint(1)\n```"),
                evaluator="bcbreuse",
            )
