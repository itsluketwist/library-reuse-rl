"""Tests for scoring the libreusebench split."""

import pytest

from src.eval.scoring import score_dataset


def _responses(*texts: str) -> list[list[tuple[str, str]]]:
    """One sampled response per row, each paired with a 'stop' stop-reason."""
    return [[(text, "stop")] for text in texts]


class TestScoreLibReuseBench:
    """The LibReuseBench route uses its complete official evaluator."""

    def test_official_statistics_are_saved(self, monkeypatch, tmp_path):
        """Responses go through official install-and-score orchestration."""
        seen = {}

        def fake_evaluate_responses(
            responses,
            output_dir,
            env_dir,
            env_root,
            confirm,
        ):
            seen["responses"] = responses
            seen["output_dir"] = output_dir
            seen["env_dir"] = env_dir
            seen["env_root"] = env_root
            seen["confirm"] = confirm
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
                    "directory": env_root,
                    "source": "response_imports",
                    "installed_packages": ["pandas"],
                    "installed_package_versions": {"pandas": "2.3.1"},
                    "package_imports": {"pandas": ["pandas"]},
                },
                "results": [
                    {
                        "task_id": "LRB/T000",
                        "idx": 0,
                        "passed": True,
                        "external_libs": ["pandas"],
                    }
                ],
            }

        monkeypatch.setattr(
            "src.eval.scoring.libreusebench.evaluate_lrb_responses",
            fake_evaluate_responses,
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "LRB/T000",
                "category": "tables",
                "test_inputs": ["1\n"],
                "expected_outputs": ["1"],
            }
        ]

        analysis, results = score_dataset(
            dataset,
            _responses("```python\nimport pandas as pd\nprint(1)\n```"),
            evaluator="libreusebench",
            environment_root=str(tmp_path / "env"),
        )

        assert seen["responses"] == {
            "LRB/T000": ["```python\nimport pandas as pd\nprint(1)\n```"],
        }
        assert seen["env_root"] == str(tmp_path / "env")
        # no cache dir, so the evaluator builds a disposable venv it deletes afterwards
        assert seen["env_dir"] is None
        assert seen["confirm"] is None  # unattended eval runs install automatically
        assert analysis["task"] == "libreusebench"
        assert analysis["pass@1"] == 1.0
        assert analysis["library_use"]["per_task"]["using_lib_rate"] == 1.0
        assert analysis["environment"]["installed_packages"] == ["pandas"]
        assert analysis["environment"]["installed_package_versions"] == {
            "pandas": "2.3.1"
        }
        assert results[0]["task_id"] == "LRB/T000"
        assert results[0]["stop_reason"] == "stop"

    def test_multiple_samples_are_all_submitted_for_their_task(self, monkeypatch):
        """Every sample for a row becomes its own entry in that task_id's response list."""
        seen = {}

        def fake_evaluate_responses(responses, output_dir, env_dir, env_root, confirm):
            seen["responses"] = responses
            return {
                "n_tasks": 1,
                "n_responses": 2,
                "k": 2,
                "pass@1": 0.5,
                "pass@k": 1.0,
                "library_use": {
                    "per_task": {"using_lib_rate": 0.0},
                    "per_response": {"using_lib_rate": 0.0},
                },
                "environment": {},
                "results": [
                    {"task_id": "LRB/T000", "idx": 0, "passed": True},
                    {"task_id": "LRB/T000", "idx": 1, "passed": False},
                ],
            }

        monkeypatch.setattr(
            "src.eval.scoring.libreusebench.evaluate_lrb_responses",
            fake_evaluate_responses,
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "LRB/T000",
                "category": "tables",
                "test_inputs": ["1\n"],
                "expected_outputs": ["1"],
            }
        ]
        responses = [
            [("print(1)", "stop"), ("print(2)", "length")],
        ]

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="libreusebench",
        )

        assert seen["responses"] == {"LRB/T000": ["print(1)", "print(2)"]}
        assert analysis["pass@1"] == 0.5
        # each sample's own stop_reason is matched back by its sample idx
        assert results[0]["stop_reason"] == "stop"
        assert results[1]["stop_reason"] == "length"

    def test_environment_setup_failure_is_fatal(self, monkeypatch):
        """An official evaluator abort caused by failed installs fails the run."""
        monkeypatch.setattr(
            "src.eval.scoring.libreusebench.evaluate_lrb_responses",
            lambda **_kwargs: {},
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "LRB/T000",
                "category": "tables",
                "test_inputs": ["1\n"],
                "expected_outputs": ["1"],
            }
        ]

        with pytest.raises(RuntimeError, match="package environment"):
            score_dataset(
                dataset,
                _responses("```python\nprint(1)\n```"),
                evaluator="libreusebench",
            )
