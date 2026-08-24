"""Tests for the harness flow in libreusebench/evaluate.py: parse, plan installs, confirm, key by id.

Exercises the pure logic only - no venv is created and nothing is pip-installed (the confirmation
gate is declined, or a fake benchmark is injected).
"""

import pytest
from libreusebench import evaluate

from src.utils.io import save_jsonl


_FAKE_BENCHMARK = {
    "test": [
        {
            "task_id": "LRB/T000",
            "category": "misc",
            "test_inputs": ["1\n"],
            "expected_outputs": ["1"],
        }
    ]
}


class _FakeEnv:
    """A stand-in Environment that records installs and reports every response as passing."""

    def __init__(self, cache_dir: str) -> None:
        self.installed: set[str] = set()

    def ensure_packages(self, packages: set[str]) -> set[str]:
        self.installed |= set(packages)
        return set()  # nothing fails to install

    def installed_distributions(self) -> dict[str, str]:
        """Return deterministic installed versions for result-recording tests."""
        return {package: "1.2.3" for package in sorted(self.installed)}

    def evaluate(
        self,
        program: str,
        inputs: list[str],
        expected_outputs: list[str],
        timeout_seconds: float,
    ) -> tuple[bool, str]:
        return True, ""


class TestPlanInstalls:
    """Planning collects the pip packages across all responses, mapping import -> distribution."""

    def test_maps_and_aggregates(self) -> None:
        responses = {
            "LRB/T000": ["```python\nimport sklearn\nimport numpy as np\n```"],
            "LRB/T001": ["```python\nimport numpy\n```"],
        }
        plan = evaluate.plan_installs(responses)
        assert plan["scikit-learn"] == {"sklearn"}  # import name remapped to pip name
        assert plan["numpy"] == {"numpy"}


class TestLoadResponses:
    """Responses load keyed by task id, accepting both the list and single-response forms."""

    def test_both_forms(self, tmp_path) -> None:
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [
                {"task_id": "LRB/T000", "responses": ["x", "y"]},
                {"task_id": "LRB/T001", "response": "z"},
            ],
            path,
        )
        assert evaluate._load_responses(str(path)) == {
            "LRB/T000": ["x", "y"],
            "LRB/T001": ["z"],
        }

    def test_missing_response_field_errors(self, tmp_path) -> None:
        path = tmp_path / "resp.jsonl"
        save_jsonl([{"task_id": "LRB/T000"}], path)
        with pytest.raises(ValueError):
            evaluate._load_responses(str(path))


class TestInstallationPlan:
    """Packages install by default, with an optional confirmation gate."""

    def test_decline_writes_env_and_aborts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [{"task_id": "LRB/T000", "response": "```python\nimport numpy\n```"}],
            path,
        )

        env_dir = tmp_path / "venv"
        out_dir = tmp_path / "out"
        result = evaluate.evaluate_responses(
            responses_file=str(path),
            output_directory=str(out_dir),
            env_dir=str(env_dir),
            confirm=lambda _plan: False,
        )
        assert result == {}
        assert not env_dir.exists()  # venv never created when declined
        # the collected packages are written out for the user to correct and reuse
        assert evaluate.read_env_file(str(out_dir / evaluate._ENV_FILENAME)) == {
            "numpy"
        }

    def test_unknown_task_id_errors(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        path = tmp_path / "resp.jsonl"
        save_jsonl([{"task_id": "LRB/T999", "response": "x"}], path)
        with pytest.raises(ValueError):
            evaluate.evaluate_responses(
                responses_file=str(path),
                output_directory=str(tmp_path / "out"),
                env_dir=str(tmp_path / "venv"),
                confirm=lambda _plan: True,
            )

    def test_approved_plan_installs_detected_response_packages(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """The complete evaluator installs packages inferred from model responses."""
        environments = []

        class _RecordingEnv(_FakeEnv):
            def __init__(self, cache_dir: str) -> None:
                super().__init__(cache_dir)
                environments.append(self)

        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _RecordingEnv)
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [
                {
                    "task_id": "LRB/T000",
                    "response": "```python\nimport sklearn\nprint(1)\n```",
                }
            ],
            path,
        )

        result = evaluate.evaluate_responses(
            responses_file=str(path),
            output_directory=str(tmp_path / "out"),
            env_dir=str(tmp_path / "venv"),
        )

        assert result["pass@1"] == 1.0
        assert environments[0].installed == {"scikit-learn"}
        assert result["environment"] == {
            "directory": str(tmp_path / "venv"),
            "source": "response_imports",
            "installed_packages": ["scikit-learn"],
            "installed_package_versions": {"scikit-learn": "1.2.3"},
            "package_imports": {"scikit-learn": ["sklearn"]},
        }


class TestEnvFile:
    """The environment file round-trips and drives the run without prompting."""

    def test_read_ignores_comments_and_blanks(self, tmp_path) -> None:
        path = tmp_path / "env.txt"
        path.write_text(
            "# a comment\n\nnumpy\nscikit-learn  # import: sklearn\npandas==3.0.5\n"
        )
        assert evaluate.read_env_file(str(path)) == {
            "numpy",
            "scikit-learn",
            "pandas==3.0.5",
        }

    def test_write_flags_failures_and_round_trips(self, tmp_path) -> None:
        path = evaluate.write_env_file(
            str(tmp_path),
            packages={"scikit-learn", "numpy"},
            origins={"scikit-learn": {"sklearn"}},
            failed={"scikit-learn"},
        )
        text = path.read_text()
        assert "import: sklearn" in text and "FAILED TO INSTALL" in text
        assert evaluate.read_env_file(str(path)) == {"scikit-learn", "numpy"}

    def test_env_file_skips_confirmation_and_runs(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _FakeEnv)

        resp = tmp_path / "resp.jsonl"
        save_jsonl(
            [{"task_id": "LRB/T000", "response": "```python\nprint(1)\n```"}], resp
        )
        env = tmp_path / "env.txt"
        env.write_text("numpy\n")

        def _fail_if_called(_plan: dict) -> bool:
            raise AssertionError("confirm must not be called when --env is supplied")

        result = evaluate.evaluate_responses(
            responses_file=str(resp),
            output_directory=str(tmp_path / "out"),
            env_dir=str(tmp_path / "venv"),
            env_file=str(env),
            confirm=_fail_if_called,
        )
        assert result["n_responses"] == 1
        assert result["pass@1"] == 1.0
        assert result["environment"]["source"] == str(env)
        assert result["environment"]["installed_packages"] == ["numpy"]
        assert result["environment"]["installed_package_versions"] == {"numpy": "1.2.3"}


class TestStatistics:
    """pass@k uses the unbiased estimator; library stats are aggregated per problem."""

    def test_pass_at_k(self) -> None:
        assert evaluate._pass_at_k(4, 2, 1) == 0.5  # 1 - C(2,1)/C(4,1)
        assert evaluate._pass_at_k(4, 0, 2) == 0.0  # no correct samples
        assert evaluate._pass_at_k(4, 4, 2) == 1.0  # all correct
        assert (
            evaluate._pass_at_k(2, 1, 2) == 1.0
        )  # one correct of two -> pass@2 certain

    def test_aggregate_per_task_and_per_response(self) -> None:
        # T000 has two responses (one using libs, one not); T001 has one, no libs
        per_response = [
            {
                "task_id": "LRB/T000",
                "passed": True,
                "external_libs": ["numpy", "pandas"],
            },
            {"task_id": "LRB/T000", "passed": False, "external_libs": []},
            {"task_id": "LRB/T001", "passed": False, "external_libs": []},
        ]
        res = evaluate._aggregate(per_response)

        assert res["n_tasks"] == 2 and res["n_responses"] == 3
        per_task = res["library_use"]["per_task"]
        assert per_task["using_lib_rate"] == 0.5  # 1 of 2 tasks used a library (union)
        assert per_task["avg_extlibs"] == 1.0  # ({numpy,pandas}=2 + {}=0) / 2
        assert per_task["library_counts"] == {"numpy": 1, "pandas": 1}

        per_resp = res["library_use"]["per_response"]
        assert (
            round(per_resp["using_lib_rate"], 3) == 0.333
        )  # 1 of 3 responses used a library
        assert round(per_resp["avg_extlibs"], 3) == 0.667  # (2 + 0 + 0) / 3
        assert per_resp["library_counts"] == {"numpy": 1, "pandas": 1}
