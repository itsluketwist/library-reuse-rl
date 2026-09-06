"""Tests for the harness flow in libreusebench/evaluate.py: parse, plan installs, confirm, key by id.

Exercises the pure logic only - no venv is created and nothing is pip-installed (the confirmation
gate is declined, or a fake benchmark is injected).
"""

import json

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

    def __init__(
        self,
        cache_dir: str | None = None,
        scratch_root: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir or "<disposable>"
        self.installed: set[str] = set()
        self.closed = False

    def __enter__(self) -> "_FakeEnv":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

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
        assert evaluate.load_responses(str(path)) == {
            "LRB/T000": ["x", "y"],
            "LRB/T001": ["z"],
        }

    def test_json_file(self, tmp_path) -> None:
        path = tmp_path / "resp.json"
        path.write_text(
            json.dumps(
                [
                    {"task_id": "LRB/T000", "responses": ["x", "y"]},
                    {"task_id": "LRB/T001", "response": "z"},
                ]
            )
        )
        assert evaluate.load_responses(str(path)) == {
            "LRB/T000": ["x", "y"],
            "LRB/T001": ["z"],
        }

    def test_missing_response_field_errors(self, tmp_path) -> None:
        path = tmp_path / "resp.jsonl"
        save_jsonl([{"task_id": "LRB/T000"}], path)
        with pytest.raises(ValueError):
            evaluate.load_responses(str(path))


class TestEvaluateResponsesInMemory:
    """evaluate_responses takes responses directly; evaluate_responses_file is a thin
    file-loading wrapper around it, for callers that keep responses on disk."""

    def test_file_wrapper_matches_in_memory_call(self, tmp_path, monkeypatch) -> None:
        """Loading a file and calling evaluate_responses directly gives the same result
        as evaluate_responses_file loading and calling it internally."""
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _FakeEnv)
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [{"task_id": "LRB/T000", "response": "```python\nprint(1)\n```"}], path
        )

        direct = evaluate.evaluate_responses(
            responses={"LRB/T000": ["```python\nprint(1)\n```"]},
            output_dir=str(tmp_path / "out-direct"),
            env_dir=str(tmp_path / "venv-direct"),
            confirm=lambda _plan: True,
        )
        via_file = evaluate.evaluate_responses_file(
            responses_file=str(path),
            output_dir=str(tmp_path / "out-file"),
            env_dir=str(tmp_path / "venv-file"),
            confirm=lambda _plan: True,
        )

        for result in (direct, via_file):
            del result["results_file"], result["environment"]["directory"]
        assert direct == via_file


class TestInstallationPlan:
    """Packages are confirmed by default before installing, unless the caller opts out."""

    def test_decline_writes_env_and_aborts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [{"task_id": "LRB/T000", "response": "```python\nimport numpy\n```"}],
            path,
        )

        env_dir = tmp_path / "venv"
        out_dir = tmp_path / "out"
        result = evaluate.evaluate_responses_file(
            responses_file=str(path),
            output_dir=str(out_dir),
            env_dir=str(env_dir),
            confirm=lambda _plan: False,
        )
        assert result == {}
        assert not env_dir.exists()  # venv never created when declined
        # the collected packages are written out for the user to correct and reuse
        assert evaluate.read_env_file(str(out_dir / evaluate._ENV_FILENAME)) == {
            "numpy"
        }

    def test_default_confirm_asks_before_installing(
        self, tmp_path, monkeypatch
    ) -> None:
        """With no `confirm` override, the default gate prompts on stdin and can decline."""
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
        path = tmp_path / "resp.jsonl"
        save_jsonl(
            [{"task_id": "LRB/T000", "response": "```python\nimport numpy\n```"}],
            path,
        )

        env_dir = tmp_path / "venv"
        result = evaluate.evaluate_responses_file(
            responses_file=str(path),
            output_dir=str(tmp_path / "out"),
            env_dir=str(env_dir),
        )
        assert result == {}
        assert not env_dir.exists()  # venv never created when declined

    def test_unknown_task_id_errors(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        path = tmp_path / "resp.jsonl"
        save_jsonl([{"task_id": "LRB/T999", "response": "x"}], path)
        with pytest.raises(ValueError):
            evaluate.evaluate_responses_file(
                responses_file=str(path),
                output_dir=str(tmp_path / "out"),
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
            def __init__(
                self,
                cache_dir: str | None = None,
                scratch_root: str | None = None,
            ) -> None:
                super().__init__(cache_dir, scratch_root)
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

        result = evaluate.evaluate_responses_file(
            responses_file=str(path),
            output_dir=str(tmp_path / "out"),
            env_dir=str(tmp_path / "venv"),
            confirm=lambda _plan: True,
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

        result = evaluate.evaluate_responses_file(
            responses_file=str(resp),
            output_dir=str(tmp_path / "out"),
            env_dir=str(tmp_path / "venv"),
            env_file=str(env),
            confirm=_fail_if_called,
        )
        assert result["n_responses"] == 1
        assert result["pass@1"] == 1.0
        assert result["environment"]["source"] == str(env)
        assert result["environment"]["installed_packages"] == ["numpy"]
        assert result["environment"]["installed_package_versions"] == {"numpy": "1.2.3"}


class TestScoreResponse:
    """A response's imported and actually-used external libraries are tracked separately."""

    def _evaluate_one(
        self,
        response: str,
        tmp_path,
        monkeypatch,
    ) -> dict:
        """Run a single response through the full evaluator with a fake, always-passing env.

        Returns its per-response result dict.
        """
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _FakeEnv)
        path = tmp_path / "resp.jsonl"
        save_jsonl([{"task_id": "LRB/T000", "response": response}], path)

        result = evaluate.evaluate_responses_file(
            responses_file=str(path),
            output_dir=str(tmp_path / "out"),
            env_dir=str(tmp_path / "venv"),
            confirm=lambda _plan: True,
        )
        return result["results"][0]

    def test_imported_but_unused_library_is_not_counted_as_used(
        self, tmp_path, monkeypatch
    ) -> None:
        # numpy is imported but never called
        response = "```python\nimport numpy\nprint(1)\n```"
        result = self._evaluate_one(response, tmp_path, monkeypatch)
        assert result["external_libs"] == ["numpy"]
        assert result["external_lib"] is True
        assert result["used_external_libs"] == []
        assert result["external_lib_used"] is False

    def test_called_library_counts_as_both_imported_and_used(
        self, tmp_path, monkeypatch
    ) -> None:
        response = "```python\nimport numpy\nprint(numpy.array([1]))\n```"
        result = self._evaluate_one(response, tmp_path, monkeypatch)
        assert result["external_libs"] == ["numpy"]
        assert result["external_lib"] is True
        assert result["used_external_libs"] == ["numpy"]
        assert result["external_lib_used"] is True

    def test_no_code_reports_no_libraries(self, tmp_path, monkeypatch) -> None:
        result = self._evaluate_one("just prose, no code here", tmp_path, monkeypatch)
        assert result == {
            "task_id": "LRB/T000",
            "idx": 0,
            "passed": False,
            "external_libs": [],
            "external_lib": False,
            "used_external_libs": [],
            "external_lib_used": False,
            "reason": "no code",
        }


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
        # T000 has two responses (one importing+using libs, one not); T001 has one, no libs
        per_response = [
            {
                "task_id": "LRB/T000",
                "passed": True,
                "external_libs": ["numpy", "pandas"],
                "used_external_libs": ["pandas"],  # numpy imported but never called
            },
            {
                "task_id": "LRB/T000",
                "passed": False,
                "external_libs": [],
                "used_external_libs": [],
            },
            {
                "task_id": "LRB/T001",
                "passed": False,
                "external_libs": [],
                "used_external_libs": [],
            },
        ]
        res = evaluate._aggregate(per_response)

        assert res["n_tasks"] == 2 and res["n_responses"] == 3

        imported_task = res["library_use"]["imported"]["per_task"]
        assert (
            imported_task["using_lib_rate"] == 0.5
        )  # 1 of 2 tasks imported a library (union)
        assert imported_task["avg_extlibs"] == 1.0  # ({numpy,pandas}=2 + {}=0) / 2
        assert imported_task["library_counts"] == {"numpy": 1, "pandas": 1}

        imported_resp = res["library_use"]["imported"]["per_response"]
        assert (
            round(imported_resp["using_lib_rate"], 3) == 0.333
        )  # 1 of 3 responses imported a library
        assert round(imported_resp["avg_extlibs"], 3) == 0.667  # (2 + 0 + 0) / 3
        assert imported_resp["library_counts"] == {"numpy": 1, "pandas": 1}

        used_task = res["library_use"]["used"]["per_task"]
        assert (
            used_task["using_lib_rate"] == 0.5
        )  # 1 of 2 tasks actually used a library
        assert used_task["avg_extlibs"] == 0.5  # ({pandas}=1 + {}=0) / 2
        assert used_task["library_counts"] == {"pandas": 1}  # numpy never used

        used_resp = res["library_use"]["used"]["per_response"]
        assert round(used_resp["using_lib_rate"], 3) == 0.333
        assert round(used_resp["avg_extlibs"], 3) == 0.333  # (1 + 0 + 0) / 3
        assert used_resp["library_counts"] == {"pandas": 1}
