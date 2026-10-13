"""Tests for the harness flow in libreusebench/evaluate.py: parse, plan installs, confirm, key by id.

Exercises the pure logic only - no venv is created and nothing is pip-installed (the confirmation
gate is declined, or a fake benchmark is injected).
"""

import json
import sys
from pathlib import Path

import pytest
from libreusebench import evaluate
from libreusebench.environment import Environment

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
            "install_failed": [],
            "task_packages": [],
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
            "lloc": None,
            "maintainability_index": None,
            "ran": False,
            "install_failed": [],
            "reason": "no code",
        }

    def test_code_size_and_maintainability_are_measured(
        self, tmp_path, monkeypatch
    ) -> None:
        # three statements: comments and blank lines don't count towards lloc
        response = "```python\n# say hi\nx = 1\n\ny = x + 1\nprint(y)\n```"
        result = self._evaluate_one(response, tmp_path, monkeypatch)
        assert result["lloc"] == 3
        assert 0.0 <= result["maintainability_index"] <= 100.0

    def test_unparseable_code_has_no_metrics(self, tmp_path, monkeypatch) -> None:
        result = self._evaluate_one("```python\ndef f(:\n```", tmp_path, monkeypatch)
        assert result["lloc"] is None
        assert result["maintainability_index"] is None


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

    def test_code_metrics_all_and_passed(self) -> None:
        # one passing and one failing measured response, plus one that couldn't be measured
        per_response = [
            {
                "task_id": "LRB/T000",
                "passed": True,
                "lloc": 10,
                "maintainability_index": 80.0,
            },
            {
                "task_id": "LRB/T000",
                "passed": False,
                "lloc": 4,
                "maintainability_index": 60.0,
            },
            {
                "task_id": "LRB/T001",
                "passed": False,
                "lloc": None,
                "maintainability_index": None,
            },
        ]
        res = evaluate._aggregate(per_response)

        # the unmeasured response is skipped rather than counted as zero
        assert res["code_metrics"]["all"] == {
            "n_measured": 2,
            "avg_lloc": 7.0,
            "avg_maintainability_index": 70.0,
        }
        assert res["code_metrics"]["passed"] == {
            "n_measured": 1,
            "avg_lloc": 10.0,
            "avg_maintainability_index": 80.0,
        }

    def test_code_metrics_none_when_nothing_measured(self) -> None:
        per_response = [{"task_id": "LRB/T000", "passed": False, "lloc": None}]
        res = evaluate._aggregate(per_response)
        assert res["code_metrics"]["passed"] == {
            "n_measured": 0,
            "avg_lloc": None,
            "avg_maintainability_index": None,
        }


class _FailingInstallEnv(_FakeEnv):
    """A fake environment where one package (e.g. "backports") can't be installed."""

    def ensure_packages(self, packages: set[str]) -> set[str]:
        failed = {package for package in packages if package == "backports"}
        self.installed |= set(packages) - failed
        return failed


class TestInstallFailures:
    """What happens when a response imports a package that can't be installed."""

    _RESPONSES = {
        "LRB/T000": [
            "```python\nfrom backports import zoneinfo\nprint(zoneinfo)\n```",
            "```python\nprint(1)\n```",
        ],
    }

    def test_continues_by_default_without_running_affected_responses(
        self, tmp_path, monkeypatch
    ) -> None:
        """Responses needing a failed package aren't run, and count as not passed; the rest
        still run, and the results list what didn't."""
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _FailingInstallEnv)

        result = evaluate.evaluate_responses(
            responses=self._RESPONSES,
            output_dir=str(tmp_path),
            confirm=None,
        )

        needs_backports, plain = result["results"]
        assert not needs_backports["ran"]
        assert not needs_backports["passed"]  # the fake env would have passed it
        assert needs_backports["install_failed"] == ["backports"]
        assert needs_backports["reason"] == "not run: could not install backports"
        # libraries still come from reading the code
        assert needs_backports["external_libs"] == ["backports"]
        assert plain["ran"] and plain["passed"]

        assert result["pass@1"] == 0.5
        assert result["not_run"] == {
            "n_responses": 1,
            "packages": ["backports"],
            "responses": [{"task_id": "LRB/T000", "idx": 0, "packages": ["backports"]}],
        }
        assert result["environment"]["install_failed"] == ["backports"]
        assert "backports" not in result["environment"]["installed_packages"]

    def test_stops_when_asked(self, tmp_path, monkeypatch) -> None:
        """stop_on_install_failure stops, so the environment file can be fixed and re-run."""
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        monkeypatch.setattr(evaluate, "Environment", _FailingInstallEnv)

        result = evaluate.evaluate_responses(
            responses=self._RESPONSES,
            output_dir=str(tmp_path),
            confirm=None,
            stop_on_install_failure=True,
        )

        assert result == {}


class _LocalEnv(Environment):
    """The real Environment's test running, with this test's own python instead of a built venv."""

    def __init__(self) -> None:
        pass

    @property
    def python(self) -> Path:
        return Path(sys.executable)


# one task in each format: a function task (it has a function_name) and a stdin/stdout task
_FUNCTION_RECORD = {
    "task_id": "LRB2/T000",
    "function_name": "double",
    "test_inputs": ["[1]", "[3]"],
    "expected_outputs": ["2", "6"],
    "reuse_value": 20,
}
_PROGRAM_RECORD = {
    "task_id": "LRB/T000",
    "test_inputs": ["1\n"],
    "expected_outputs": ["2"],
}


class TestFunctionTasks:
    """A record with a function_name is tested by calling the function, any other as a program."""

    def test_each_record_is_scored_in_its_own_format(self) -> None:
        function_answer = "```python\ndef double(x):\n    return 2 * x\n```"
        program_answer = "```python\nprint(int(input()) * 2)\n```"
        result = evaluate.score_responses(
            responses={
                # the right function, then a program that never defines it
                "LRB2/T000": [function_answer, "```python\nprint(2)\n```"],
                # the right program, then a function that prints nothing
                "LRB/T000": [program_answer, function_answer],
            },
            benchmark={
                "LRB2/T000": _FUNCTION_RECORD,
                "LRB/T000": _PROGRAM_RECORD,
            },
            env=_LocalEnv(),
        )

        rows = result["results"]
        assert [row["passed"] for row in rows] == [True, False, True, False]
        assert "function not defined: double" in rows[1]["reason"]
        assert "test 1: expected '2'" in rows[3]["reason"]

        # the function task's reuse value is copied onto its results, and summarised
        assert rows[0]["reuse_value"] == 20
        assert "reuse_value" not in rows[2]
        assert result["reuse_value"] == {
            "tiers": {"high": {"n_tasks": 1, "pass@1": 0.5, "library_use": 0.0}},
            "high_value_library_use": 0.0,
        }


class TestRecordsArgument:
    """evaluate_responses scores against the records it is given, not the bundled splits."""

    _RECORDS = [
        {
            "task_id": "MINE/000",
            "test_inputs": ["1\n"],
            "expected_outputs": ["1"],
        }
    ]

    def test_given_records_replace_the_bundled_splits(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        def _fail_if_called() -> dict:
            raise AssertionError("the bundled splits are not needed with records")

        monkeypatch.setattr(evaluate, "load_dataset", _fail_if_called)
        monkeypatch.setattr(evaluate, "Environment", _FakeEnv)

        result = evaluate.evaluate_responses(
            responses={"MINE/000": ["```python\nprint(1)\n```"]},
            output_dir=str(tmp_path),
            confirm=None,
            records=self._RECORDS,
        )

        assert result["n_tasks"] == 1
        assert result["pass@1"] == 1.0
        assert result["results"][0]["task_id"] == "MINE/000"

    def test_a_tasks_own_packages_are_installed_too(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """A response may need a package it never imports, such as a parser loaded behind the
        scenes. The task's library solution names it, so it is installed (unpinned)."""
        monkeypatch.setattr(evaluate, "Environment", _FakeEnv)
        records = [
            self._RECORDS[0]
            | {
                "ground_truth": {
                    "gold_extlib_packages": ["lxml==6.1.3", "pandas==3.0.6"]
                }
            }
        ]

        result = evaluate.evaluate_responses(
            responses={"MINE/000": ["```python\nimport numpy\nprint(1)\n```"]},
            output_dir=str(tmp_path),
            confirm=None,
            records=records,
        )

        environment = result["environment"]
        assert environment["installed_packages"] == ["lxml", "numpy", "pandas"]
        assert environment["task_packages"] == ["lxml", "pandas"]
        # only the response's own import is recorded as coming from a response
        assert environment["package_imports"]["numpy"] == ["numpy"]

    def test_a_task_missing_from_the_records_errors(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        # LRB/T000 is in the (fake) bundled splits, but not in the records given
        monkeypatch.setattr(evaluate, "load_dataset", lambda: _FAKE_BENCHMARK)
        with pytest.raises(ValueError, match="unknown task ids"):
            evaluate.evaluate_responses(
                responses={"LRB/T000": ["```python\nprint(1)\n```"]},
                output_dir=str(tmp_path),
                confirm=None,
                records=self._RECORDS,
            )


class TestReuseValue:
    """Results that carry a reuse value are also summarised per reuse tier."""

    def test_tier_bounds(self) -> None:
        tiers = [evaluate.reuse_tier(value) for value in (-3, 4, 5, 14, 15, 80)]
        assert tiers == ["low", "low", "medium", "medium", "high", "high"]

    def test_pass_rate_and_library_use_per_tier(self) -> None:
        per_response = [
            # a high-value task: both responses pass, and one of them uses a library
            {
                "task_id": "A",
                "passed": True,
                "used_external_libs": ["bs4"],
                "reuse_value": 55,
            },
            {
                "task_id": "A",
                "passed": True,
                "used_external_libs": [],
                "reuse_value": 55,
            },
            # a second high-value task: its one response fails, without a library
            {
                "task_id": "B",
                "passed": False,
                "used_external_libs": [],
                "reuse_value": 15,
            },
            # a low-value task, where the library solution is the longer one
            {
                "task_id": "C",
                "passed": True,
                "used_external_libs": [],
                "reuse_value": -2,
            },
            # a control task has no reuse value, so it is in no tier
            {
                "task_id": "D",
                "passed": True,
                "used_external_libs": ["numpy"],
            },
        ]

        block = evaluate._aggregate(per_response)["reuse_value"]

        # high: tasks A (2 of 2 passed) and B (0 of 1), and 1 of their 3 responses used a library
        assert block["tiers"] == {
            "low": {"n_tasks": 1, "pass@1": 1.0, "library_use": 0.0},
            "high": {
                "n_tasks": 2,
                "pass@1": 0.5,
                "library_use": pytest.approx(1 / 3),
            },
        }
        assert block["high_value_library_use"] == pytest.approx(1 / 3)

    def test_no_block_without_reuse_values(self) -> None:
        results = evaluate._aggregate([{"task_id": "A", "passed": True}])
        assert "reuse_value" not in results
