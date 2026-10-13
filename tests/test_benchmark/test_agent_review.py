"""Tests for stage 03 of the benchmark build: the blind solves, the rubric, and what gets fixed.

The stage is a script (benchmark/pipeline/03_agent_review.py), so it is loaded by path. No model
is called and nothing is installed: the replies are stubbed, and solutions run with this test's
own python.
"""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import build
import pytest
from libreusebench.environment import InstallResult

from src.utils.io import save_jsonl


_SCRIPT = Path(__file__).parents[2] / "benchmark" / "pipeline" / "03_agent_review.py"
_spec = importlib.util.spec_from_file_location("agent_review", _SCRIPT)
assert _spec is not None and _spec.loader is not None
agent_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_review)


_RIGHT = "def double(number):\n    return 2 * number\n"
_WRONG = "def double(number):\n    return number + 2\n"

# a reply to the rubric call that keeps the task
_KEEP = {
    "spec_complete": True,
    "example_consistent": True,
    "tests_fair": True,
    "no_library_leak": True,
    "library_fit": True,
    "text_simple": True,
    "request_natural": True,
    "verdict": "keep",
    "note": "",
}


def _task(
    seed_id: str = "BigCodeBench/1",
    **changes: Any,
) -> dict[str, Any]:
    """A task as stage 02 keeps it: double a number."""
    task = {
        "seed_id": seed_id,
        "kept": True,
        "function_name": "double",
        "prompt": "Write a function `double(number)` that returns twice the number.",
        "primary_library": "numpy",
        "stdlib_solution": _RIGHT,
        "extlib_solution": "import numpy\n\n\ndef double(number):\n    return int(numpy.multiply(number, 2))\n",
        "test_inputs": ["[1]", "[5]"],
        "expected_outputs": ["2", "10"],
        "stdlib_lloc": 2,
        "extlib_lloc": 3,
    }
    task |= changes
    return task | {"task_hash": build.task_hash(task)}


@contextmanager
def _no_install(
    python_executable: str,
    packages: list[str],
) -> Iterator[InstallResult]:
    """Stand in for temporary_packages: install nothing."""
    yield InstallResult(failed=set(), versions={})


@pytest.fixture(autouse=True)
def _run_locally(monkeypatch) -> None:
    """Run solutions with this test's python, and never install anything."""
    monkeypatch.setattr(agent_review, "BUILD_PYTHON", sys.executable)
    monkeypatch.setattr(agent_review, "temporary_packages", _no_install)


def _stub_model(
    monkeypatch,
    replies: list[dict],
) -> list[str]:
    """Answer each model call with the next of `replies`.

    Returns the list the requests are recorded in.
    """
    requests: list[str] = []

    def fake_complete_json(prompt: str, **kwargs: Any) -> tuple[dict, str]:
        requests.append(prompt)
        return replies[len(requests) - 1], "claude-opus-5-5"

    monkeypatch.setattr(agent_review, "complete_json", fake_complete_json)
    return requests


class TestReviewTask:
    """A review is two blind solves, then a rubric call that sees how they did."""

    def test_a_clean_review_keeps_the_task(self, monkeypatch) -> None:
        requests = _stub_model(
            monkeypatch,
            [{"solution": _RIGHT}, {"solution": _RIGHT}, _KEEP],
        )
        task = _task()

        record = agent_review.review_task(task)

        # the blind solves see only the task text, never a solution or a test
        for request in requests[:2]:
            assert task["prompt"] in request
            assert "numpy" not in request and "10" not in request
        assert "Use only the Python standard library" in requests[0]
        assert "passed every test" in requests[2]
        assert record["verdict"] == "keep"
        assert record["task_hash"] == task["task_hash"]
        assert record["rubric"]["tests_fair"] is True
        assert record["blind"]["stdlib"]["passed"]
        assert record["blind"]["free"]["passed"]

    def test_a_failed_blind_solve_is_recorded_and_shown_to_the_rubric(
        self,
        monkeypatch,
    ) -> None:
        fix = _KEEP | {
            "spec_complete": False,
            "verdict": "fix",
            "note": "say twice what",
        }
        requests = _stub_model(
            monkeypatch,
            [{"solution": _WRONG}, {"solution": _RIGHT}, fix],
        )

        record = agent_review.review_task(_task())

        failed = record["blind"]["stdlib"]
        assert not failed["passed"]
        assert "wrong result on test 1" in failed["message"]
        assert failed["solution"] == _WRONG
        assert "standard-library-only attempt FAILED" in requests[2]
        assert "return number + 2" in requests[2]
        assert record["verdict"] == "fix"
        assert record["note"] == "say twice what"
        assert record["rubric"]["spec_complete"] is False

    def test_a_standard_library_solve_may_not_import_a_library(
        self,
        monkeypatch,
    ) -> None:
        _stub_model(
            monkeypatch,
            [{"solution": "import numpy\n" + _RIGHT}, {"solution": _RIGHT}, _KEEP],
        )
        record = agent_review.review_task(_task())
        solve = record["blind"]["stdlib"]
        assert not solve["passed"]
        assert solve["message"] == "imported a third-party library: ['numpy']"

    def test_the_free_solve_records_the_libraries_it_chose(self, monkeypatch) -> None:
        # regex is installed with the project, so the solve really runs with it
        with_library = "import regex\n\n\ndef double(number):\n    return 2 * number\n"
        _stub_model(
            monkeypatch,
            [{"solution": _RIGHT}, {"solution": with_library}, _KEEP],
        )
        record = agent_review.review_task(_task())
        assert record["blind"]["free"]["passed"]
        assert record["blind"]["free"]["libraries"] == ["regex"]

    @pytest.mark.parametrize(
        "replies",
        [
            [{}],  # an unreadable blind solve
            [{"solution": _RIGHT}, {"solution": _RIGHT}, {}],  # an unreadable rubric
        ],
    )
    def test_an_unreadable_reply_leaves_no_record(
        self,
        monkeypatch,
        replies: list[dict],
    ) -> None:
        _stub_model(monkeypatch, replies)
        with pytest.raises(RuntimeError, match="no usable"):
            agent_review.review_task(_task())

    def test_the_control_rubric_asks_whether_a_library_would_help(self) -> None:
        solve = {"passed": True, "message": "", "libraries": [], "solution": _RIGHT}
        request = agent_review.build_rubric_request(
            _task(),
            {"stdlib": solve, "free": solve},
            track="control",
        )
        assert "is this a task where no third-party library would help?" in request
        assert "library solution, using" not in request


class TestWhatStandsAfterReview:
    """A review counts only for the version of the task it looked at."""

    @pytest.fixture(autouse=True)
    def _artefacts(self, tmp_path, monkeypatch) -> None:
        """Four tasks: kept, to fix, already regenerated for a fix, and changed since review."""
        monkeypatch.setattr(build, "ARTEFACTS_DIR", tmp_path)
        tasks = [
            _task("BigCodeBench/1"),
            _task("BigCodeBench/2"),
            _task("BigCodeBench/3", fixed_from="an earlier hash"),
            _task("BigCodeBench/4"),
        ]
        save_jsonl(tasks, tmp_path / "02_generated.jsonl")
        # each task's review: its verdict, and whether it is of the task's current version.
        # task 4 has changed since it was reviewed
        reviews = [
            {
                "seed_id": task["seed_id"],
                "verdict": verdict,
                "task_hash": task["task_hash"] if current else "an earlier hash",
            }
            for task, verdict, current in zip(
                tasks,
                ["keep", "fix", "fix", "keep"],
                [True, True, True, False],
            )
        ]
        save_jsonl(reviews, tmp_path / "03_agent_reviews.jsonl")

    def test_a_stale_review_does_not_count(self) -> None:
        assert set(build.agent_reviews()) == {
            "BigCodeBench/1",
            "BigCodeBench/2",
            "BigCodeBench/3",
        }
        assert [task["seed_id"] for task in build.agent_kept_tasks()] == [
            "BigCodeBench/1"
        ]

    def test_a_task_is_only_sent_back_for_a_fix_once(self) -> None:
        assert agent_review.seeds_to_fix() == ["BigCodeBench/2"]
