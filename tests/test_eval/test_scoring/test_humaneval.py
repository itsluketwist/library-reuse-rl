"""Tests for scoring HumanEval+ responses."""

import sys

import pytest

from src.eval.scoring import humaneval, score_dataset


# a tiny HumanEval+-style task: check() is called on the answer's function
TASK = {
    "task_id": "HumanEval/X",
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
}

CORRECT = "```python\ndef add(a, b):\n    return a + b\n```"


class FakeEnvironment:
    """Stands in for libreusebench's Environment: runs with this python, installs nothing."""

    # the packages each instance was asked to install, and which of them should "fail"
    requested: set[str] = set()
    failing: set[str] = set()

    def __init__(self, cache_dir: str | None, scratch_root: str | None) -> None:
        self.python = sys.executable

    def __enter__(self) -> "FakeEnvironment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def ensure_packages(self, packages: set[str]) -> set[str]:
        FakeEnvironment.requested = packages
        return packages & FakeEnvironment.failing

    def warm_up(self, import_names: set[str]) -> None:
        pass

    def installed_distributions(self) -> dict[str, str]:
        return {"numpy": "2.4.6"}


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch):
    """Score every test with FakeEnvironment, so no venv is built or package installed."""
    FakeEnvironment.failing = set()
    monkeypatch.setattr(humaneval, "Environment", FakeEnvironment)


def _score(
    *samples: str,
) -> tuple[dict, list[dict]]:
    """Score the given responses as samples of TASK.

    Returns the (analysis, results) pair.
    """
    return score_dataset(
        [TASK],
        [[(sample, "stop", 10) for sample in samples]],
        evaluator="humaneval",
    )


class TestScoreHumanEval:
    """Answers are run against the task's check(), and pass@k counts the samples."""

    def test_pass_at_k_over_correct_wrong_and_missing_code(self):
        """One passing sample of three gives pass@1 of a third, and pass@3 of one."""
        wrong = "```python\ndef add(a, b):\n    return a - b\n```"
        analysis, results = _score(CORRECT, wrong, "no code at all")

        assert [result["passed"] for result in results] == [True, False, False]
        assert analysis["pass@1"] == 1 / 3
        assert analysis["pass@3"] == 1.0
        assert analysis["n_no_code"] == 1
        assert "AssertionError" in results[1]["output"]

    def test_answer_names_cannot_clash_with_the_test(self):
        """An answer's own `check` and a demo block reading stdin don't break the test."""
        answer = (
            "```python\n"
            "def check(x):\n"
            "    return x\n"
            "\n"
            "def add(a, b):\n"
            "    return check(a) + b\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(add(int(input()), 1))\n"
            "```"
        )
        _analysis, results = _score(answer)

        assert results[0]["passed"], results[0]["output"]

    def test_installs_numpy_and_the_answers_libraries(self):
        """The venv gets numpy for the tests, plus each package an answer imports."""
        answer = "```python\nimport sympy\n\ndef add(a, b):\n    return a + b\n```"
        analysis, _results = _score(answer)

        assert FakeEnvironment.requested == {"numpy", "sympy"}
        assert analysis["environment"]["installed_packages"] == ["numpy", "sympy"]

    def test_missing_numpy_stops_scoring(self):
        """Without numpy every test would fail, so scoring stops instead."""
        FakeEnvironment.failing = {"numpy"}
        with pytest.raises(RuntimeError, match="numpy"):
            _score(CORRECT)
