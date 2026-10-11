"""Tests for scoring the function-style LibReuseBench pilot: lenient matching and running tests."""

import sys
from pathlib import Path

import pytest

from src.eval.scoring import score_dataset
from src.eval.scoring.libreusebench_function import (
    RAISES,
    results_match,
    run_function_tests,
)


class _ArrayLike:
    """A stand-in for a numpy array: plain python comes out of its tolist method."""

    def __init__(
        self,
        values: list,
    ) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


class TestResultsMatch:
    """Return values are compared leniently: equivalent values match, different ones don't."""

    @pytest.mark.parametrize(
        ("got", "expected"),
        [
            (0.30000000001, 0.3),
            (100.04, 100.0),
            ((1, 2), [1, 2]),
            ([[1.0, 2.0]], ((1, 2),)),
            ({2, 1}, {1, 2}),
            ([2, 1], {1, 2}),
            ({5: 1, 7: [1, 2]}, {7: (1, 2), 5: 1}),
            ("a  b\n", "a b"),
            (_ArrayLike([1, 2]), [1, 2]),
            (None, None),
            (float("nan"), float("nan")),
        ],
    )
    def test_equivalent_values_match(self, got: object, expected: object) -> None:
        assert results_match(got, expected)

    @pytest.mark.parametrize("rel_tol", [1e-3, 1e-2])
    def test_neighbouring_numeric_keys_are_not_confused(self, rel_tol: float) -> None:
        """2021 is "close" to 2020, but must still pair with its own key."""
        assert results_match(
            {2021: 3, 2020: 2},
            {2020: 2, 2021: 3},
            rel_tol=rel_tol,
        )
        assert results_match({2021, 2020}, [2020, 2021], rel_tol=rel_tol)
        assert not results_match(
            {2021: 2, 2020: 3},
            {2020: 2, 2021: 3},
            rel_tol=rel_tol,
        )

    def test_a_float_key_can_match_leniently(self) -> None:
        assert results_match({0.30000000001: 1}, {0.3: 1})

    @pytest.mark.parametrize(
        ("got", "expected"),
        [
            (0.31, 0.3),
            ([1, 2], [2, 1]),
            ([1, 2], [1, 2, 3]),
            ({5: 1}, {5: 2}),
            ({5: 1}, {6: 1}),
            ("ab", "a b"),
            ("5", 5),
            (None, 0),
        ],
    )
    def test_different_values_do_not_match(
        self,
        got: object,
        expected: object,
    ) -> None:
        assert not results_match(got, expected)


def _run(
    code: str,
    expected_outputs: list[str],
    test_inputs: list[str] | None = None,
) -> tuple[bool, str]:
    """Run `code`'s `double` function against cases, with this test's own python."""
    return run_function_tests(
        code=code,
        function_name="double",
        test_inputs=test_inputs or ["[1]", "[2.5]"],
        expected_outputs=expected_outputs,
        python_executable=sys.executable,
    )


class TestRunFunctionTests:
    """A response passes only if its named function returns a matching value for every case."""

    def test_correct_function_passes(self) -> None:
        assert _run("def double(x):\n    return 2 * x\n", ["2", "5.0"]) == (True, "")

    def test_demo_block_and_stray_prints_are_ignored(self) -> None:
        code = (
            "print('loading')\n"
            "def double(x):\n"
            "    print('working')\n"
            "    return 2 * x\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit('the demo block ran')\n"
        )
        assert _run(code, ["2", "5.0"]) == (True, "")

    def test_missing_function_fails(self) -> None:
        passed, message = _run("def triple(x):\n    return 3 * x\n", ["2", "5.0"])
        assert not passed and "function not defined" in message

    def test_wrong_result_fails(self) -> None:
        passed, message = _run("def double(x):\n    return x\n", ["2", "5.0"])
        assert not passed and "wrong result on test 1" in message

    def test_an_expected_exception_can_be_any_exception(self) -> None:
        code = "def double(x):\n    if x < 0:\n        raise TypeError('negative')\n    return 2 * x\n"
        assert _run(code, ["2", RAISES], test_inputs=["[1]", "[-1]"]) == (True, "")

    def test_returning_where_an_exception_is_expected_fails(self) -> None:
        passed, message = _run(
            "def double(x):\n    return 2 * x\n",
            ["2", RAISES],
            test_inputs=["[1]", "[-1]"],
        )
        assert not passed and "expected an exception" in message


class _FakeEnvironment:
    """A stand-in for the benchmark's venv: runs code with this test's own python."""

    def __init__(
        self,
        cache_dir: str | None = None,
        scratch_root: str | None = None,
    ) -> None:
        self.python = Path(sys.executable)

    def __enter__(self) -> "_FakeEnvironment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def ensure_packages(
        self,
        packages: set[str],
    ) -> set[str]:
        return set()


class TestScoreLibReuseBenchFunction:
    """Every sample is tested, and the summary has the official evaluator's fields."""

    def test_samples_are_scored_and_aggregated(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "src.eval.scoring.libreusebench_function.Environment",
            _FakeEnvironment,
        )
        dataset = [
            {
                "idx": 0,
                "task_id": "LRB/T000",
                "function_name": "double",
                "test_inputs": ["[1]", "[3]"],
                "expected_outputs": ["2", "6"],
            }
        ]
        responses = [
            [
                ("```python\ndef double(x):\n    return 2 * x\n```", "stop"),
                ("```python\ndef double(x):\n    return x + 1\n```", "stop"),
                ("no code here", "stop"),
            ]
        ]

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="libreusebench_function",
            environment_root=str(tmp_path),
        )

        assert analysis["task"] == "libreusebench_function"
        assert analysis["n_total"] == 3
        assert analysis["pass@1"] == pytest.approx(1 / 3)
        assert [result["passed"] for result in results] == [True, False, False]
        assert results[2]["reason"] == "no code"
        assert "wrong result on test 2" in results[1]["reason"]
