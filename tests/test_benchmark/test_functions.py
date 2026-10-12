"""Tests for scoring function tasks in libreusebench/functions.py: lenient matching and running tests."""

import sys
from decimal import Decimal
from fractions import Fraction

import pytest
from libreusebench.functions import RAISES, evaluate_function_record, results_match


class _ArrayLike:
    """A stand-in for a numpy array: plain python comes out of its tolist method."""

    def __init__(
        self,
        values: list,
    ) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


class _SeriesLike:
    """A stand-in for a pandas or polars series: it converts with to_dict and to_list."""

    def __init__(
        self,
        data: dict,
    ) -> None:
        self.data = data

    def to_dict(self) -> dict:
        return dict(self.data)

    def to_list(self) -> list:
        return list(self.data.values())


class _ScalarLike:
    """A stand-in for a numpy number: the plain python number comes out of its item method."""

    def __init__(
        self,
        value: float,
    ) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


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

    @pytest.mark.parametrize(
        ("got", "expected"),
        [
            # a dict is expected: to_dict if the value has it, else dict()
            (_SeriesLike({"a": 1, "b": 2}), {"a": 1, "b": 2}),
            ([("a", 1), ("b", 2)], {"a": 1, "b": 2}),
            # a list, tuple or set is expected: tolist / to_list, else list() of an iterable
            (_SeriesLike({"a": 1, "b": 2}), [1, 2]),
            (range(1, 3), [1, 2]),
            ({"a": 1, "b": 2}.values(), (1, 2)),
            (_ArrayLike([2, 1]), {1, 2}),
            # a number is expected: item if the value has it, else float()
            (_ScalarLike(5), 5),
            (Decimal("0.3"), 0.3),
            (Fraction(1, 2), 0.5),
            # and the same at every level of nesting
            ({"a": _ArrayLike([1, 2])}, {"a": [1, 2]}),
            ([_ScalarLike(1.5), Decimal("2")], [1.5, 2]),
            (_SeriesLike({"a": _ScalarLike(1)}), {"a": 1}),
            ({_ScalarLike(2020), _ScalarLike(2021)}, [2021, 2020]),
        ],
    )
    def test_library_values_are_converted_to_the_expected_type(
        self,
        got: object,
        expected: object,
    ) -> None:
        assert results_match(got, expected)

    @pytest.mark.parametrize(
        ("got", "expected"),
        [
            # a string is never a number, and a string, bytes or dict is never a list
            ("1.5", 1.5),
            ("ab", ["a", "b"]),
            (b"ab", [97, 98]),
            ({"a": 1}, ["a"]),
            # converting doesn't hide a wrong value
            (_SeriesLike({"a": 1}), {"a": 2}),
            (_ScalarLike(5), 6),
            # a value that can't be compared simply doesn't match
            (_ArrayLike([1, 2]), None),
            (_ArrayLike([1, 2]), 1),
        ],
    )
    def test_conversion_is_not_too_lenient(
        self,
        got: object,
        expected: object,
    ) -> None:
        assert not results_match(got, expected)

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
    return evaluate_function_record(
        code=code,
        function_name="double",
        test_inputs=test_inputs or ["[1]", "[2.5]"],
        expected_outputs=expected_outputs,
        python_executable=sys.executable,
    )


class TestEvaluateFunctionRecord:
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
