"""Score function tasks: call the named function on each test case and compare the results.

A function task asks for one named Python function. Its tests are plain pairs - the function's
arguments and its expected return value, both written as Python literals in the record's
`test_inputs` and `expected_outputs` - and a solution passes when its function returns a
matching value for every case. Values are compared leniently (see results_match): the benchmark
measures whether a model reaches for a library, so a correct answer must not fail because of the
type a library returns (a numpy array where a list is expected, a pandas series for a dict).
"""

import inspect
from pathlib import Path

from libreusebench.environment import _DEFAULT_TIMEOUT_SECONDS, BUILD_PYTHON, execute


# marks a test case where the function should reject its input by raising an exception. it is
# not a valid python literal, so it can never be mistaken for a real expected value
RAISES = "!raises"

# how close a returned number must be to the expected one to count as a match
REL_TOL = 1e-3
ABS_TOL = 1e-6


def results_match(
    got: object,
    expected: object,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """Compare a function's return value with the expected one, leniently.

    The returned value is first turned into the plain python type of the expected value, at
    every level of nesting, so that a library's own types are accepted:
    - where a dict is expected, `to_dict()` is used if the value has it, else `dict(...)`;
    - where a list, tuple or set is expected, `tolist()` / `to_list()` is used if the value has
      it, else `list(...)` for any other iterable (but never for a string, bytes or a dict);
    - where a number is expected, `item()` is used if the value has it, else `float(...)` for
      number types such as Decimal and Fraction. A string is never turned into a number.

    Then numbers match within `rel_tol` / `abs_tol`, lists and tuples are interchangeable, sets
    (and a list or tuple compared with a set) ignore order, dicts compare key by key, and
    strings ignore differences in whitespace.

    This function is self-contained (its import and helpers are inside it, and its defaults are
    written out) because its source is copied into the program that tests each solution - see
    build_test_program.

    Returns True if the two values match.
    """
    import math
    from typing import Any

    sequences = (list, tuple)
    sets = (set, frozenset)

    def coerce(value: Any, target: object) -> Any:
        # turn a returned value into the plain python type of the expected one. a value that
        # can't be converted is returned unchanged, and then simply fails to match
        try:
            if isinstance(target, dict) and not isinstance(value, dict):
                # e.g. a pandas series, or a list of (key, value) pairs
                return value.to_dict() if hasattr(value, "to_dict") else dict(value)
            if isinstance(target, sequences + sets) and not isinstance(
                value, sequences + sets + (str, bytes, dict)
            ):
                # numpy arrays and pandas series have tolist, polars has to_list, and a pandas
                # dataframe has to_numpy (which gives an array, to convert in turn)
                for method in ("tolist", "to_list", "to_numpy"):
                    if hasattr(value, method):
                        return coerce(getattr(value, method)(), target)
                return list(value)
            if isinstance(target, (int, float)) and not isinstance(
                value, (int, float, str, bytes)
            ):
                # numpy numbers have item, and Decimal and Fraction convert with float
                return value.item() if hasattr(value, "item") else float(value)
        except Exception:
            pass
        return value

    def identical(a: Any, b: object) -> bool:
        a = coerce(a, b)
        return type(a) is type(b) and a == b

    def match(a: Any, b: object) -> bool:
        a = coerce(a, b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, float) and isinstance(b, float):
                if math.isnan(a) and math.isnan(b):
                    return True
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if isinstance(a, str) and isinstance(b, str):
            return a.split() == b.split()
        if isinstance(a, sequences) and isinstance(b, sequences):
            return len(a) == len(b) and all(match(x, y) for x, y in zip(a, b))
        if isinstance(a, sets + sequences) and isinstance(b, sets + sequences):
            # at least one is a set, so order doesn't matter: pair each expected item with a
            # different returned item - identical items first, so that a lenient match can't
            # take an item another one needs (e.g. 2021 standing in for 2020)
            remaining = list(a)
            if len(remaining) != len(b):
                return False
            unmatched = []
            for y in b:
                exact = [i for i, x in enumerate(remaining) if identical(x, y)]
                if exact:
                    del remaining[exact[0]]
                else:
                    unmatched.append(y)
            for y in unmatched:
                close = [i for i, x in enumerate(remaining) if match(x, y)]
                if not close:
                    return False
                del remaining[close[0]]
            return True
        if isinstance(a, dict) and isinstance(b, dict):
            if len(a) != len(b):
                return False
            for key_b, value_b in b.items():
                if key_b in a:
                    key_a = key_b
                else:
                    # no identical key, so accept a lenient one - but only if exactly one
                    # returned key is close, so neighbouring keys (2020, 2021) never mix up
                    close = [key_a for key_a in a if match(key_a, key_b)]
                    if len(close) != 1:
                        return False
                    key_a = close[0]
                if not match(a[key_a], value_b):
                    return False
            return True
        try:
            return bool(a == b)
        except Exception:
            # e.g. comparing a numpy array with None gives an array, which is not a yes or no
            return False

    return match(got, expected)


# the program that tests one solution: it defines results_match, loads the solution's code, then
# calls the function on every case. it exits with a message at the first failing case
_TEST_PROGRAM = """\
import ast
import contextlib
import io

{results_match_source}

SOLUTION = {code!r}
FUNCTION_NAME = {function_name!r}
CASES = {cases!r}
RAISES = {raises!r}

# run the solution's code as a module named "solution", so its own `if __name__ == "__main__"`
# demo block is skipped, and swallow anything it prints
namespace = {{"__name__": "solution"}}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(SOLUTION, "solution.py", "exec"), namespace)
function = namespace.get(FUNCTION_NAME)
if not callable(function):
    raise SystemExit("function not defined: " + FUNCTION_NAME)

for number, (arguments, expected) in enumerate(CASES, start=1):
    # the arguments are rebuilt from their literal for every case, so a function that changes
    # its arguments can't affect another case
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = function(*ast.literal_eval(arguments))
    except Exception as error:
        if expected == RAISES:
            continue
        raise SystemExit(f"raised on test {{number}}: {{type(error).__name__}}: {{error}}")
    if expected == RAISES:
        raise SystemExit(f"wrong result on test {{number}}: expected an exception, got {{result!r}}")
    if not results_match(result, ast.literal_eval(expected), {rel_tol!r}, {abs_tol!r}):
        raise SystemExit(f"wrong result on test {{number}}: expected {{expected}}, got {{result!r}}")
"""


def build_test_program(
    code: str,
    function_name: str,
    test_inputs: list[str],
    expected_outputs: list[str],
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
) -> str:
    """Build the program that tests one solution's function against a task's cases.

    Each case is an argument list and an expected return value, both as python literals (or
    RAISES, where the function should raise an exception - any exception counts).

    Returns the program's source, which exits normally only if every case passes.
    """
    return _TEST_PROGRAM.format(
        results_match_source=inspect.getsource(results_match),
        code=code,
        function_name=function_name,
        cases=list(zip(test_inputs, expected_outputs)),
        raises=RAISES,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def evaluate_function_record(
    code: str,
    function_name: str,
    test_inputs: list[str],
    expected_outputs: list[str],
    python_executable: str | Path = BUILD_PYTHON,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
) -> tuple[bool, str]:
    """Check a solution's function against a function task's (test_inputs, expected_outputs).

    The function-task counterpart of environment.evaluate_record, run in the same sandbox.
    `rel_tol` and `abs_tol` say how close a returned number must be (see results_match) - 0 for
    both checks numbers exactly.

    Returns a (passed, empty string or the failure message) tuple.
    """
    passed, output = execute(
        code=build_test_program(
            code=code,
            function_name=function_name,
            test_inputs=test_inputs,
            expected_outputs=expected_outputs,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    return passed, "" if passed else output
