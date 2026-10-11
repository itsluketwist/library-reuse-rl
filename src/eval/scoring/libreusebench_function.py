"""Score function-style LibReuseBench responses: call the named function on each test case.

The function-style pilot (benchmark/pipeline/function_pilot/) rewords each LibReuseBench task as
a request for one named Python function. Its tests are plain pairs - the function's arguments and
its expected return value, both written as Python literals in the row's `test_inputs` and
`expected_outputs` - and a response passes when its function returns a matching value for every
case. Values are compared leniently (see results_match), since this pilot cares more about
whether a model reaches for a library than about exact output formatting.

Everything else follows the official stdin/stdout evaluator: the same code block is picked, the
same packages are installed, and the summary has the same fields (see score_libreusebench).
"""

import inspect
from typing import Any, Iterable

from libreusebench.environment import Environment, execute
from libreusebench.evaluate import _aggregate, plan_installs
from libreusebench.libraries import (
    extract_code,
    imported_libraries,
    pip_package,
    used_libraries,
)

from src.response.metrics import code_metrics


# marks a test case where the function should reject its input by raising an exception. it is
# not a valid python literal, so it can never be mistaken for a real expected value
RAISES = "!raises"

# how close a returned number must be to the expected one to count as a match
REL_TOL = 1e-3
ABS_TOL = 1e-6

# seconds allowed for one response to run every one of its test cases (as for stdin/stdout tasks)
TIMEOUT_SECONDS = 30.0


def results_match(
    got: object,
    expected: object,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """
    Compare a function's return value with the expected one, leniently.

    Numbers match within `rel_tol` / `abs_tol`, lists and tuples are interchangeable, sets (and a
    list or tuple compared with a set) ignore order, dicts compare key by key, and strings ignore
    differences in whitespace. numpy and pandas values are turned into plain python first, via
    their `tolist` / `item` / `to_numpy` methods, so neither library needs importing.

    This function is self-contained (its import and helpers are inside it) because its source is
    copied into the program that tests each response - see build_test_program.

    Returns True if the two values match.
    """
    import math

    def plain(value: object) -> object:
        # numpy arrays and scalars and pandas series have tolist, a pandas dataframe has to_numpy
        if isinstance(value, (str, bytes)):
            return value
        for method in ("tolist", "item", "to_numpy"):
            if hasattr(value, method):
                try:
                    return plain(getattr(value, method)())
                except Exception:
                    continue
        return value

    def match(a: object, b: object) -> bool:
        a, b = plain(a), plain(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, float) and isinstance(b, float):
                if math.isnan(a) and math.isnan(b):
                    return True
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if isinstance(a, str) and isinstance(b, str):
            return a.split() == b.split()
        sequences = (list, tuple)
        sets = (set, frozenset)
        if isinstance(a, sequences) and isinstance(b, sequences):
            return len(a) == len(b) and all(match(x, y) for x, y in zip(a, b))
        if isinstance(a, sets + sequences) and isinstance(b, sets + sequences):
            # at least one is a set, so order doesn't matter: pair each expected item with a
            # different returned item - identical items first, so that a lenient match can't
            # take an item another one needs (e.g. 2021 standing in for 2020)
            remaining = [plain(x) for x in a]
            if len(remaining) != len(b):
                return False
            unmatched = []
            for y in (plain(y) for y in b):
                exact = [
                    i for i, x in enumerate(remaining) if type(x) is type(y) and x == y
                ]
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
            got_items = {plain(key): value for key, value in a.items()}
            for key_b, value_b in b.items():
                key_b = plain(key_b)
                if key_b in got_items:
                    key_a = key_b
                else:
                    # no identical key, so accept a lenient one - but only if exactly one
                    # returned key is close, so neighbouring keys (2020, 2021) never mix up
                    close = [key_a for key_a in got_items if match(key_a, key_b)]
                    if len(close) != 1:
                        return False
                    key_a = close[0]
                if not match(got_items[key_a], value_b):
                    return False
            return True
        return a == b

    return match(got, expected)


# the program that tests one response: it defines results_match, loads the response's code, then
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

# run the response's code as a module named "solution", so its own `if __name__ == "__main__"`
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
    """
    Build the program that tests one response's function against a task's cases.

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


def run_function_tests(
    code: str,
    function_name: str,
    test_inputs: list[str],
    expected_outputs: list[str],
    python_executable: str,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """
    Run one response's function against a task's cases, in a sandboxed subprocess.

    Returns a (passed, empty string or the failure message) tuple.
    """
    passed, output = execute(
        code=build_test_program(
            code=code,
            function_name=function_name,
            test_inputs=test_inputs,
            expected_outputs=expected_outputs,
        ),
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    return passed, "" if passed else output


def _score_response(
    response: str,
    row: dict,
    python_executable: str,
    install_failed: set[str],
) -> dict[str, Any]:
    """
    Score one response, the way the official evaluator does for a stdin/stdout task.

    The code is only run when every package it imports was installed; otherwise it counts as
    not passed, with `ran` False and the missing packages in `install_failed`.

    Returns the response's result (passed, libraries, code size, ...).
    """
    code = extract_code(response)
    if code is None:
        return {
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

    external = imported_libraries(response)
    used = used_libraries(response)
    lloc, maintainability_index = code_metrics(code)
    details = {
        "external_libs": sorted(external),
        "external_lib": bool(external),
        "used_external_libs": sorted(used),
        "external_lib_used": bool(used),
        "lloc": lloc,
        "maintainability_index": maintainability_index,
    }

    # the packages this code imports that failed to install - it can't run without them
    missing = sorted({pip_package(name) for name in external} & install_failed)
    if missing:
        return {
            "passed": False,
            **details,
            "ran": False,
            "install_failed": missing,
            "reason": f"not run: could not install {', '.join(missing)}",
        }

    passed, message = run_function_tests(
        code=code,
        function_name=row["function_name"],
        test_inputs=row["test_inputs"],
        expected_outputs=row["expected_outputs"],
        python_executable=python_executable,
    )
    return {
        "passed": passed,
        **details,
        "ran": True,
        "install_failed": [],
        "reason": message,
    }


def score_libreusebench_function(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Score function-style LibReuseBench rows: install what the responses import, then test them.

    Every sampled response of a row is scored. As in score_libreusebench, the packages the
    responses import are installed automatically into a disposable venv under
    `environment_root` (or the reusable one at `environment_cache_dir`), and a package that
    fails to install doesn't stop the run - the responses that import it count as not passed.

    Returns the (analysis, results) pair, with the official evaluator's summary fields.
    """
    del python_executable
    rows = list(dataset)
    response_texts = {
        row["task_id"]: [response for response, _stop_reason in samples]
        for row, samples in zip(rows, responses)
    }
    packages = set(plan_installs(response_texts))

    results: list[dict[str, Any]] = []
    with Environment(
        cache_dir=environment_cache_dir,
        scratch_root=environment_root,
    ) as environment:
        install_failed = environment.ensure_packages(packages)
        for row, samples in zip(rows, responses):
            for sample_idx, (response, stop_reason) in enumerate(samples):
                result = _score_response(
                    response=response,
                    row=row,
                    python_executable=str(environment.python),
                    install_failed=install_failed,
                )
                results.append(
                    {
                        "task_id": row["task_id"],
                        "idx": sample_idx,
                        **result,
                        "stop_reason": stop_reason,
                        "response": response,
                    }
                )

    analysis = {
        "task": "libreusebench_function",
        "n_total": len(results),
        **_aggregate(results),
        "environment": {
            "installed_packages": sorted(packages - install_failed),
            "failed_packages": sorted(install_failed),
        },
    }
    return analysis, results
