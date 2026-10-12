"""Pilot: reword LibReuseBench tasks as requests for one named function, with input/output tests.

The benchmark poses every task as a stdin/stdout program, which may itself put models off using
libraries (it reads like a competitive-programming exercise). This pilot rewords each task in the
style of BigCodeBench, but without its imports or code stub: a short request for a function with a
given name, natural python arguments, and one example call. Opus is given everything we have - the
original BigCodeBench task (description, reference solution and tests), the benchmark's
stdin/stdout rewrite, both gold solutions and the test cases - and asked for the reworded task,
the gold solutions converted into that function, and a converter for the tests.

The tests become plain pairs of the function's arguments and its expected return value:
- the arguments come from running Opus's throwaway `convert_input` on each original test input
  (copying long inputs out by hand is error-prone);
- the expected value is what the converted stdlib gold solution returns for those arguments;
- the conversion is checked: the stdlib gold must match Opus's transcription of each original
  printed output (within the original's rounding), and the converted library gold must agree
  with the stdlib gold - so the function task keeps the original's behaviour.

Responses are scored by calling their function on each case and comparing leniently, see
src/eval/scoring/libreusebench_function.py. Accepted rewrites are cached in function_tasks.json,
and test.jsonl / control.jsonl are rebuilt from the cache after every run. Point LRB_BUILD_PYTHON
at the build venv (benchmark/pipeline/setup_build_venv.sh) - it defaults to .venv-build.

Usage: python benchmark/pipeline/function_pilot/generate.py [--split test|control]
       [--tasks ID,ID] [--limit N] [--skip-cached] [--workers N] [--model M] [--effort E]
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# the numbered stages import their shared helpers (build.py, generation.py) by name from
# benchmark/pipeline, which python puts on the path when it runs them - this script sits one
# folder deeper, so that folder is added by hand
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import DATA_DIR, artefact_path, full_solution  # noqa: E402
from generation import complete_json  # noqa: E402
from libreusebench.environment import execute, temporary_packages  # noqa: E402
from libreusebench.libraries import (  # noqa: E402
    analyse_source,
    find_named_libraries,
    required_pip_packages,
)
from llm_cgr import save_jsonl  # noqa: E402
from lure import analyse_code  # noqa: E402

from src.eval.scoring.libreusebench_function import (  # noqa: E402
    ABS_TOL,
    RAISES,
    REL_TOL,
    results_match,
    run_function_tests,
)
from src.utils.io import load_jsonl  # noqa: E402


PILOT_DIR = Path(__file__).resolve().parent
CACHE_FILE = PILOT_DIR / "function_tasks.json"
REPO_ROOT = PILOT_DIR.parents[2]

# the interpreter the generated code is run with while checking it: the isolated build venv, so
# the library solutions' packages are installed there (and removed again), never in the project's
VALIDATION_PYTHON = os.environ.get("LRB_BUILD_PYTHON") or str(
    REPO_ROOT / ".venv-build" / "bin" / "python"
)

# the generation model and how hard it thinks (opus 5.5 always thinks; its default effort is medium)
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"

# the longest reworded task accepted, in characters - it should read as a short request
MAX_PROMPT_CHARS = 1200

# how close the stdlib gold's result must be to the transcribed original output. looser than the
# eval's tolerance, since the original printed rounded numbers (e.g. 3.47 printed as "3.5")
ANCHOR_REL_TOL = 1e-2
ANCHOR_ABS_TOL = 5e-2

# names too generic to say what a function does
_GENERIC_NAMES = {"task_func", "solve", "solution", "main", "func", "function", "f"}

# the stage-01 BigCodeBench seeds each split was built from
_SEED_FILES = {
    "test": "01_bcb_extlibs.jsonl",
    "control": "01_bcb_stdlibs.jsonl",
}

# installing into the shared build venv isn't safe from two threads at once
_INSTALL_LOCK = threading.Lock()


_SYSTEM = (
    "You turn coding benchmark problems into short, library-agnostic function-writing tasks, and "
    "convert their reference solutions and tests to match. Reply with JSON only."
)

_PROMPT = """\
Here is one coding problem, in its original form and as a stdin/stdout program, with reference
solutions and tests. They are given only so you understand the exact required behaviour.

## The original task (from BigCodeBench)

{bcb_description}

Its reference solution:
```python
{bcb_solution}
```

Its tests:
```python
{bcb_test}
```

## The same task, reworded as a stdin/stdout program

{stdio_prompt}

A standard-library solution of this version:
```python
{gold_stdlib}
```
{gold_extlib_section}
Its test cases, numbered, each as the standard input and then the expected standard output:

{cases}

## What to write

Rewrite the task as a short request for ONE Python function, in plain natural language:
- name the function and its parameters, in order, in the text - e.g. "Write a function
  `count_by_length(letter, words)` that ...". Use a descriptive snake_case name that says what the
  function does: not task_func, solve or main, and not the name of any library's function. Give no
  imports, signature stub, type hints or any other code apart from the one example call;
- the parameters are natural Python values (numbers, strings, lists, tuples, dicts, booleans,
  None), never a library's type. If parsing a text format is the point of the task (e.g. CSV,
  HTML, dates), take that text as a string; otherwise pass the data already parsed;
- say WHAT the function returns, not HOW to compute it: name no library, module, import, class
  or library function, and don't narrate an algorithm or hint at an implementation;
- pin down the return value's structure (e.g. "a dict mapping each length to its count") and any
  ordering or tie-breaking rules, so every input has exactly one correct result. Return values
  are compared leniently (numbers within a small tolerance, lists and tuples alike, whitespace
  ignored), so don't ask for rounding or string formatting unless it is part of the task itself;
- the return value is a plain Python value too, so it can be written as a literal;
- where the original rejects invalid input, say the function should raise a ValueError then;
- give exactly one example, as a call and its return value, e.g. "For example,
  `count_by_length('a', ['apple', 'kiwi', 'avocado'])` returns `{{5: 1, 7: 1}}`.";
- keep it short: at most {max_chars} characters.

Then convert the solutions and the tests:
- stdlib_solution: the standard-library solution, rewritten as the function, with the same
  behaviour - standard library only, and no top-level code besides imports and definitions;
- extlib_solution: {extlib_instruction}
- convert_input: Python code defining `convert_input(stdin_text)`, which turns one original test
  input into the list of positional arguments for the function (standard library only - it is
  only used to convert the tests);
- transcribed_outputs: one entry per test case above, in order - the value the function should
  return for that case, as a Python literal transcribed from its expected standard output (at
  the precision it was printed), or exactly "{raises}" where that output reports invalid input;
- example_call and example_result: the example from your text, exactly as written there (the
  call as a Python expression, the result as a Python literal).

If the task can't honestly be posed as one function taking and returning plain Python values,
set fits_function to false, say why in reason, and leave the other fields empty.
{feedback}"""

_GOLD_EXTLIB_SECTION = """
A solution of this version that uses a third-party library:
```python
{gold_extlib}
```
"""

_EXTLIB_INSTRUCTION = (
    "the third-party-library solution, rewritten as the function, with the same behaviour and "
    "still using the same library - no top-level code besides imports and definitions;"
)
_NO_EXTLIB_INSTRUCTION = "leave empty (this task has no library solution);"

# appended when a previous attempt was rejected, so the model can fix what was wrong
_FEEDBACK = """
Your previous attempt was rejected for these reasons - write a new attempt that fixes them:
{problems}
"""

# the reply's fields and their json types - every one is required
_FIELD_TYPES: dict[str, dict[str, Any]] = {
    "fits_function": {"type": "boolean"},
    "reason": {"type": "string"},
    "function_name": {"type": "string"},
    "prompt": {"type": "string"},
    "stdlib_solution": {"type": "string"},
    "extlib_solution": {"type": "string"},
    "convert_input": {"type": "string"},
    "transcribed_outputs": {"type": "array", "items": {"type": "string"}},
    "example_call": {"type": "string"},
    "example_result": {"type": "string"},
}

_SCHEMA = {
    "type": "object",
    "properties": _FIELD_TYPES,
    "required": list(_FIELD_TYPES),
    "additionalProperties": False,
}

# a fingerprint of everything the request is built from, stamped on each cache entry - so
# --skip-cached only skips tasks generated with the current prompt
PROMPT_VERSION = hashlib.sha256(
    "\n".join(
        [
            _SYSTEM,
            _PROMPT,
            _GOLD_EXTLIB_SECTION,
            _EXTLIB_INSTRUCTION,
            _NO_EXTLIB_INSTRUCTION,
            _FEEDBACK,
            str(MAX_PROMPT_CHARS),
        ]
    ).encode()
).hexdigest()[:12]


# --- building the request ----------------------------------------------------------------------


def _format_cases(
    row: dict,
) -> str:
    """Write a task's stdin/stdout test cases out, numbered, for the request.

    Returns the formatted cases.
    """
    return "\n\n".join(
        f"Case {number}:\nInput:\n```\n{stdin}\n```\nOutput:\n```\n{stdout}\n```"
        for number, (stdin, stdout) in enumerate(
            zip(row["test_inputs"], row["expected_outputs"]),
            start=1,
        )
    )


def build_request(
    row: dict,
    seed: dict,
    problems: list[str] | None = None,
) -> str:
    """Build the request asking for one task's function-style rewrite.

    `problems` are a previous attempt's check failures, fed back so the next one can fix them.

    Returns the request text.
    """
    gold_extlib = row["ground_truth"].get("gold_extlib")
    return _PROMPT.format(
        bcb_description=(seed["instruct_prompt"] or seed["complete_prompt"]).strip(),
        bcb_solution=full_solution(
            seed["complete_prompt"],
            seed["canonical_solution"],
        ).strip(),
        bcb_test=seed["test"].strip(),
        stdio_prompt=row["prompt"].strip(),
        gold_stdlib=row["ground_truth"]["gold_stdlib"].strip(),
        gold_extlib_section=(
            _GOLD_EXTLIB_SECTION.format(gold_extlib=gold_extlib.strip())
            if gold_extlib
            else ""
        ),
        cases=_format_cases(row),
        max_chars=f"{MAX_PROMPT_CHARS:,}",
        extlib_instruction=(
            _EXTLIB_INSTRUCTION if gold_extlib else _NO_EXTLIB_INSTRUCTION
        ),
        raises=RAISES,
        feedback=(
            _FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems))
            if problems
            else ""
        ),
    )


# --- checking a rewrite ------------------------------------------------------------------------


def _short(
    text: str,
    limit: int = 200,
) -> str:
    """Cut a long value down for a problem message.

    Returns the text, shortened with "..." if it was longer than `limit`.
    """
    return text if len(text) <= limit else text[:limit] + "..."


def _is_literal(
    text: str,
) -> bool:
    """Whether a string is a python literal (as ast.literal_eval reads them).

    Returns True if it parses as one.
    """
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return True


def _library_leaks(
    text: str,
    row: dict,
) -> list[str]:
    """Library names `text` mentions that the benchmark's own stdin/stdout prompt doesn't.

    The names checked are the benchmark's blocklist plus the task's own libraries (its seed's and
    its library gold's). A name the stdin/stdout prompt already uses is the task's subject (e.g.
    "Markdown" in a Markdown task), not a leak, so it is allowed.

    Returns the leaked names.
    """
    gold_extlib = row["ground_truth"].get("gold_extlib") or ""
    names = set(find_named_libraries(text))
    task_libraries = {
        *row["source"].get("seed_libs", []),
        *analyse_source(gold_extlib)[1],
        *(
            package.split("==")[0]
            for package in row["ground_truth"].get("gold_extlib_packages", [])
        ),
    }
    lowered = text.lower()
    names |= {
        name
        for name in task_libraries
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered)
    }
    stdio_prompt = row["prompt"].lower()
    return sorted(name for name in names if name.lower() not in stdio_prompt)


def check_text(
    entry: dict,
    row: dict,
) -> list[str]:
    """Check a rewrite's text, names and code without running anything.

    Returns the problems found (empty if it passes).
    """
    problems = []
    name = entry["function_name"]
    prompt = entry["prompt"]

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in _GENERIC_NAMES:
        problems.append(f"function name {name!r} is not a descriptive snake_case name")
    if f"{name}(" not in prompt:
        problems.append(f"the text doesn't name the function as {name}(...)")
    if len(prompt) > MAX_PROMPT_CHARS:
        problems.append(
            f"the text is {len(prompt):,} characters, over the {MAX_PROMPT_CHARS:,} limit"
        )
    if not entry["example_call"].startswith(f"{name}("):
        problems.append(f"example_call doesn't call {name}")
    for field in ("example_call", "example_result"):
        if entry[field] not in prompt:
            problems.append(f"{field} doesn't appear in the text exactly as written")
    if not _is_literal(entry["example_result"]):
        problems.append("example_result is not a python literal")
    leaks = _library_leaks(f"{prompt} {name}", row)
    if leaks:
        problems.append(f"the text names libraries: {', '.join(leaks)}")

    if f"def {name}(" not in entry["stdlib_solution"]:
        problems.append(f"stdlib_solution doesn't define {name}")
    if analyse_source(entry["stdlib_solution"])[1]:
        problems.append("stdlib_solution imports a third-party library")
    if row["ground_truth"].get("gold_extlib"):
        if f"def {name}(" not in entry["extlib_solution"]:
            problems.append(f"extlib_solution doesn't define {name}")
        if not analyse_code(entry["extlib_solution"]).external_used:
            problems.append("extlib_solution doesn't use a third-party library")
    if "def convert_input(" not in entry["convert_input"]:
        problems.append("convert_input doesn't define convert_input(stdin_text)")

    n_cases = len(row["test_inputs"])
    transcribed = entry["transcribed_outputs"]
    if len(transcribed) != n_cases:
        problems.append(
            f"transcribed_outputs has {len(transcribed)} entries, for {n_cases} test cases"
        )
    bad = [
        str(number)
        for number, value in enumerate(transcribed, start=1)
        if value != RAISES and not _is_literal(value)
    ]
    if bad:
        problems.append(
            f"transcribed_outputs {', '.join(bad)} are not python literals or {RAISES!r}"
        )
    return problems


# the program that converts a task's tests: it turns each original input into arguments with
# convert_input, runs the stdlib gold on them, and evaluates the example call. it prints one line
# of json: every value as a python literal (or an error)
_CONVERT_PROGRAM = """\
import ast
import contextlib
import io
import json

CONVERTER = {converter!r}
SOLUTION = {solution!r}
FUNCTION_NAME = {function_name!r}
INPUTS = {inputs!r}
EXAMPLE_CALL = {example_call!r}
RAISES = {raises!r}


def as_literal(value):
    # a value's python literal, only if reading it back gives the same value (so not, e.g., a
    # datetime, a numpy value or nan)
    text = repr(value)
    try:
        same = ast.literal_eval(text) == value
    except Exception:
        same = False
    return text if same else None


converter = {{"__name__": "converter"}}
solution = {{"__name__": "solution"}}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(CONVERTER, "converter.py", "exec"), converter)
    exec(compile(SOLUTION, "solution.py", "exec"), solution)
function = solution[FUNCTION_NAME]

cases = []
for stdin_text in INPUTS:
    case = {{}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            arguments = list(converter["convert_input"](stdin_text))
    except Exception as error:
        cases.append({{"error": f"convert_input failed: {{type(error).__name__}}: {{error}}"}})
        continue
    case["arguments"] = as_literal(arguments)
    if case["arguments"] is None:
        cases.append({{"error": "convert_input returned values that aren't python literals"}})
        continue
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = function(*ast.literal_eval(case["arguments"]))
    except Exception as error:
        case["result"] = RAISES
        case["raised"] = f"{{type(error).__name__}}: {{error}}"
    else:
        case["result"] = as_literal(result)
        if case["result"] is None:
            case["error"] = f"returned {{type(result).__name__}}, which isn't a python literal"
    cases.append(case)

try:
    with contextlib.redirect_stdout(io.StringIO()):
        example = repr(eval(EXAMPLE_CALL, solution))
except Exception as error:
    example = None
print(json.dumps({{"cases": cases, "example": example}}))
"""


def convert_tests(
    entry: dict,
    row: dict,
) -> tuple[list[str], list[str], str | None, list[str]]:
    """Turn a task's stdin/stdout tests into function tests, by running the converted stdlib gold.

    Returns (test inputs, expected outputs, the example call's result, problems), with each
    value as a python literal; the problems say which cases couldn't be converted.
    """
    ok, output = execute(
        code=_CONVERT_PROGRAM.format(
            converter=entry["convert_input"],
            solution=entry["stdlib_solution"],
            function_name=entry["function_name"],
            inputs=row["test_inputs"],
            example_call=entry["example_call"],
            raises=RAISES,
        ),
        python_executable=VALIDATION_PYTHON,
        timeout_seconds=120.0,
    )
    if not ok:
        return [], [], None, [f"converting the tests failed: {_short(output, 400)}"]

    report = json.loads(output.strip().splitlines()[-1])
    test_inputs, expected_outputs, problems = [], [], []
    for number, case in enumerate(report["cases"], start=1):
        if "error" in case:
            problems.append(f"test {number}: {case['error']}")
            continue
        test_inputs.append(case["arguments"])
        expected_outputs.append(case["result"])
    return test_inputs, expected_outputs, report["example"], problems


def check_behaviour(
    entry: dict,
    row: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Convert a rewrite's tests and check the conversion kept the original's behaviour.

    The stdlib gold must match the transcribed original outputs (within their rounding), the
    example must hold, and the library gold (test tasks) must pass every converted case.

    Returns (problems, test inputs, expected outputs).
    """
    test_inputs, expected_outputs, example, problems = convert_tests(entry, row)
    if problems or not test_inputs:
        return problems or ["no test cases were converted"], [], []

    for number, (result, transcribed) in enumerate(
        zip(expected_outputs, entry["transcribed_outputs"]),
        start=1,
    ):
        original = _short(row["expected_outputs"][number - 1])
        if (result == RAISES) != (transcribed == RAISES):
            problems.append(
                f"test {number}: stdlib_solution gives {_short(result)}, but the original "
                f"printed {original!r} (transcribed as {_short(transcribed)})"
            )
        elif result != RAISES and not results_match(
            ast.literal_eval(result),
            ast.literal_eval(transcribed),
            rel_tol=ANCHOR_REL_TOL,
            abs_tol=ANCHOR_ABS_TOL,
        ):
            problems.append(
                f"test {number}: stdlib_solution returns {_short(result)}, which doesn't "
                f"match the original output {original!r} (transcribed as "
                f"{_short(transcribed)})"
            )

    if (
        example is None
        or not _is_literal(example)
        or not results_match(
            ast.literal_eval(example),
            ast.literal_eval(entry["example_result"]),
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        )
    ):
        problems.append(
            f"the example call gives {example}, not {entry['example_result']}"
        )

    if row["ground_truth"].get("gold_extlib") and not problems:
        problems += _check_extlib(entry, row, test_inputs, expected_outputs)
    return problems, test_inputs, expected_outputs


def _check_extlib(
    entry: dict,
    row: dict,
    test_inputs: list[str],
    expected_outputs: list[str],
) -> list[str]:
    """Run the converted library gold against the converted tests, in the build venv.

    Its packages are the benchmark's validated pins, plus anything else it imports.

    Returns the problems found (empty if it passes every case).
    """
    pinned = row["ground_truth"].get("gold_extlib_packages", [])
    pinned_names = {package.split("==")[0].lower() for package in pinned}
    extra = sorted(
        package
        for package in required_pip_packages(
            f"```python\n{entry['extlib_solution']}\n```"
        )
        if package.lower() not in pinned_names
    )
    with (
        _INSTALL_LOCK,
        temporary_packages(VALIDATION_PYTHON, pinned + extra) as install,
    ):
        if install.failed:
            return [f"could not install {sorted(install.failed)} for extlib_solution"]
        passed, message = run_function_tests(
            code=entry["extlib_solution"],
            function_name=entry["function_name"],
            test_inputs=test_inputs,
            expected_outputs=expected_outputs,
            python_executable=VALIDATION_PYTHON,
        )
    return [] if passed else [f"extlib_solution fails: {_short(message, 400)}"]


def weak_tests(
    expected_outputs: list[str],
) -> bool:
    """Whether one expected value covers more than half of a task's cases.

    A function that always returns that value (e.g. None, [], or always raising) would then pass
    most cases, so such a task's tests tell answers apart poorly. This inherits from the
    original's tests, so it is recorded, not fixed.

    Returns True for weak tests.
    """
    _, most_common = Counter(expected_outputs).most_common(1)[0]
    return most_common > len(expected_outputs) / 2


# --- generating -------------------------------------------------------------------------------


def generate_function_task(
    row: dict,
    seed: dict,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_attempts: int = 3,
) -> dict[str, Any] | None:
    """Ask the model for one task's function-style rewrite, retrying with feedback on failure.

    Each attempt is checked (check_text, then check_behaviour); a rejected attempt's problems
    are fed back to the next, up to `max_attempts` times. A task the model judges can't be posed
    as a function comes back with fits_function false.

    Returns the cache entry, or None if every attempt failed.
    """
    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        data, resolved_model = complete_json(
            prompt=build_request(row, seed, problems),
            schema=_SCHEMA,
            system=_SYSTEM,
            model=model,
            # thinking counts towards max_tokens, and the reply holds two full solutions
            max_tokens=64000,
            effort=effort,
        )
        if not data:
            print(f"  {row['task_id']} attempt {attempt}: no usable reply")
            continue

        entry: dict[str, Any] = {
            field: data.get(field, "")
            for field in _FIELD_TYPES
            if field != "transcribed_outputs"
        }
        entry["transcribed_outputs"] = list(data.get("transcribed_outputs", []))
        entry.update(
            {
                "task_id": row["task_id"],
                "model": resolved_model,
                "effort": effort,
                "prompt_version": PROMPT_VERSION,
            }
        )
        if not entry["fits_function"]:
            return entry

        problems = check_text(entry, row)
        if not problems:
            problems, test_inputs, expected_outputs = check_behaviour(entry, row)
        if not problems:
            return {
                **entry,
                "test_inputs": test_inputs,
                "expected_outputs": expected_outputs,
                "weak_tests": weak_tests(expected_outputs),
            }
        print(f"  {row['task_id']} attempt {attempt} rejected: {problems}")
    return None


def load_cache() -> dict[str, dict[str, Any]]:
    """Load the cached rewrites, or {} if there are none yet.

    Returns the {task_id: entry} mapping.
    """
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def write_cache(
    cache: dict[str, dict[str, Any]],
) -> None:
    """Write the cached rewrites, ordered by task id so diffs stay readable."""
    CACHE_FILE.write_text(
        json.dumps(dict(sorted(cache.items())), indent=2, ensure_ascii=False) + "\n"
    )


def build_rows(
    rows: list[dict],
    cache: dict[str, dict[str, Any]],
) -> list[dict]:
    """Build the function-style dataset rows for one split, from its cached rewrites.

    Rows keep the benchmark's fields and order; tasks that weren't rewritten (or don't fit a
    function) are left out.

    Returns the dataset rows.
    """
    built = []
    for row in rows:
        entry = cache.get(row["task_id"])
        if not entry or not entry["fits_function"]:
            continue
        ground_truth = {"gold_stdlib": entry["stdlib_solution"]}
        if row["ground_truth"].get("gold_extlib"):
            ground_truth["gold_extlib"] = entry["extlib_solution"]
            ground_truth["gold_extlib_packages"] = row["ground_truth"][
                "gold_extlib_packages"
            ]
        built.append(
            {
                "task_id": row["task_id"],
                "category": row["category"],
                "prompt": entry["prompt"],
                "function_name": entry["function_name"],
                "test_inputs": entry["test_inputs"],
                "expected_outputs": entry["expected_outputs"],
                "ground_truth": ground_truth,
                "source": {**row["source"], "variant": "function"},
                "stdio_prompt": row["prompt"],
                "weak_tests": entry["weak_tests"],
            }
        )
    return built


def show(
    row: dict,
    entry: dict[str, Any],
) -> None:
    """Print a rewrite beside the benchmark's own wording, with two converted test cases."""
    print(f"\n===== {row['task_id']} ({row['category']})")
    if not entry["fits_function"]:
        print(f"doesn't fit a function: {entry['reason']}")
        return
    print(f"--- stdin/stdout ({len(row['prompt']):,} characters)\n{row['prompt']}")
    print(f"--- function ({len(entry['prompt']):,} characters)\n{entry['prompt']}")
    for arguments, expected in list(
        zip(entry["test_inputs"], entry["expected_outputs"])
    )[:2]:
        print(f"--- test: {_short(arguments)} -> {_short(expected)}")
    if entry["weak_tests"]:
        print("--- note: one expected value covers more than half the cases")


def main() -> None:
    """Reword the chosen tasks as function-style tasks, then rebuild the dataset files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["test", "control"],
        action="append",
        help="benchmark split(s) to reword (default: both)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="comma-separated task ids to reword (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only reword the first N tasks"
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        help="skip tasks already cached with the current prompt",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="tasks generated at once"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="generation model id")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, help="generation effort (low to max)"
    )
    args = parser.parse_args()

    if not Path(VALIDATION_PYTHON).exists():
        sys.exit(
            f"no build venv at {VALIDATION_PYTHON} - create it with "
            "benchmark/pipeline/setup_build_venv.sh, or set LRB_BUILD_PYTHON"
        )

    splits = args.split or ["test", "control"]
    rows = {split: load_jsonl(DATA_DIR / f"{split}.jsonl") for split in splits}
    seeds = {
        seed["seed_id"]: seed
        for split in splits
        for seed in load_jsonl(artefact_path(_SEED_FILES[split]))
    }

    cache = load_cache()
    todo = [row for split in splits for row in rows[split]]
    if args.tasks:
        wanted = set(args.tasks.split(","))
        todo = [row for row in todo if row["task_id"] in wanted]
    if args.skip_cached:
        todo = [
            row
            for row in todo
            if cache.get(row["task_id"], {}).get("prompt_version") != PROMPT_VERSION
        ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} task(s) to reword with {args.model} (prompt {PROMPT_VERSION})")

    def generate(
        row: dict,
    ) -> tuple[dict, dict[str, Any] | None]:
        """Generate one task's rewrite, in a worker thread.

        Returns the benchmark row and its cache entry (None if it failed).
        """
        try:
            entry = generate_function_task(
                row=row,
                seed=seeds[row["source"]["seed_id"]],
                model=args.model,
                effort=args.effort,
            )
        except Exception as error:
            # e.g. an api timeout - the task stays uncached, so a --skip-cached rerun retries it
            print(f"  {row['task_id']}: {type(error).__name__}: {error}")
            entry = None
        return row, entry

    # only this thread writes the cache, as each task finishes
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, row) for row in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            row, entry = future.result()
            if entry is None:
                print(f"{done}/{len(todo)} | {row['task_id']} -> failed, not cached")
                continue
            cache[row["task_id"]] = entry
            write_cache(cache)
            show(row, entry)
            print(f"{done}/{len(todo)} | {row['task_id']} -> cached")

    # the dataset files always reflect the whole cache, for every split
    for split in ("test", "control"):
        split_rows = rows.get(split) or load_jsonl(DATA_DIR / f"{split}.jsonl")
        built = build_rows(split_rows, cache)
        save_jsonl(data=built, file_path=str(PILOT_DIR / f"{split}.jsonl"))
        print(
            f"{split}: {len(built)} function-style task(s) in {PILOT_DIR / split}.jsonl"
        )


if __name__ == "__main__":
    main()
