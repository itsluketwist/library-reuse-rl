"""Stage 02: generate one new function task per BigCodeBench seed, and check it by running it.

One model call per seed writes everything a task needs: the request for one named function, a
solution that uses a third-party library, a solution that uses only the standard library, and a
set of test inputs. The task is NEW, inspired by the seed, not a rewording of it - the aim is a
task where a library does substantial work that takes far more code by hand.

Nothing is taken on trust. Each reply is checked, and a failed check is fed back to the model with
its own reply, for up to MAX_ATTEMPTS attempts:
1. the text: it names the function, shows its one example, stays within the length cap and names
   no library; the standard-library solution imports nothing else; the library solution really
   uses a library; every test input is a list literal;
2. the expected values come from running the library solution, twice, in the build venv - they
   must be plain python values, and the same both times;
3. the standard-library solution must pass every test, scored exactly as the benchmark scores a
   model's answer;
4. the example in the text must hold;
5. the tests must tell answers apart: no one result covers more than half of them, and at most
   MAX_RAISES expect an exception;
6. the reuse value (the logical lines of code the library saves) is measured, and sets the
   task's tier. A task that comes out in the low tier gets one more try at giving the library a
   substantial job.

--track reuse (the default) uses the seeds whose reference solution uses a library. --track
control uses the standard-library-only seeds, and asks for a task where no library would help: it
has no library solution, so its expected values come from the standard-library solution.

Records are appended to benchmark/artefacts/02_generated.jsonl (02_generated_control.jsonl for
the control track), one per seed, and a seed's last record is its current state. Generated code
only ever runs in the build venv (benchmark/pipeline/setup_build_venv.sh).

Usage: python benchmark/pipeline/02_generate.py [--track reuse|control] [--seeds ID,ID]
       [--limit N] [--skip-done] [--workers N] [--model M] [--effort E]
"""

import argparse
import ast
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from typing import Any

from build import (
    BUILD_PYTHON,
    SEED_FILES,
    TRACKS,
    artefact_path,
    full_solution,
    latest_records,
    reuse_stats,
    task_hash,
    track_artefact_path,
)
from generation import DEFAULT_EFFORT, DEFAULT_MODEL, complete_json, usage_summary
from libreusebench.environment import execute, temporary_packages
from libreusebench.functions import (
    ABS_TOL,
    RAISES,
    REL_TOL,
    evaluate_function_record,
    results_match,
)
from libreusebench.libraries import analyse_source, find_named_libraries
from lure import analyse_code, pip_package

from src.utils.io import append_jsonl, load_jsonl


# the most attempts one seed gets, counting the extra try a low-tier task is given
MAX_ATTEMPTS = 3

# the longest task text accepted, in characters - it should read as a short request
MAX_PROMPT_CHARS = 1500

# the fewest test cases a task needs, and the most of them that may expect an exception
MIN_TESTS = 10
MAX_RAISES = 3

# the fixed list of domains a task is filed under
DOMAINS = [
    "statistics",
    "linear algebra",
    "machine learning",
    "text and NLP",
    "parsing and markup",
    "dates and times",
    "cryptography and hashing",
    "images",
    "data wrangling",
    "numeric and scientific",
    "encoding and serialisation",
    "other",
]

# names too generic to say what a function does
_GENERIC_NAMES = {"task_func", "solve", "solution", "main", "func", "function", "f"}

# the two or three worked examples shown in every reuse-track request (copied from the v1
# function pilot, see benchmark/v1/)
_EXAMPLES_FILE = artefact_path("02_worked_examples.json")

# installing into the shared build venv isn't safe from two threads at once, and a library
# solution must run while its own packages are the ones installed
_INSTALL_LOCK = threading.Lock()


# --- the request --------------------------------------------------------------------------------

_SYSTEM = (
    "You write programming tasks, with reference solutions and tests, for a benchmark that "
    "measures whether a coding model reuses libraries. Reply with JSON only."
)

# what the benchmark is for, and what makes a task useful to it
_INTRO = {
    "reuse": """\
We are building a benchmark that measures whether a coding model reaches for a third-party
library when it is free to choose. Each task asks for one Python function, and names no library.
Every task must be solvable both with a library and with the standard library alone.

## What we have learnt

An earlier version of this benchmark showed what makes a task useful:
- A good task is one where a mature library does substantial work that takes far more code by
  hand, yet a careful standard-library solution still exists. Models reach for a library on such
  tasks, and not on others.
- We measure this as the logical lines of code the library saves. Models only reached for a
  library where it saved more than about 20 lines. So aim for a task where an idiomatic
  standard-library solution is at least 20 lines longer than the library solution, and at least
  twice its length. Never get there by padding: choose a task where the library really replaces
  that much work.
- Lines saved are not enough on their own. What draws a library is a specialised job that is
  hard to get right by hand: parsing a markup or data format, a statistical test, fitting a
  model, a cryptographic primitive. Tasks that only standardise, rescale, filter, group or
  reshape numbers and tables drew no library use at all, even where a library saved about 25
  lines: models simply write the loops.
- Convenience use is not a library job. Taking a mean or a sum, or building a table just to add a
  column, saves almost no code, so it gives no reason to use the library.
- The standard-library solution may be long and complex. That is fine, and good.
- The two solutions have to agree on every test. A hand-written solution can't reproduce a
  library's internal heuristics or data tables (a stemmer's rules, a tokeniser, a sentiment
  lexicon), so choose a job whose result is fully defined by a standard, a format or a formula.
  Where a result is only defined up to a sign or an order (principal components, eigenvectors,
  cluster labels), fix a convention in the text, or ask for something that doesn't depend on it.
""",
    "control": """\
We are building a benchmark that measures whether a coding model reaches for a third-party
library when it is free to choose. Each task asks for one Python function, and names no library.

This task is for the benchmark's CONTROL split: tasks where no third-party library would help,
so that a good solution uses only the standard library. They show whether a model reaches for a
library even where there is no reason to.
""",
}

_SEED = """
## The seed task (from BigCodeBench)

{description}

Its reference solution, which uses {libraries}:
```python
{solution}
```

Its tests:
```python
{tests}
```
"""

_EXAMPLES_HEADER = """
## Worked examples

These are tasks from an earlier version of the benchmark, with their two solutions.
"""

_GOOD_EXAMPLE = """
### Example {number}: a good task

The library ({library}) does the heavy lifting: its solution is {extlib_lloc} logical lines of
code, against {stdlib_lloc} by hand.

The task text:
{prompt}

The library solution:
```python
{extlib_solution}
```

The standard-library solution:
```python
{stdlib_solution}
```
"""

_TRIVIAL_EXAMPLE = """
### Example {number}: a task to avoid

The library ({library}) does almost nothing here: its solution is {extlib_lloc} logical lines of
code, against {stdlib_lloc} by hand. A task like this gives no reason to use a library.

The task text:
{prompt}

The library solution:
```python
{extlib_solution}
```
"""

_WHAT_TO_WRITE = {
    "reuse": """
## What to write

Write a NEW task inspired by the seed. Do not reword the seed.
- Stay in the seed's domain, with the seed's library (or a library of the same family) doing the
  heavy lifting in the library solution. Change the details, the inputs and the scale as needed,
  so that the library's job is substantial.
- Keep everything outside the library's job simple. The library should do most of the work, so
  that the library solution stays short (about 25 logical lines at most). Don't add rules that
  both solutions then have to implement by hand: they make the task longer without giving any
  more reason to use the library.
- Where the seed draws a plot, ask for the computed data that the plot would show.
- Where the seed generates random data, pass that data in as arguments.
- Where the seed reads or writes files, pass their contents in as strings or bytes, and return
  what would have been written.

Reply with verdict "skip", and say why in reason, if the task:
- needs the network, the operating system or a GUI;
- can't be made deterministic;
- would need a data asset written out by hand (a word list, a pretrained model), not code, to
  solve without the library;
- can't be given a substantial library job.
Otherwise reply with verdict "task". For a skip, still give the domain, and leave every other
field empty.
""",
    "control": """
## What to write

Write a NEW task inspired by the seed. Do not reword the seed.
- Stay in the seed's domain. The task must be one where no third-party library would help: the
  standard library solves it directly, and reaching for a library would gain nothing.
- Where the seed generates random data, pass that data in as arguments.
- Where the seed reads or writes files, pass their contents in as strings or bytes, and return
  what would have been written.

Reply with verdict "skip", and say why in reason, if the task:
- needs the network, the operating system or a GUI;
- can't be made deterministic;
- is one where a third-party library would clearly help.
Otherwise reply with verdict "task". For a skip, still give the domain, and leave every other
field empty.
""",
}

_TEXT_RULES = """
## The task text (the `prompt` field)

- Name the function and its parameters, in order, in the text - e.g. "Write a function
  `count_by_length(letter, words)` that ...". Use a descriptive snake_case name that says what
  the function does: not task_func, solve or main, and not the name of any library's function.
  Give no imports, signature stub, type hints or other code apart from the one example call.
- The arguments and the return value are natural Python values (numbers, strings, bytes, lists,
  tuples, dicts, sets, booleans, None), never a library's type. If reading a text format is part
  of the task (HTML, CSV, JSON, dates), take that text as a string; otherwise pass the data
  already parsed.
- The result must be writable as a Python literal: no NaN or infinity (return None, or raise),
  and no objects such as dates (return a string or a tuple).
- Say WHAT the function returns, not HOW to compute it: name no library, module, import, class
  or library function, and don't narrate an algorithm. A text that spells the computation out
  step by step invites a hand-written solution. Name the standard method or format instead
  (e.g. "a two-sided paired t-test"), which is part of WHAT, and let the details follow from it.
- Pin down the return value's structure (e.g. "a dict mapping each length to its count") and any
  ordering or tie-breaking rules, so that every input has exactly one correct result. Results
  are compared leniently (numbers within a relative tolerance of 0.001, lists and tuples alike,
  whitespace in strings ignored), so don't ask for rounding or string formatting unless it is
  part of the task itself.
- If the function rejects some inputs, say which exception it raises for them (a ValueError,
  unless another is more natural).
- Give exactly one example, as a call and its return value, e.g. "For example,
  `count_by_length('a', ['apple', 'kiwi', 'avocado'])` returns `{{5: 1, 7: 1}}`."
- Write it the way a developer would ask a colleague for the function, in at most {max_chars}
  characters.
"""

_FIELDS = {
    "reuse": """
## The other fields

- function_name: the function's name, as used in the text.
- domain: one of: {domains}.
- category: a short lower-case slug for the kind of task, e.g. "html-table-extraction".
- primary_library: the import name of the library that does the heavy lifting, e.g. "bs4".
- extlib_solution: an idiomatic, concise solution that uses the library. It must return plain
  Python values, so convert the library's own types before returning them. No top-level code
  besides imports and definitions.
- extlib_packages: the pip packages the library solution needs, e.g. ["beautifulsoup4"].
- stdlib_solution: a correct solution that uses only the standard library, and behaves the same
  on every input. It may run to hundreds of lines; write it idiomatically, with no padding. No
  top-level code besides imports and definitions.
- test_inputs: at least {min_tests} test cases, each the function's positional arguments written
  as a Python list literal, e.g. "['a', ['apple', 'kiwi']]". Cover typical inputs, realistically
  sized inputs and edge cases, and (only where the text says an input is rejected) at most
  {max_raises} invalid inputs. Make the results differ: no single result should be right for
  more than half the cases. Give no expected values: they are taken by running the library
  solution, and the standard-library solution must then agree with it on every case, with
  numbers to three significant figures. So choose inputs whose results are numerically stable.
  All the cases together must run in a few seconds.
- example_call and example_result: the example from the text, exactly as written there (the
  call as a Python expression, the result as a Python literal).
""",
    "control": """
## The other fields

- function_name: the function's name, as used in the text.
- domain: one of: {domains}.
- category: a short lower-case slug for the kind of task, e.g. "roman-numeral-conversion".
- stdlib_solution: a correct, idiomatic solution that uses only the standard library. No
  top-level code besides imports and definitions.
- test_inputs: at least {min_tests} test cases, each the function's positional arguments written
  as a Python list literal, e.g. "['a', ['apple', 'kiwi']]". Cover typical inputs, realistically
  sized inputs and edge cases, and (only where the text says an input is rejected) at most
  {max_raises} invalid inputs. Make the results differ: no single result should be right for
  more than half the cases. Give no expected values: they are taken by running the solution.
  All the cases together must run in a few seconds.
- example_call and example_result: the example from the text, exactly as written there (the
  call as a Python expression, the result as a Python literal).
""",
}

# appended when an earlier reply was not accepted, so the model can repair it. the reply itself
# is shown too: each request stands alone, so the model would not otherwise see what it wrote
_FEEDBACK = """
## Your previous reply

```json
{previous}
```

It was not accepted, for these reasons:
{problems}

Write a new reply that fixes them. Keep what was right, and change the task itself if that is
the simplest fix.
"""

# the reply's fields and their json types - every one is required
_FIELD_TYPES: dict[str, dict[str, Any]] = {
    "verdict": {"type": "string", "enum": ["task", "skip"]},
    "reason": {"type": "string"},
    "function_name": {"type": "string"},
    "prompt": {"type": "string"},
    "domain": {"type": "string", "enum": DOMAINS},
    "category": {"type": "string"},
    "primary_library": {"type": "string"},
    "stdlib_solution": {"type": "string"},
    "extlib_solution": {"type": "string"},
    "extlib_packages": {"type": "array", "items": {"type": "string"}},
    "test_inputs": {"type": "array", "items": {"type": "string"}},
    "example_call": {"type": "string"},
    "example_result": {"type": "string"},
}

# the fields only a reuse-track task has: the control track has no library solution
_LIBRARY_FIELDS = ("primary_library", "extlib_solution", "extlib_packages")


def reply_fields(
    track: str,
) -> list[str]:
    """The names of the fields a track's reply has, in order.

    Returns the field names.
    """
    return [
        field
        for field in _FIELD_TYPES
        if track == "reuse" or field not in _LIBRARY_FIELDS
    ]


def reply_schema(
    track: str,
) -> dict[str, Any]:
    """The json schema a track's reply must match.

    Returns the schema.
    """
    fields = reply_fields(track)
    return {
        "type": "object",
        "properties": {field: _FIELD_TYPES[field] for field in fields},
        "required": fields,
        "additionalProperties": False,
    }


@cache
def _examples_section() -> str:
    """Write out the worked examples for the request, each with the size of its solutions.

    Returns the section's text.
    """
    examples = json.loads(_EXAMPLES_FILE.read_text())
    text = _EXAMPLES_HEADER
    for number, example in enumerate(examples, start=1):
        stats = reuse_stats(
            stdlib_solution=example["stdlib_solution"],
            extlib_solution=example["extlib_solution"],
        )
        template = _GOOD_EXAMPLE if example["kind"] == "good" else _TRIVIAL_EXAMPLE
        text += template.format(
            number=number,
            library=example["primary_library"],
            extlib_lloc=stats["extlib_lloc"],
            stdlib_lloc=stats["stdlib_lloc"],
            prompt=example["prompt"],
            extlib_solution=example["extlib_solution"].strip(),
            stdlib_solution=example["stdlib_solution"].strip(),
        )
    return text


def build_request(
    seed: dict,
    track: str = "reuse",
    previous: dict | None = None,
    problems: list[str] | None = None,
) -> str:
    """Build the request asking for one new task inspired by a seed.

    `previous` is an earlier reply for this seed that was not accepted, and `problems` say why:
    both are shown, so the new reply can fix them.

    Returns the request text.
    """
    libraries = ", ".join(seed["ext_libs"]) or "only the standard library"
    request = _INTRO[track] + _SEED.format(
        description=(seed["instruct_prompt"] or seed["complete_prompt"]).strip(),
        libraries=libraries,
        solution=full_solution(
            seed["complete_prompt"],
            seed["canonical_solution"],
        ).strip(),
        tests=seed["test"].strip(),
    )
    if track == "reuse":
        request += _examples_section()
    request += _WHAT_TO_WRITE[track]
    request += _TEXT_RULES.format(max_chars=f"{MAX_PROMPT_CHARS:,}")
    request += _FIELDS[track].format(
        domains=", ".join(DOMAINS),
        min_tests=MIN_TESTS,
        max_raises=MAX_RAISES,
    )
    if previous is not None and problems:
        request += _FEEDBACK.format(
            previous=json.dumps(
                {field: previous.get(field) for field in reply_fields(track)},
                indent=2,
                ensure_ascii=False,
            ),
            problems="\n".join(f"- {problem}" for problem in problems),
        )
    return request


# --- checking a reply ---------------------------------------------------------------------------


def _short(
    text: str,
    limit: int = 200,
) -> str:
    """Cut a long value down for a problem message.

    Returns the text, shortened with "..." if it was longer than `limit`.
    """
    return text if len(text) <= limit else text[:limit] + "..."


def _literal(
    text: str,
) -> tuple[bool, Any]:
    """Read a string as a python literal (as ast.literal_eval reads them).

    Returns (True, the value), or (False, None) if it is not a literal.
    """
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _same_result(
    first: str,
    second: str | None,
) -> bool:
    """Whether two runs gave the same result, each written as a python literal (or RAISES).

    The values are compared, not their text: a set can print its items in a different order on
    every run, without the result having changed.

    Returns True if the results are the same.
    """
    if first == second or second is None or RAISES in (first, second):
        return first == second
    return _literal(first)[1] == _literal(second)[1]


def _named_libraries(
    entry: dict,
) -> list[str]:
    """Library names a task's text mentions: the benchmark's blocklist, and the task's own.

    The task's own names (its library, its packages and what its library solution imports) must
    match as whole words in the same case. So a task about the Markdown or YAML format can still
    name the format, while `markdown` or `yaml` written as the library's name is caught.

    Returns the names found.
    """
    text = f"{entry['prompt']} {entry['function_name']}"
    own = {
        entry.get("primary_library", ""),
        *entry.get("extlib_packages", []),
        *analyse_source(entry.get("extlib_solution", ""))[1],
    }
    found = set(find_named_libraries(text))
    found |= {
        name for name in own if name and re.search(rf"\b{re.escape(name)}\b", text)
    }
    return sorted(found)


def check_text(
    entry: dict,
    track: str = "reuse",
) -> list[str]:
    """Check a reply's text, names, code and test inputs, without running anything.

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
    if not _literal(entry["example_result"])[0]:
        problems.append("example_result is not a python literal")
    leaks = _named_libraries(entry)
    if leaks:
        problems.append(f"the text names libraries: {', '.join(leaks)}")

    if f"def {name}(" not in entry["stdlib_solution"]:
        problems.append(f"stdlib_solution doesn't define {name}")
    if analyse_source(entry["stdlib_solution"])[1]:
        problems.append("stdlib_solution imports a third-party library")
    if track == "reuse":
        if f"def {name}(" not in entry["extlib_solution"]:
            problems.append(f"extlib_solution doesn't define {name}")
        if not analyse_code(entry["extlib_solution"]).external_used:
            problems.append("extlib_solution doesn't use a third-party library")

    test_inputs = entry["test_inputs"]
    if len(test_inputs) < MIN_TESTS:
        problems.append(
            f"test_inputs has {len(test_inputs)} cases, fewer than the {MIN_TESTS} needed"
        )
    if len(set(test_inputs)) < len(test_inputs):
        problems.append("test_inputs repeats a case")
    bad = [
        str(number)
        for number, arguments in enumerate(test_inputs, start=1)
        if not isinstance(_literal(arguments)[1], list)
    ]
    if bad:
        problems.append(f"test_inputs {', '.join(bad)} are not python list literals")
    return problems


# the program that runs a gold solution on every test input, and on the example call. it prints
# one line of json: every result as a python literal, RAISES where the function raised, or an
# error where the result can't be written as a literal
_RUN_PROGRAM = """\
import ast
import contextlib
import io
import json

SOLUTION = {solution!r}
FUNCTION_NAME = {function_name!r}
INPUTS = {inputs!r}
EXAMPLE_CALL = {example_call!r}
RAISES = {raises!r}


def plain(value):
    # turn a library's value into plain python, at every level of nesting: a numpy array into a
    # list, a numpy number into an int or float, a pandas table into a dict
    if type(value) in (bool, int, float, complex, str, bytes, type(None)):
        return value
    if isinstance(value, dict):
        return {{plain(key): plain(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(items)
        return set(items) if isinstance(value, (set, frozenset)) else items
    for method in ("tolist", "item", "to_dict"):
        if hasattr(value, method):
            return plain(getattr(value, method)())
    return value


def as_literal(value):
    # a value's python literal, only if reading it back gives the same value (so not, e.g., a
    # datetime or nan, which can't be written as one)
    try:
        value = plain(value)
        text = repr(value)
        return text if ast.literal_eval(text) == value else None
    except Exception:
        return None


namespace = {{"__name__": "solution"}}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(SOLUTION, "solution.py", "exec"), namespace)
function = namespace[FUNCTION_NAME]

cases = []
for arguments in INPUTS:
    # the arguments are rebuilt from their literal for every case, so a function that changes
    # its arguments can't affect another case
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = function(*ast.literal_eval(arguments))
    except Exception as error:
        cases.append({{"result": RAISES, "raised": f"{{type(error).__name__}}: {{error}}"}})
        continue
    literal = as_literal(result)
    if literal is None:
        cases.append({{"error": f"returned a {{type(result).__name__}} that can't be written as a plain python literal"}})
    else:
        cases.append({{"result": literal}})

try:
    with contextlib.redirect_stdout(io.StringIO()):
        example = as_literal(eval(EXAMPLE_CALL, namespace))
except Exception as error:
    example = None
print(json.dumps({{"cases": cases, "example": example}}))
"""


def run_gold(
    solution: str,
    entry: dict,
) -> tuple[dict | None, str]:
    """Run a gold solution on a reply's test inputs and example call, in the build venv.

    Returns (the report of _RUN_PROGRAM, ""), or (None, the error) if it could not run.
    """
    ok, output = execute(
        code=_RUN_PROGRAM.format(
            solution=solution,
            function_name=entry["function_name"],
            inputs=entry["test_inputs"],
            example_call=entry["example_call"],
            raises=RAISES,
        ),
        python_executable=BUILD_PYTHON,
    )
    if not ok:
        return None, _short(output, 400)
    return json.loads(output.strip().splitlines()[-1]), ""


def _run_reference(
    entry: dict,
    track: str,
) -> tuple[dict | None, dict | None, list[str], str]:
    """Run a reply's reference solution twice: the library one, or the control track's only one.

    A library solution's packages are installed for the run and removed again afterwards.

    Returns (first report, second report, pinned packages, error); the error is empty on success.
    """
    if track != "reuse":
        first, error = run_gold(entry["stdlib_solution"], entry)
        second, _ = run_gold(entry["stdlib_solution"], entry)
        return first, second, [], error

    # the packages the reply names, plus any other its library solution imports
    packages = sorted(
        {
            *entry["extlib_packages"],
            *map(pip_package, analyse_source(entry["extlib_solution"])[1]),
        }
    )
    with _INSTALL_LOCK, temporary_packages(BUILD_PYTHON, packages) as install:
        if install.failed:
            return None, None, [], f"could not install {sorted(install.failed)}"
        first, error = run_gold(entry["extlib_solution"], entry)
        second, _ = run_gold(entry["extlib_solution"], entry)
        pinned = [
            f"{name}=={version}" for name, version in sorted(install.versions.items())
        ]
    return first, second, pinned, error


def check_behaviour(
    entry: dict,
    track: str = "reuse",
) -> tuple[list[str], dict[str, Any]]:
    """Run a reply's solutions: take the expected values, then check every solution agrees.

    The expected values are what the reference solution returns (the library solution, or the
    control track's standard-library one). They must be plain values, the same on a second run,
    and varied enough to tell answers apart. The example must hold, and the standard-library
    solution must pass every case as the benchmark would score it.

    Returns (problems, {expected_outputs, gold_extlib_packages}); the dict is empty on problems.
    """
    reference = "extlib_solution" if track == "reuse" else "stdlib_solution"
    first, second, pinned, error = _run_reference(entry, track)
    if first is None or second is None:
        return [f"{reference} could not be run: {error}"], {}

    problems = []
    for number, (case, again) in enumerate(
        zip(first["cases"], second["cases"]),
        start=1,
    ):
        if "error" in case:
            problems.append(f"test {number}: {reference} {case['error']}")
        elif not _same_result(case["result"], again.get("result")):
            problems.append(
                f"test {number}: {reference} gave a different result on a second run"
            )
    if problems:
        return problems, {}
    expected_outputs = [case["result"] for case in first["cases"]]

    is_literal, example = _literal(first["example"] or "")
    if not is_literal or not results_match(
        example,
        ast.literal_eval(entry["example_result"]),
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        problems.append(
            f"the example call gives {_short(str(first['example']))}, not the "
            f"example_result {_short(entry['example_result'])}"
        )

    # the tests must tell answers apart
    result, count = Counter(expected_outputs).most_common(1)[0]
    if count > len(expected_outputs) / 2:
        problems.append(
            f"{count} of the {len(expected_outputs)} tests expect the same result "
            f"({_short(result, 80)}): make the results differ"
        )
    raised = [
        f"test {number} ({case['raised']})"
        for number, case in enumerate(first["cases"], start=1)
        if case["result"] == RAISES
    ]
    if len(raised) > MAX_RAISES:
        problems.append(
            f"{reference} raises on {len(raised)} tests, more than the {MAX_RAISES} "
            f"allowed: {_short('; '.join(raised), 400)}"
        )

    # the standard-library solution must pass, scored exactly as a model's answer would be
    passed, message = evaluate_function_record(
        code=entry["stdlib_solution"],
        function_name=entry["function_name"],
        test_inputs=entry["test_inputs"],
        expected_outputs=expected_outputs,
        python_executable=BUILD_PYTHON,
    )
    if not passed:
        problems.append(
            f"stdlib_solution disagrees with {reference}: {_short(message, 400)}"
        )
        if raised and track == "reuse":
            # an exception may be a bug in the library solution, not an input it should reject
            problems.append(f"note that {reference} raised on {'; '.join(raised)}")

    if problems:
        return problems, {}
    checked: dict[str, Any] = {"expected_outputs": expected_outputs}
    if track == "reuse":
        checked["gold_extlib_packages"] = pinned
    return problems, checked


# --- generating ---------------------------------------------------------------------------------


def generate_task(
    seed: dict,
    track: str = "reuse",
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    previous: dict | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the model for one new task from a seed, retrying with feedback until it passes.

    Each reply is checked (check_text, then check_behaviour). A reply that fails is fed back with
    its problems, for up to MAX_ATTEMPTS attempts in all. A reuse-track task that passes but saves
    under REUSE_TIER_MEDIUM lines (the low tier) is fed back once, to give the library a
    substantial job; if that try does no better, the low-tier task is kept as it was.

    `previous` and `problems` start the run from an earlier task that needs fixing, e.g. one the
    agent review sent back.

    Returns the seed's record: kept (with its tests and reuse value), skipped, or failed.
    """
    base = {"seed_id": seed["seed_id"], "track": track, "seed_libs": seed["ext_libs"]}
    fields = reply_fields(track)
    problems = list(problems or [])
    entry: dict[str, Any] = {}
    # an accepted low-tier task, returned if the extra try does no better
    low_tier: dict[str, Any] | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        data, resolved_model = complete_json(
            prompt=build_request(seed, track, previous, problems),
            schema=reply_schema(track),
            system=_SYSTEM,
            model=model,
            effort=effort,
        )
        if not data:
            print(f"  {seed['seed_id']} attempt {attempt}: no usable reply")
            continue

        entry = {field: data.get(field) for field in fields}
        entry |= {"model": resolved_model, "effort": effort, "attempts": attempt}
        if entry["verdict"] == "skip":
            if low_tier is not None:
                break
            return base | entry | {"kept": False, "drop_reason": "skipped"}

        problems = check_text(entry, track)
        checked: dict[str, Any] = {}
        if not problems:
            problems, checked = check_behaviour(entry, track)
        if problems:
            print(f"  {seed['seed_id']} attempt {attempt} rejected: {problems}")
            previous = entry
            continue

        accepted = base | entry | checked | {"kept": True, "drop_reason": None}
        if track == "reuse":
            accepted |= reuse_stats(
                stdlib_solution=entry["stdlib_solution"],
                extlib_solution=entry["extlib_solution"],
            )
            if accepted["reuse_tier"] == "low" and low_tier is None:
                low_tier = accepted
                previous = entry
                problems = [
                    f"the library saves only {accepted['reuse_value']} logical lines of "
                    "code over the standard-library solution: make the library's job "
                    "substantial, or skip"
                ]
                continue
        return accepted | {"task_hash": task_hash(accepted)}

    if low_tier is not None:
        return low_tier | {"task_hash": task_hash(low_tier)}
    return (
        base
        | entry
        | {
            "kept": False,
            "drop_reason": "checks failed" if entry else "no usable reply",
            "problems": problems,
        }
    )


def describe(
    record: dict,
) -> str:
    """Sum a seed's record up in a few words, for the progress line.

    Returns the description.
    """
    if not record["kept"]:
        detail = record.get("reason") or "; ".join(record.get("problems", []))
        return f"{record['drop_reason']}: {_short(detail, 160)}"
    if "reuse_tier" not in record:
        return "kept"
    return (
        f"kept, {record['reuse_tier']} tier: the library saves {record['reuse_value']} "
        f"lines ({record['stdlib_lloc']} against {record['extlib_lloc']})"
    )


def fix_request(
    done: dict | None,
    review: dict | None,
) -> tuple[dict | None, list[str]]:
    """Find the fix the agent review asked for on a seed's current task, if any.

    Stage 03 marks a task "fix" with a note. The note counts only for the task it was written
    about, not for an earlier or later version of it.

    Returns (the task to fix, [the reviewer's note]), or (None, []) if there is nothing to fix.
    """
    if (
        done is None
        or review is None
        or not done.get("kept")
        or review.get("verdict") != "fix"
        or review.get("task_hash") != done.get("task_hash")
    ):
        return None, []
    return done, [f"an independent reviewer asked for a fix: {review['note']}"]


def main() -> None:
    """Generate a task for each chosen seed, appending one record per seed as it finishes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="reuse",
        help="reuse: seeds that use a library; control: standard-library-only seeds",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed ids or numbers to generate, e.g. 723,BigCodeBench/340",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only generate the first N seeds"
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip seeds that already have a record, whatever its outcome",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="seeds generated at once"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="generation model id")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, help="generation effort (low to max)"
    )
    args = parser.parse_args()

    if not Path(BUILD_PYTHON).exists():
        sys.exit(
            f"no build venv at {BUILD_PYTHON} - create it with "
            "benchmark/pipeline/setup_build_venv.sh, or set LRB_BUILD_PYTHON"
        )

    output = track_artefact_path("02_generated.jsonl", args.track)
    done = latest_records(output)
    reviews = latest_records(track_artefact_path("03_agent_reviews.jsonl", args.track))

    todo = load_jsonl(artefact_path(SEED_FILES[args.track]))
    if args.seeds:
        wanted = {
            seed if "/" in seed else f"BigCodeBench/{seed}"
            for seed in args.seeds.split(",")
        }
        todo = [seed for seed in todo if seed["seed_id"] in wanted]
    if args.skip_done:
        todo = [seed for seed in todo if seed["seed_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} {args.track} seed(s) to generate with {args.model}")

    def generate(
        seed: dict,
    ) -> dict[str, Any] | None:
        """Generate one seed's task, in a worker thread.

        Returns the seed's record, or None if the run itself failed (e.g. an api timeout).
        """
        seed_id = seed["seed_id"]
        previous, problems = fix_request(done.get(seed_id), reviews.get(seed_id))
        try:
            record = generate_task(
                seed=seed,
                track=args.track,
                model=args.model,
                effort=args.effort,
                previous=previous,
                problems=problems,
            )
        except Exception as error:
            # no record is written, so a --skip-done rerun tries this seed again
            print(f"  {seed_id}: {type(error).__name__}: {error}")
            return None
        if previous is not None:
            # remember this task replaced one the agent review sent back
            record["fixed_from"] = previous["task_hash"]
        return record

    # only this thread writes the artefact, as each seed finishes
    outcomes: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, seed): seed["seed_id"] for seed in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record is None:
                outcomes["run failed, no record"] += 1
                print(f"{count}/{len(todo)} | {futures[future]} -> run failed")
                continue
            append_jsonl(record, output)
            outcomes[
                f"kept ({record.get('reuse_tier', 'control')})"
                if record["kept"]
                else record["drop_reason"]
            ] += 1
            print(f"{count}/{len(todo)} | {record['seed_id']} -> {describe(record)}")

    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}: {count}")
    print(usage_summary())


if __name__ == "__main__":
    main()
