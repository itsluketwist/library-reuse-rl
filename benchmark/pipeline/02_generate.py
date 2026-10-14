"""Stage 02: reword each BigCodeBench seed as a short function task, and check it by running it.

One model call per seed rewords the seed's task as a short, library-agnostic request for one
named function, and writes what the task needs beside it: a solution that uses the seed's
libraries, a solution that uses only the standard library, and a set of test inputs.

The tasks wanted are CONTESTED ones: tasks a developer might reasonably solve with a library, and
might just as reasonably write by hand. Only those can show whether one model reuses libraries
more than another - on an easy task every model writes the code by hand, and on one too hard to
write by hand every model uses the library. The request explains this, lets the model make a
small change to a seed's task where that makes it contested, and tells it to skip the seeds that
no small change would rescue. Many are skipped.

Nothing is taken on trust. Each reply is checked, and a failed check is fed back to the model with
its own reply, for up to MAX_ATTEMPTS attempts:
1. the text: it names the function, shows its one example, stays within the length cap and names
   no library; the standard-library solution imports nothing else; the library solution really
   uses a library; every test input is a list literal;
2. the expected values come from running the library solution, twice, in the build venv - they
   must be plain python values, and the same both times;
3. the standard-library solution must pass every test, scored exactly as the benchmark scores a
   model's answer;
4. the example in the text must show exactly what the function returns;
5. the tests must tell answers apart: no one result covers more than half of them, and at most
   MAX_RAISES expect an exception.

A task's reuse value (the logical lines of code the library saves) is then measured from its two
solutions, and sets its tier. It is only measured: the request never mentions it, so that the
solutions are not written to a target.

--track reuse (the default) uses the seeds whose reference solution uses a library. --track
control uses the standard-library-only seeds, for tasks where no library would help: these have
no library solution, so their expected values come from the standard-library solution.

Records are appended to benchmark/artefacts/02_generated.jsonl (02_generated_control.jsonl for
the control track), one per seed, and a seed's last record is its current state. Each carries the
fingerprint of the request it came from (PROMPT_VERSION), so tasks from an earlier wording of the
request can be told apart, and --skip-done only skips seeds done with the current one. Generated
code only ever runs in the build venv (benchmark/pipeline/setup_build_venv.sh).

Usage: python benchmark/pipeline/02_generate.py [--track reuse|control] [--seeds ID,ID]
       [--limit N] [--skip-done] [--workers N] [--model M] [--effort E]
"""

import argparse
import ast
import hashlib
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from libreusebench.environment import execute, temporary_packages, warm_up_imports
from libreusebench.functions import RAISES, evaluate_function_record, results_match
from libreusebench.libraries import analyse_source, find_named_libraries
from lure import analyse_code, pip_package

from src.utils.io import append_jsonl, load_jsonl


# the most attempts one seed gets
MAX_ATTEMPTS = 3

# the longest task text accepted, in characters - it should read as a short request. three in
# four of the v1 function tasks, whose wording this stage follows, were within it
MAX_PROMPT_CHARS = 900

# the fewest test cases a task needs, and the most of them that may expect an exception
MIN_TESTS = 10
MAX_RAISES = 3

# how closely the example in the text must match what the function returns. it is far tighter
# than the tolerance tests are scored with: a rounded number in an example reads as a request to
# round, and a solver who rounds then fails the tests
EXAMPLE_REL_TOL = 1e-9
EXAMPLE_ABS_TOL = 1e-12

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

# a few task texts shown in every request as examples of the wording wanted (copied from the v1
# function pilot, see benchmark/v1/)
_EXAMPLES_FILE = artefact_path("02_worked_examples.json")

# installing into the shared build venv isn't safe from two threads at once, and a library
# solution must run while its own packages are the ones installed
_INSTALL_LOCK = threading.Lock()


# --- the request --------------------------------------------------------------------------------

_SYSTEM = (
    "You reword coding benchmark problems as short, library-agnostic function-writing tasks, "
    "with reference solutions and tests to match. Reply with JSON only."
)

# what the benchmark measures, why that decides which tasks are useful, and what is asked for
_INTRO = {
    "reuse": """\
We are building a benchmark about code reuse. Language models that write code are known to write
too many custom implementations: asked for a function, they often write their own code for a
problem that has already been solved, and solved well, by an established library. A model that
parses HTML or XML by hand, writes its own cryptographic code, or reimplements a common utility
(date handling, fuzzy string matching, summary statistics of a table) produces more code to read
and maintain, and more places for bugs, where a well-tested library would do the job in a call
or two. The benchmark measures how often a model does this: given a task, and free to solve it
however it likes, does it reuse a library, or write its own implementation?

So the tasks we want are problems that a well-known library already solves, but that a model
could plausibly solve by hand instead. Both answers must be realistic, or there is no choice to
measure:
- a developer might reasonably use a library for it, and
- a developer might just as reasonably write it by hand, with the standard library alone.
We call such a task CONTESTED, and contested tasks are what we want. If the job is only a few
lines with the standard library, writing it by hand is simply the right answer. If it is so
complicated that nobody would attempt it by hand (a statistical distribution, a matrix
decomposition or an iterative model fit, written from scratch), every model uses the library.
The tasks we want sit in between: writing them by hand is a real job, but a feasible one, and a
library makes the code shorter and more reliable.

The wording matters just as much, because it must leave the choice open. So a task is a short,
plain request, the way one developer would ask another for a function. It names no library, and
it does not spell out an algorithm or a list of rules: a specification reads like an exercise,
and pushes a model towards writing everything by hand.

Below is a task from BigCodeBench, whose reference solution uses particular libraries. Reword it
as one of our tasks, if it suits. BigCodeBench was written to make its libraries necessary, so
many of its tasks are not contested as they stand: some are only feasible with the library, some
use it for no real reason, and some only make sense as a script that plots or writes files. A
small change to the task is sometimes enough to make it contested, and you may make one (see
below). Where it is not enough, skip the seed: that is expected, and much better than forcing a
task out of it.
""",
    "control": """\
We are building a benchmark that measures a choice. A coding model is asked for a function, and
is free to write it however it likes: does it reach for a third-party library, or write the code
itself?

This task is for the benchmark's CONTROL split. Its tasks are ones where that choice should not
arise: the standard library does the job directly, and no third-party library would help. They
show whether a model reaches for a library even where there is no reason to. So a control task
must be easy to write by hand, and must not be a job that a well-known library exists to do.

The wording matters just as much. A task is a short, plain request, the way one developer would
ask another for a function. It names no library, and it does not spell out an algorithm or a
list of rules.

Below is a task from BigCodeBench whose reference solution uses only the standard library.
Reword it as a control task, if it suits. Many seeds don't suit, and skipping those is expected.
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
## The wording we want

These task texts have the wording we want: each is a short, plain request that names no
library, for a problem that a library solves but a model could plausibly write by hand.
"""

_EXAMPLE = """
Example {number}:
{prompt}
"""

# the adaptations a self-contained function needs, shared by both tracks
_ADAPTATIONS = """\
- where the seed generates data at random or reads it from a file, pass the data in as arguments;
- where the seed writes a file or draws a plot, return the values it would have written or drawn.
Don't make the task more elaborate than it needs to be: every extra requirement makes the text
less of a plain request.
"""

# the reasons a seed is skipped that both tracks share
_SHARED_SKIPS = """\
- it needs the network, the operating system, the clock or a GUI, which a self-contained
  function that is tested by calling it can't have;
- its result is random, or it only draws a plot, so that nothing worth returning is left once
  the data is passed in;
- it can't be asked for simply. If a clear request would need more than a few sentences, or a
  list of special rules, the text would stop being a plain request.
Otherwise reply with verdict "task". For a skip, still give the domain, and leave every other
field empty.
"""

_WHAT_TO_WRITE = {
    "reuse": """
## What to write

Reword the seed as a short request for ONE Python function. Stay close to the seed: the same
kind of computation, with the seed's libraries doing the same kind of job in the library
solution. We build on BigCodeBench's tasks, and don't invent our own, so that the benchmark rests
on an established set of tasks.

You may make a small change to the seed's task where that brings it into contested territory:
- if the seed is too difficult by hand, narrow it to a part a model could plausibly write
  itself, but keep enough of it that the library still saves real work: narrowed too far, it
  becomes a few lines that anyone would write by hand;
- if the seed gives the library nothing real to do, widen it a little, so that the library does
  a job a developer would actually want it for, while the task stays easy enough to write by
  hand.
Change as little as you can. The aim is a task where both answers are reasonable, and never one
that favours either of them.

A self-contained function needs these changes too:
"""
    + _ADAPTATIONS
    + """
Reply with verdict "skip", and say why in reason, if the seed doesn't suit. Skip it if:
- it is too difficult without a library, and no small change would fix that. If even a narrowed
  task would mean writing something nobody would sensibly write by hand (a statistical
  distribution's quantiles or p-values, a matrix decomposition, an iterative model fit), every
  model uses the library, and there is no choice to measure. Parsing a format such as HTML or
  XML, routine cryptography such as signing or checking a token, and common utilities are not
  too difficult: they are exactly what models tend to write themselves;
- the library has no real job in it, and no small change would give it one. If a developer
  would never reach for the seed's library for anything like this, there is no choice to
  measure either;
- it can't be solved with the standard library without writing out a data asset by hand (a word
  list, a pretrained model);
"""
    + _SHARED_SKIPS,
    "control": """
## What to write

Reword the seed as a short request for ONE Python function. Keep it the seed's task: the same
computation, still solved with the standard library alone. Change only what a self-contained
function needs:
"""
    + _ADAPTATIONS
    + """
Reply with verdict "skip", and say why in reason, if the seed doesn't suit. Skip it if:
- a third-party library would clearly help with it, since a control task is one where none does;
- it is hard to write by hand, since a control task should leave no reason to look for help;
"""
    + _SHARED_SKIPS,
}

_TEXT_RULES = """
## The task text (the `prompt` field)

Write it the way a developer would ask a colleague for the function: a few plain sentences, and
at most {max_chars} characters. Each rule below is there either to keep the choice open, or to
make the task testable.
- Name the function and its parameters, in order, in the text - e.g. "Write a function
  `count_by_length(letter, words)` that ...". The tests call the function by that name. Use a
  descriptive snake_case name that says what the function does: not task_func, solve or main,
  and not the name of any library's function, which would point at that library. Give no
  imports, signature stub, type hints or other code apart from the one example call.
- The arguments and the return value are natural Python values (numbers, strings, bytes, lists,
  tuples, dicts, sets, booleans, None), never a library's type: a task that hands over a
  library's table or array has already made the choice. If reading a text format is part of the
  task (HTML, CSV, JSON, dates), take that text as a string; otherwise pass the data already
  parsed.
- Say WHAT the function returns, not HOW to compute it. Name no library, module, import, class
  or library function, and don't describe an algorithm step by step: steps are instructions for
  writing it by hand. Naming a standard method (e.g. "a two-sided paired t-test") is fine. But
  don't pin an obscure convention or fitting method that a library's usual call doesn't follow
  (e.g. Filliben's quantile estimates, or a fit by conditional least squares): answers that use
  the library the usual way then fail, so the task tests the convention, not the choice.
- Say what the return value looks like (e.g. "a dict mapping each length to its count"), and its
  order where that matters, so that there is one right answer to test for. Don't legislate for
  unusual inputs: leave out any rule the tests don't need, and anything a reasonable developer
  would do without being told. Every extra rule makes the text more of a specification.
- The result must be writable as a Python literal: no NaN or infinity, and no objects such as
  dates (return a string or a tuple). The tests compare it with a value written out in a file.
- Results are compared leniently (numbers within a relative tolerance of 0.001, lists and tuples
  alike, whitespace in strings ignored), so don't ask for rounding or string formatting unless
  the seed's task is about that.
- If the seed rejects some inputs, say which exception the function raises for them (a
  ValueError, unless another is more natural).
- Give exactly one example, as a call and its return value, e.g. "For example,
  `count_by_length('a', ['apple', 'kiwi', 'avocado'])` returns `{{5: 1, 7: 1}}`." Write the
  result exactly as the function returns it, with every number in full: a rounded number in an
  example reads as a request to round.
"""

# how the test inputs are to be chosen, shared by both tracks
_TEST_INPUTS = """\
- test_inputs: at least {min_tests} test cases, each the function's positional arguments written
  as a Python list literal, e.g. "['a', ['apple', 'kiwi']]". Write every value out in full: an
  expression such as 'a' * 100 is not a literal. Cover typical inputs and a few simple edge
  cases. Test only what the text clearly determines: leave out any case whose result depends on
  a rule the text doesn't state. That is what lets the text stay short, and a test for an
  unstated rule would be unfair. Where the text says an input is rejected, include at most
  {max_raises} invalid inputs. Make the results differ: no single result should be right for
  more than half the cases. All the cases together must run in a few seconds.
"""

_FIELDS = {
    "reuse": """
## The other fields

- function_name: the function's name, as used in the text.
- domain: one of: {domains}.
- category: a short lower-case slug for the kind of task, e.g. "html-table-extraction".
- primary_library: the import name of the main library the library solution uses, e.g. "bs4".
- extlib_solution: the seed's reference solution, adapted to the function: the same libraries
  doing the same job, written idiomatically. It must return plain Python values, so convert the
  library's own types before returning them. No top-level code besides imports and definitions.
- extlib_packages: the pip packages the library solution needs, e.g. ["beautifulsoup4"].
- stdlib_solution: a correct solution that uses only the standard library, written
  idiomatically, as a developer without the library would write it. No top-level code besides
  imports and definitions.
"""
    + _TEST_INPUTS
    + """\
  Give no expected values: they are taken by running the library solution, and the
  standard-library solution must then agree with it on every case, with numbers to three
  significant figures. So choose inputs whose results are numerically stable.
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
"""
    + _TEST_INPUTS
    + """\
  Give no expected values: they are taken by running the solution.
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

Write a new reply that fixes them, keeping what was right. If the seed turns out not to suit a
short, simple task, reply with verdict "skip".
"""


def _examples_text() -> str:
    """Write out the example task texts for the request.

    Returns the section's text.
    """
    examples = json.loads(_EXAMPLES_FILE.read_text())
    return _EXAMPLES_HEADER + "".join(
        _EXAMPLE.format(number=number, prompt=example["prompt"])
        for number, example in enumerate(examples, start=1)
    )


# a fingerprint of everything the request is built from, stamped on every record - so tasks
# written from an earlier wording of the request can be told apart from the current ones
PROMPT_VERSION = hashlib.sha256(
    "\n".join(
        [
            _SYSTEM,
            *_INTRO.values(),
            _SEED,
            _EXAMPLES_HEADER,
            _EXAMPLE,
            *_WHAT_TO_WRITE.values(),
            _examples_text(),
            _TEXT_RULES,
            *_FIELDS.values(),
            _FEEDBACK,
            str((MAX_PROMPT_CHARS, MIN_TESTS, MAX_RAISES)),
        ]
    ).encode()
).hexdigest()[:12]

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


def build_request(
    seed: dict,
    track: str = "reuse",
    previous: dict | None = None,
    problems: list[str] | None = None,
) -> str:
    """Build the request asking for one seed to be reworded as a function task.

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
    request += _examples_text()
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
        # a library's first import after installing is slow, and must not count against the run
        warm_up_imports(
            set(analyse_source(entry["extlib_solution"])[1]),
            python_executable=BUILD_PYTHON,
        )
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
    and varied enough to tell answers apart. The example must show exactly what the function
    returns, and the standard-library solution must pass every case as the benchmark would
    score it.

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
        rel_tol=EXAMPLE_REL_TOL,
        abs_tol=EXAMPLE_ABS_TOL,
    ):
        problems.append(
            f"the example call gives {_short(str(first['example']))}, not the "
            f"example_result {_short(entry['example_result'])} - show the result exactly "
            "as returned, with every number in full"
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
    """Ask the model to reword one seed as a task, retrying with feedback until it passes.

    Each reply is checked (check_text, then check_behaviour). A reply that fails is fed back with
    its problems, for up to MAX_ATTEMPTS attempts in all. A reuse-track task that passes has its
    reuse value measured and recorded, whatever it is.

    `previous` and `problems` start the run from an earlier task that needs fixing, e.g. one the
    agent review sent back.

    Returns the seed's record: kept (with its tests and reuse value), skipped, or failed.
    """
    base = {
        "seed_id": seed["seed_id"],
        "track": track,
        "seed_libs": seed["ext_libs"],
        "prompt_version": PROMPT_VERSION,
    }
    fields = reply_fields(track)
    problems = list(problems or [])
    entry: dict[str, Any] = {}

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
            # measured, never asked for: the request says nothing about how long a solution is
            accepted |= reuse_stats(
                stdlib_solution=entry["stdlib_solution"],
                extlib_solution=entry["extlib_solution"],
            )
        return accepted | {"task_hash": task_hash(accepted)}

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
        help="skip seeds that already have a record from the current request, "
        "whatever its outcome",
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
        todo = [
            seed
            for seed in todo
            if done.get(seed["seed_id"], {}).get("prompt_version") != PROMPT_VERSION
        ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(todo)} {args.track} seed(s) to generate with {args.model} "
        f"(request {PROMPT_VERSION})"
    )

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
