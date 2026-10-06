"""LLM seam that diversifies the wording of library-usage tasks (the hybrid layer).

The deterministic build (build.py) already turns each task into one row per mode (repo/simple/
script) via one randomly-picked template. This module adds *claude-generated* wordings on top,
cached so the build stays reproducible and needs no model at build time:

- simple / repo: one extra wording per task, cached to simple_prompts.json / repo_prompts.json.
- script: three *staged* wordings of one concrete stdin/stdout exercise per task, written in
  LibReuseBench's template at increasing levels of detail (see SCRIPT_STAGES), cached to
  script_stage_prompts.json.

The build reads these caches if present; regenerating them (via this file's CLI) is a
deliberate, one-off offline step that needs ANTHROPIC_API_KEY. Until a cache is regenerated,
build.py produces the template-only rows for that mode without it.

Usage: python data/build/library/augment.py [--type simple|script|repo|all] [--n N]
       [--model M] [--limit N] [--pilot N] [--skip-cached]
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml

from data.build import llm
from data.build.contexts import Context, build_contexts
from data.build.paths import LIBRARY_REPOS_DIR, LIBRARY_TAXONOMY_FILE, REPOS_DIR


_HERE = Path(__file__).parent
TASKS_FILE = _HERE / "tasks.yaml"

# the committed caches the build folds in - {seed_task ("<category_id>.<number>", from
# tasks.yaml): [claude-generated prompts]}. "script" is the dataset-v2 script wording, kept only
# so the script pilot (build.py --pilot) can score it beside the new staged wordings - it is
# retired once the staged cache below is fully generated
CACHE_FILES: dict[str, Path] = {
    "simple": _HERE / "simple_prompts.json",
    "script": _HERE / "script_prompts.json",
    "repo": _HERE / "repo_prompts.json",
}

# the staged script wordings - {seed_task: {fits_stdio, reason, exercise, outline, precise,
# benchmark, model, prompt_version}}, see generate_script_stages
SCRIPT_STAGE_CACHE_FILE = _HERE / "script_stage_prompts.json"

# seed for picking the pilot's sample of tasks (see pilot_seed_tasks)
PILOT_SEED = 42

# duplicated from benchmark/libreusebench/libraries.py::_LIBRARY_NAME_TERMS (not imported -
# data/ and benchmark/ stay independent packages). 25 terms as of this writing; keep the two
# lists in sync by hand.
_LIBRARY_NAME_TERMS = {
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "scikit-learn",
    "statsmodels",
    "dateutil",
    "pytz",
    "beautifulsoup",
    "bs4",
    "lxml",
    "requests",
    "flask",
    "django",
    "nltk",
    "regex",
    "pyyaml",
    "pillow",
    "opencv",
    "tensorflow",
    "pytorch",
    "torch",
    "keras",
}


def _has_library_leak(
    text: str,
    extra_banned_terms: list[str],
) -> bool:
    """True if `text` names a library from the shared blocklist or `extra_banned_terms`.

    `extra_banned_terms` is the task's category libraries: naming one defeats the organic-choice
    design, so it is rejected here rather than trusted.

    Returns whether a banned term appears (case-insensitive substring match).
    """
    lowered = text.lower()
    banned = _LIBRARY_NAME_TERMS | {term.lower() for term in extra_banned_terms}
    return any(term in lowered for term in banned)


def load_prompt_cache(
    prompt_type: str,
) -> dict[str, list[str]]:
    """Load the committed cache for `prompt_type` ("simple"/"script"/"repo"), or {} if absent.

    Returns the {seed_task: [claude-generated prompts]} mapping.
    """
    cache_file = CACHE_FILES[prompt_type]
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return {}


def write_prompt_cache(
    prompt_type: str,
    cache: dict[str, list[str]],
) -> None:
    """Write `cache` to CACHE_FILES[prompt_type] (the offline regeneration step)."""
    CACHE_FILES[prompt_type].write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n"
    )


# --- simple: a short, natural task request, optionally hinting at a signature ---------------

_SIMPLE_SYSTEM = (
    "You write short, natural task requests a developer might type to a coding assistant, for "
    "a dataset that measures whether a model reaches for a library when solving a coding task. "
    "Reply with JSON only."
)

_SIMPLE_PROMPT = """\
A developer wants a Python function that does the following:
{task}

Write {n} different short, natural requests a developer might type to a coding assistant asking
for this - the way a person actually chats to an assistant, not a code stub or docstring. For
each one:
- write one or two plain sentences, e.g. "Can you write a function that ...?" or "I need
  something that takes X and gives back Y."
- you may optionally mention a plausible function name and/or its arguments in plain words (e.g.
  "a function called normalize_date that takes a date string") - vary how much of this you
  include across the {n} versions, including at least one with no signature mentioned at all
- if you do mention argument or return types, keep them to plain built-in Python types (str,
  int, float, bool, list, dict) - never a type that hints at a specific library (no DataFrame,
  ndarray, datetime object, Path, etc.), since that would bias which library the solver reaches
  for
- do NOT name any specific library, package, or tool anywhere in the prompt
- do NOT write actual code, a docstring block, or a doctest example - this should read as
  something a person typed in a chat box, not a code artifact

Reply with JSON only: {{"prompts": ["<request 1>", "<request 2>", ...]}}
"""

# fallback used once a task has failed _SIMPLE_PROMPT twice in a row - for a task whose category
# library name is also the natural English word for the action (e.g. "slugify" a title), the
# model tends to invent a function name that leaks it every time, so asking for a signature at
# all is the problem, not a fixable wording issue
_SIMPLE_PROMPT_NO_SIG = """\
A developer wants a Python function that does the following:
{task}

Write {n} different short, natural requests a developer might type to a coding assistant asking
for this - the way a person actually chats to an assistant, not a code stub or docstring. For
each one:
- write one or two plain sentences, e.g. "Can you write something that ...?" or "I need
  something that takes X and gives back Y."
- do NOT mention a function name, argument names, or any kind of signature at all - describe
  only what the code should do, in plain words
- do NOT name any specific library, package, or tool anywhere in the prompt
- do NOT write actual code, a docstring block, or a doctest example - this should read as
  something a person typed in a chat box, not a code artifact

Reply with JSON only: {{"prompts": ["<request 1>", "<request 2>", ...]}}
"""

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
    "additionalProperties": False,
}


def build_simple_prompt(
    task: str,
    n: int,
    no_sig: bool = False,
) -> str:
    """Build the LLM prompt asking for `n` short, natural chat-style requests for `task`.

    `no_sig` selects the fallback wording that doesn't ask for a function name/signature at all
    (see `generate_simple_prompts`).

    Returns the prompt string.
    """
    template = _SIMPLE_PROMPT_NO_SIG if no_sig else _SIMPLE_PROMPT
    return template.format(task=task, n=n)


def generate_simple_prompts(
    task: str,
    banned_terms: list[str],
    n: int = 1,
    model: str = llm.DEFAULT_GEN_MODEL,
    max_attempts: int = 2,
    fallback_attempts: int = 3,
) -> list[str]:
    """Ask the LLM for `n` short, natural chat-style requests for `task`.

    Rejects any candidate that leaks a library name (shared blocklist or `banned_terms`). Tries
    up to `max_attempts` times with the normal prompt (which may invent a function signature);
    if every attempt leaks, the category's library name is likely also a natural English word
    for the task (e.g. "slugify"), so up to `fallback_attempts` further tries drop the signature
    ask entirely rather than keep re-rolling the same collision.

    Returns the accepted prompts, deduplicated, in the model's order (empty if every attempt,
    including the fallback, kept leaking).
    """
    for no_sig, attempts in ((False, max_attempts), (True, fallback_attempts)):
        for _ in range(attempts):
            data, _ = llm.complete_json(
                prompt=build_simple_prompt(task, n, no_sig=no_sig),
                schema=_SIMPLE_SCHEMA,
                system=_SIMPLE_SYSTEM,
                model=model,
            )
            candidates = [
                str(p).strip() for p in data.get("prompts", []) if str(p).strip()
            ]
            accepted = [p for p in candidates if not _has_library_leak(p, banned_terms)]
            if accepted:
                return list(dict.fromkeys(accepted))
    return []


# --- script: one stdin/stdout exercise, in three staged LibReuseBench-template statements ------

# the staged wordings, from the loosest to the most benchmark-like. all three describe the same
# concrete exercise, and differ only in how much of it they spell out (see _SCRIPT_STAGE_PROMPT)
SCRIPT_STAGES = ("outline", "precise", "benchmark")

# character-length bounds per stage, checked by check_script_stage. LibReuseBench's test prompts
# run from 880 to 2,060 characters (10-90%, median 1,353, 95% 2,425), so the benchmark stage may
# reach its 95th percentile and the looser stages sit below it. the generation prompt asks for
# somewhat less (_SCRIPT_STAGE_TARGETS), since the model tends to overshoot
SCRIPT_STAGE_LENGTHS: dict[str, tuple[int, int]] = {
    "outline": (350, 1300),
    "precise": (600, 1900),
    "benchmark": (850, 2500),
}

# the most characters the generation prompt asks for, per stage
_SCRIPT_STAGE_TARGETS = {
    "outline": 1000,
    "precise": 1500,
    "benchmark": 2000,
}

# library names that are also the standard name of a data format or concept, or a plain english
# word - a benchmark-style statement legitimately says "a YAML document", "exponential backoff" or
# "the Levenshtein distance", so _stage_library_leaks does not count these as leaks
_DOMAIN_WORDS = {
    "arrow",
    "backoff",
    "barcode",
    "click",
    "colour",
    "levenshtein",
    "limits",
    "markdown",
    "regex",
    "requests",
    "rich",
    "schedule",
    "sh",
    "toml",
    "yaml",
}

# library-specific type names a statement must not lean on
_LIBRARY_TYPE_TERMS = {"dataframe", "ndarray"}

# LibReuseBench's statements never talk about libraries (one mention in 268 prompts), so any of
# these words counts as a leak too - it catches "use a library for this" phrasings
_LIBRARY_TALK = re.compile(
    r"\b(?:librar(?:y|ies)|packages?|modules?|imports?|pip|third[- ]party|built-?in)\b",
    flags=re.IGNORECASE,
)

# a character written as a literal json escape sequence, e.g. "±" instead of "±"
_ESCAPED_CHARACTER = re.compile(r"\\u[0-9a-fA-F]{4}")

# python code inside a statement (the benchmark's statements are prose only)
_CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import )|```python",
    flags=re.MULTILINE,
)


def _stage_library_leaks(
    text: str,
    banned_terms: list[str],
) -> list[str]:
    """Find library names, and talk of libraries, in a staged statement.

    Unlike `_has_library_leak` (a substring match, fine for one-line chat requests), this matches
    whole words only and skips `_DOMAIN_WORDS`: a benchmark-style statement names data formats and
    concepts that share a name with a library, e.g. "a YAML document".

    Returns the leaked terms found (empty if none).
    """
    terms = (
        _LIBRARY_NAME_TERMS
        | _LIBRARY_TYPE_TERMS
        | {term.lower() for term in banned_terms}
    ) - _DOMAIN_WORDS
    leaks = [
        term
        for term in sorted(terms)
        # a whole word, not part of a longer name such as "pandas_ta"
        if re.search(
            rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, flags=re.IGNORECASE
        )
    ]
    leaks += sorted({match.lower() for match in _LIBRARY_TALK.findall(text)})
    return leaks


def _line_start(
    pattern: str,
    text: str,
) -> int | None:
    """Find the first line of `text` that starts with `pattern`.

    Returns the match's position, or None if no line matches.
    """
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.start() if match else None


def check_script_stage(
    text: str,
    stage: str,
    banned_terms: list[str],
) -> list[str]:
    """Check one staged statement against LibReuseBench's template, deterministically.

    The template: an opening sentence ("Write a program..." or "You are given...") that mentions
    standard input, then an "Input format:" section, an "Output format:" section (either may be
    prefixed "Standard"), and exactly one "Example:" with an "Input:" block before an "Output:"
    block, in that order. The statement must be prose (no python code), name no library, and have
    a length within the stage's bounds (SCRIPT_STAGE_LENGTHS).

    Returns the problems found (empty if the statement passes).
    """
    problems = []
    if not text.startswith(("Write a program", "You are given")):
        problems.append('does not open with "Write a program" or "You are given"')
    if "standard input" not in text.lower():
        problems.append('never mentions "standard input"')

    input_at = _line_start(r"^(?:Standard )?[Ii]nput format:", text)
    output_at = _line_start(r"^(?:Standard )?[Oo]utput format:", text)
    examples = [m.start() for m in re.finditer(r"^Example:", text, flags=re.MULTILINE)]
    if input_at is None:
        problems.append('no "Input format:" section')
    if output_at is None:
        problems.append('no "Output format:" section')
    if len(examples) != 1:
        problems.append(f'{len(examples)} "Example:" sections, not exactly one')
    else:
        # the example must show an input, then the output it produces
        example = text[examples[0] :]
        example_input = _line_start(r"^Input:", example)
        example_output = _line_start(r"^Output:", example)
        if (
            example_input is None
            or example_output is None
            or example_input > example_output
        ):
            problems.append(
                'the example has no "Input:" block before an "Output:" block'
            )
        if (
            input_at is not None
            and output_at is not None
            and not input_at < output_at < examples[0]
        ):
            problems.append(
                "sections are not ordered input format, output format, example"
            )

    if _CODE_LINE.search(text):
        problems.append("contains python code")
    # the model occasionally writes a character as its json escape ("±" for "±")
    if _ESCAPED_CHARACTER.search(text):
        problems.append(
            "contains an escaped character such as \\u00b1 - write the character itself"
        )
    low, high = SCRIPT_STAGE_LENGTHS[stage]
    if not low <= len(text) <= high:
        problems.append(f"{len(text)} characters, outside {low}-{high}")
    leaks = _stage_library_leaks(text, banned_terms)
    if leaks:
        problems.append(f"mentions libraries: {leaks}")
    return problems


_SCRIPT_STAGE_SYSTEM = (
    "You write stdin/stdout programming-exercise statements for a research dataset about when a "
    "model reuses an existing library instead of hand-rolling the same logic. The statements must "
    "read exactly like those of an existing benchmark, and must never name, suggest or mention "
    "any library. Reply with JSON only."
)

# two real LibReuseBench statements, embedded to calibrate the "benchmark" stage's tone, precision
# and length. both come from its dev split (benchmark/libreusebench/data/dev.jsonl), never test,
# so no evaluated prompt leaks into training data. LRB/D007 keeps a raw document as its input and
# describes what to print without narrating how; LRB/D005 shows the section layout and the
# rounding/validation precision. neither is related to the tasks we generate for
_SCRIPT_STAGE_EXAMPLES = """\
<example>
Write a program that reads an XML document (as a single block of text) from standard input, extracts tabular data from it, and prints the result as CSV to standard output.

The expected XML format is a document that contains one or more <item> elements located anywhere in the tree. Each <item> element has one or more direct child elements; the tag name of each child becomes a column name, and the child's text content becomes the value for that column in that row. The children of the first <item> element (in the order they appear) determine the set and order of the output columns; every other <item> element is expected to have exactly the same child tags in the same order.

Behavior:
- If the input text is not well-formed XML, or if the document (once parsed) contains no <item> elements anywhere, print exactly the single line:
Error: XML structure does not match expected format.
and nothing else.
- Otherwise, print a CSV table to standard output:
 * The first line is the column names (the tag names of the first item's children), separated by commas, in the order they first appear.
 * Each subsequent line corresponds to one <item> element, in the same order the items appear in the document, listing the text values of its children (in the same column order) separated by commas.
 * Do not print an index column, and do not add quoting or extra spaces around the values.

Example:
Input:
<root><item><name>John</name><age>25</age></item><item><name>Jane</name><age>30</age></item></root>

Output:
name,age
John,25
Jane,30
</example>

<example>
Write a program that reads stock-price records from standard input and reports summary statistics for one chosen field.

Standard input format:
- Line 1: the name of the column to analyze. It must be exactly one of: Open, High, Low, Close, Volume.
- Line 2: a non-negative integer N, the number of data rows that follow.
- The next N lines each describe one day of trading data as six whitespace-separated tokens, in this fixed order: a date token (any string with no whitespace), then the Open, High, Low, Close, and Volume values. Each of these five values is either a number (integer or decimal) or the literal token "nan" meaning the value is missing for that day.

Validation rules:
- If the column name given on line 1 is not one of the five allowed names, print exactly:
Invalid column name
and stop.
- If, after that check passes, any data row does not have exactly six tokens, or one of its five numeric tokens is neither a valid number nor the literal "nan", print exactly:
Invalid data
and stop.

Otherwise, using only the values found in the requested column across all N rows (ignoring the date and the other four columns), compute:
- sum: the total of all non-missing values in that column (0 if N is 0 or if every value is missing),
- mean: the average of all non-missing values (undefined if there are no non-missing values),
- min: the smallest non-missing value (undefined if there are no non-missing values),
- max: the largest non-missing value (undefined if there are no non-missing values).

"nan" tokens must be excluded when computing mean, min, and max, but they do not affect the sum (missing values simply contribute nothing to the sum).

Print exactly four lines, in this order: sum, mean, min, max. Each value must be printed as a fixed-point number with exactly 6 digits after the decimal point; any statistic that is undefined (because there are no non-missing values, including the case N = 0) must be printed as the literal text "nan" (without decimals).

Example:
Input:
Open
3
2022-01-01 100 105 95 102 10000
2022-01-02 102 108 100 105 15000
2022-01-03 105 110 103 108 20000

Output:
307.000000
102.333333
100.000000
105.000000
</example>"""

_SCRIPT_STAGE_PROMPT = """\
Here is a coding task from the "{category}" category:
{task}

The usual way to hand-roll it is: {diy_temptation}. That hand-rolled part is the work a mature
third-party library (not the Python standard library) would save, so the exercise you write must
keep it - the solver has to deal with it properly, rather than having it removed or pre-digested
by the input format.

Step 1 - decide whether this task can be posed honestly as a text-in, text-out program: all input
read from standard input as text, all output printed to standard output as text, and the output
a deterministic function of the input (no files, network, clock, randomness, environment
variables, command-line arguments, running other programs or commands, binary data or interactive
terminal behaviour). If posing it
that way would turn it into a different, trivial task (for example, replacing an audio file with
a list of integers), set "fits_stdio" to false, explain why in "reason", and leave the other
fields empty.

Step 2 - otherwise, set "fits_stdio" to true, invent ONE concrete exercise for this task,
summarise it in one sentence in "exercise", and write it as three statements: "outline",
"precise" and "benchmark".

Rules for the exercise:
- Keep the input in its natural, real-world text form - e.g. timestamps with UTC offsets or IANA
  time zone names, HTML, XML, JSON or YAML documents, URLs, phone numbers, cron expressions -
  never pre-parsed into plain numbers or tokens that remove the hard part.
- Lay the input out the way the task naturally implies. Like most of the benchmark's exercises,
  prefer simple line-based input - e.g. a count followed by one record per line, or
  whitespace-separated fields. Use CSV only when the task itself is about CSV or spreadsheet data.
- Of the ways to pose this task, choose the one where a correct hand-rolled solution takes the
  most work, while a careful solver could still write one with the Python standard library alone.
  Never require output that only one particular library's quirks would produce, and fix every
  floating-point output to a stated number of decimal places.
- Keep the required output conventional - standard formats and simple rounding - so that a
  solution built on a well-known library's usual behaviour needs little extra formatting. Do not
  invent bespoke typing, conversion or formatting rules that force custom code whichever approach
  the solver takes.
- Keep the example small: a few short lines of input, enough to show the format.
- Refer to standard formats and concepts by their usual names (e.g. "ISO 8601", "IANA time zone",
  "Welch's t-test", "Levenshtein distance") instead of redefining them.
- Describe only WHAT to read and print, never HOW: do not describe an algorithm, a formula or the
  steps of the core computation, and do not hint at an implementation approach.
- Never name or allude to any library, package, module, import, class or function - not even to
  allow or forbid one - and avoid library-specific terms such as "DataFrame" or "ndarray".
- Plain prose and "-" bullet lists only, no code. Do not mention this dataset, benchmarks, or why
  the exercise exists.

Every statement must follow the benchmark's template, in this order:
1. An opening sentence that starts "Write a program that reads" (or "You are given") and says what
   is read from standard input and what is printed to standard output.
2. A section headed "Input format:" with "-" bullets giving exactly what appears on standard
   input.
3. A section headed "Output format:" with "-" bullets giving exactly what to print.
4. Optionally (mainly in "benchmark"), further sections such as "Validation rules:" or
   "Processing rules:" covering input checks and edge-case behaviour - never the core
   computation.
5. Exactly one section headed "Example:", containing an "Input:" block and then an "Output:"
   block, where the output is exactly what a correct program prints for that input.

The three statements describe the SAME exercise, with the same input and output formats and the
same example, and differ only in how much they spell out:
- "outline": the job in the task's own domain terms, the exact input layout, and an output format
  just tight enough to be unambiguous - nothing more. At most {outline_max} characters.
- "precise": everything in "outline", plus the rules needed to match the output exactly:
  ordering and tie-breaking, rounding, and how to handle empty, missing or malformed input (with
  the exact messages to print). At most {precise_max} characters.
- "benchmark": the same exercise in the full register of the benchmark statements below, with
  every term precisely defined and the edge cases and validation behaviour covered. At most
  {benchmark_max} characters.
These limits count every character, including the example, and are strict: a statement over its
limit is rejected. Say each rule once, plainly, rather than restating it.

Two real statements from the benchmark, to calibrate the tone, precision and length of the
"benchmark" statement (they are unrelated to your task):

{examples}
{feedback}"""

# appended to the prompt when a previous attempt was rejected, so the model can fix what was
# wrong instead of repeating it (e.g. writing too much)
_SCRIPT_STAGE_FEEDBACK = """
Your previous attempt was rejected for these reasons - write a new attempt that fixes them:
{problems}
"""

_SCRIPT_STAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "fits_stdio": {"type": "boolean"},
        "reason": {"type": "string"},
        "exercise": {"type": "string"},
        "outline": {"type": "string"},
        "precise": {"type": "string"},
        "benchmark": {"type": "string"},
    },
    "required": ["fits_stdio", "reason", "exercise", "outline", "precise", "benchmark"],
    "additionalProperties": False,
}

# a fingerprint of everything the staged-wording request is built from, stamped on each cache
# entry - so --skip-cached only skips tasks generated with the current prompt, and an entry from
# an earlier prompt is regenerated rather than silently kept
SCRIPT_STAGE_PROMPT_VERSION = hashlib.sha256(
    "\n".join(
        [
            _SCRIPT_STAGE_SYSTEM,
            _SCRIPT_STAGE_PROMPT,
            _SCRIPT_STAGE_FEEDBACK,
            _SCRIPT_STAGE_EXAMPLES,
            json.dumps(_SCRIPT_STAGE_TARGETS, sort_keys=True),
        ]
    ).encode()
).hexdigest()[:12]


def build_script_stage_prompt(
    task: str,
    category: str,
    diy_temptation: str,
    problems: list[str] | None = None,
) -> str:
    """Build the LLM prompt asking for one exercise for `task`, as three staged statements.

    `category` is the task's category name and `diy_temptation` how the task is usually
    hand-rolled (both from taxonomy.yaml) - the library names are deliberately left out, so the
    model can't shape the exercise around one library's output. `problems` are a previous
    attempt's check failures, fed back so the next attempt can fix them.

    Returns the prompt string.
    """
    feedback = (
        _SCRIPT_STAGE_FEEDBACK.format(
            problems="\n".join(f"- {problem}" for problem in problems)
        )
        if problems
        else ""
    )
    return _SCRIPT_STAGE_PROMPT.format(
        task=task,
        category=category,
        diy_temptation=diy_temptation.rstrip("."),
        examples=_SCRIPT_STAGE_EXAMPLES,
        feedback=feedback,
        **{
            f"{stage}_max": f"{limit:,}"
            for stage, limit in _SCRIPT_STAGE_TARGETS.items()
        },
    )


def generate_script_stages(
    task: str,
    category: str,
    diy_temptation: str,
    banned_terms: list[str],
    model: str = llm.DEFAULT_GEN_MODEL,
    max_attempts: int = 3,
) -> dict[str, Any] | None:
    """Ask the LLM for one concrete stdin/stdout exercise for `task`, in three staged statements.

    Every statement is checked with check_script_stage, against the shared blocklist and
    `banned_terms` (the category's libraries). If any statement fails, the whole attempt is
    retried with the failures fed back - the three describe one exercise, so they are kept or
    dropped together - up to `max_attempts` times. A task the model judges can't be posed
    honestly as text in, text out comes back with fits_stdio false and empty statements, so the
    build gives it no staged rows.

    Returns the cache entry ({fits_stdio, reason, exercise, outline, precise, benchmark, model,
    prompt_version}), or None if every attempt failed.
    """
    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        data, resolved_model = llm.complete_json(
            prompt=build_script_stage_prompt(task, category, diy_temptation, problems),
            schema=_SCRIPT_STAGE_SCHEMA,
            system=_SCRIPT_STAGE_SYSTEM,
            model=model,
            # thinking counts towards max_tokens, and the three statements together are long -
            # at 16,000 a hard task sometimes spent it all on thinking, leaving no reply
            max_tokens=32000,
        )
        if not data:
            print(f"  attempt {attempt}: no usable reply")
            continue

        entry: dict[str, Any] = {
            "fits_stdio": bool(data.get("fits_stdio")),
            "reason": str(data.get("reason", "")).strip(),
            "exercise": str(data.get("exercise", "")).strip(),
            **{stage: str(data.get(stage, "")).strip() for stage in SCRIPT_STAGES},
            "model": resolved_model,
            "prompt_version": SCRIPT_STAGE_PROMPT_VERSION,
        }
        if not entry["fits_stdio"]:
            # no statements for a task that doesn't fit, whatever the model wrote in them
            return {**entry, **{stage: "" for stage in SCRIPT_STAGES}}

        problems = [
            f"{stage}: {problem}"
            for stage in SCRIPT_STAGES
            for problem in check_script_stage(entry[stage], stage, banned_terms)
        ]
        if not problems:
            return entry
        print(f"  attempt {attempt} rejected: {problems}")
    return None


def load_script_stage_cache() -> dict[str, dict[str, Any]]:
    """Load the committed staged script cache, or {} if it doesn't exist yet.

    Returns the {seed_task: entry} mapping (see generate_script_stages).
    """
    if SCRIPT_STAGE_CACHE_FILE.exists():
        return json.loads(SCRIPT_STAGE_CACHE_FILE.read_text())
    return {}


def write_script_stage_cache(
    cache: dict[str, dict[str, Any]],
) -> None:
    """Write the staged script cache, ordered by seed task so diffs stay readable."""
    SCRIPT_STAGE_CACHE_FILE.write_text(
        json.dumps(dict(sorted(cache.items())), indent=2, ensure_ascii=False) + "\n"
    )


def pilot_seed_tasks(
    taxonomy: dict,
    tasks: dict[str, dict[int, str]],
    n: int,
) -> list[str]:
    """Pick a reproducible sample of `n` seed tasks for the script pilot.

    One task from each of `n` categories, half benchmark-aligned and half not: aligned script
    prompts give the least library signal today, so the pilot must cover them as well as the rest.

    Returns the sampled seed tasks ("<category_id>.<number>"), sorted.
    """
    rng = random.Random(PILOT_SEED)
    categories = [entry for entry in taxonomy.get("code", []) if tasks.get(entry["id"])]
    aligned = [entry["id"] for entry in categories if entry.get("benchmark_aligned")]
    unaligned = [
        entry["id"] for entry in categories if not entry.get("benchmark_aligned")
    ]
    rng.shuffle(aligned)
    rng.shuffle(unaligned)

    n_aligned = min(n // 2, len(aligned))
    chosen = aligned[:n_aligned] + unaligned[: n - n_aligned]
    return sorted(f"{cid}.{rng.choice(sorted(tasks[cid]))}" for cid in chosen)


def regenerate_script_stage_cache(
    taxonomy: dict,
    tasks: dict[str, dict[int, str]],
    model: str,
    only: set[str] | None = None,
    limit: int | None = None,
    skip_cached: bool = False,
) -> None:
    """Generate the staged script wordings, writing the cache after every task.

    Covers every task, or just the `only` subset (e.g. the pilot's), optionally skipping tasks
    already cached with the current prompt (SCRIPT_STAGE_PROMPT_VERSION) and stopping after
    `limit` tasks. Writing after each task means an interrupted run loses nothing - rerun with
    `skip_cached` to pick up where it stopped.
    """
    cache = load_script_stage_cache()
    todo = [
        (entry, f"{entry['id']}.{number}", task)
        for entry in taxonomy.get("code", [])
        for number, task in tasks.get(entry["id"], {}).items()
    ]
    if only is not None:
        todo = [item for item in todo if item[1] in only]
    if skip_cached:
        # only an entry made with the current prompt counts as done
        todo = [
            item
            for item in todo
            if cache.get(item[1], {}).get("prompt_version")
            != SCRIPT_STAGE_PROMPT_VERSION
        ]
    if limit is not None:
        todo = todo[:limit]

    for i, (entry, seed_task, task) in enumerate(todo, start=1):
        result = generate_script_stages(
            task=task,
            category=entry["name"],
            diy_temptation=entry["diy_temptation"],
            banned_terms=entry["seed_libraries"],
            model=model,
        )
        if result is None:
            print(
                f"script: {i}/{len(todo)} | {seed_task} -> every attempt failed, not cached"
            )
            continue
        cache[seed_task] = result
        write_script_stage_cache(cache)
        status = "staged" if result["fits_stdio"] else f"not stdio: {result['reason']}"
        print(f"script: {i}/{len(todo)} | {seed_task} -> {status}")
    print(f"script: {len(cache)} cached task(s) in {SCRIPT_STAGE_CACHE_FILE}")


# --- repo: a natural request framed inside the task's chosen repo ----------------------------

_REPO_SYSTEM = (
    "You write natural feature requests a developer might make inside a specific existing "
    "codebase, for a dataset that measures whether a model reaches for a library when it is "
    "free to. Reply with JSON only."
)

_REPO_PROMPT = """\
Here is the README and file layout of a small existing Python project:
{context}

A developer working on this project wants to add this feature:
{task}

Write {n} short, natural ways this developer might ask a coding assistant to add this feature
to THIS project - referencing "this project" / "this codebase" / its existing modules where it
reads naturally, the way a real request inside this repo would.

Also choose ONE existing file from the project layout above (give its exact relative path, e.g.
"toolbox/tasks.py") that this feature's code would most naturally be added to or near - do not
invent a filename that isn't in the layout above.

Rules:
- do NOT name any specific library, package, or tool - keep every phrasing tool-agnostic
- do NOT invent new file or module names beyond what the project layout above already shows
- vary the phrasing

Reply with JSON only: {{"file": "<chosen file path>", "prompts": ["<request 1>",
"<request 2>", ...]}}
"""

_REPO_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {"type": "string"},
        "prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["file", "prompts"],
    "additionalProperties": False,
}


# how much of a chosen file's content to embed (defensive cap - every file in our authored
# repos today is well under this)
_FILE_CONTENT_CHARS = 4000


def build_repo_prompt(
    task: str,
    context: Context,
    n: int,
) -> str:
    """Build the LLM prompt asking for `n` requests to add `task` to `context`'s project.

    Returns the prompt string.
    """
    return _REPO_PROMPT.format(context=context.summary, task=task, n=n)


def _repo_file_content(
    context_id: str,
    file: str,
) -> str | None:
    """Read `file`'s current content from the `context_id` repo (pytoolbox, python-api, ...).

    Returns the file's text (capped at `_FILE_CONTENT_CHARS`), or None if `file` doesn't exist
    under that repo or escapes it (e.g. the model invented or mistyped a path despite being told
    not to - the caller should treat that as an invalid candidate, not silently drop the file).
    """
    for repos_dir in (LIBRARY_REPOS_DIR, REPOS_DIR):
        repo_root = repos_dir / context_id
        if not repo_root.is_dir():
            continue
        candidate = (repo_root / file).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError:
            return None
        return (
            candidate.read_text()[:_FILE_CONTENT_CHARS] if candidate.is_file() else None
        )
    return None


def generate_repo_prompts(
    task: str,
    context: Context,
    banned_terms: list[str],
    n: int = 1,
    model: str = llm.DEFAULT_GEN_MODEL,
) -> list[str]:
    """Ask the LLM for `n` natural in-project requests for `task`, framed inside `context`.

    Each accepted request is combined with the file the model chose for it and that file's
    current content, e.g. "In `toolbox/tasks.py`, which currently contains:\\n\\n<code>\\n\\n
    <request>" - so the cached prompt is grounded in real code from the project, not just an
    abstract "this project". If the chosen file doesn't actually exist in `context`'s repo, or
    its real content itself leaks a library name (an existing stub can coincidentally be named
    after one, e.g. pytoolbox's placeholder `slugify_title`), the whole attempt is treated as
    invalid (returns no candidates) rather than rendered ungrounded or leaked. Also rejects any
    candidate whose request text or chosen file path leaks a library name.

    Returns the accepted, file-grounded prompts, deduplicated, in the model's order.
    """
    data, _ = llm.complete_json(
        prompt=build_repo_prompt(task, context, n),
        schema=_REPO_SCHEMA,
        system=_REPO_SYSTEM,
        model=model,
    )
    file = str(data.get("file", "")).strip()
    content = _repo_file_content(context.context_id, file) if file else None
    if file and (content is None or _has_library_leak(content, banned_terms)):
        return []

    candidates = [str(p).strip() for p in data.get("prompts", []) if str(p).strip()]
    accepted = []
    for p in candidates:
        if _has_library_leak(p, banned_terms) or _has_library_leak(file, banned_terms):
            continue
        if file:
            accepted.append(
                f"In `{file}`, which currently contains:\n\n```python\n{content}\n```\n\n{p}"
            )
        else:
            accepted.append(p)
    return list(dict.fromkeys(accepted))


_GENERATORS = {
    "simple": generate_simple_prompts,
    # repo needs the extra `context` argument, and script its own staged cache - both are
    # handled separately in main()
}


def main() -> None:
    """Regenerate one (or all) of the committed claude-generated prompt caches offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type",
        choices=["simple", "script", "repo", "all"],
        default="all",
        help="which prompt cache(s) to regenerate",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="candidates to request per task"
    )
    parser.add_argument(
        "--model", default=llm.DEFAULT_GEN_MODEL, help="generation model id"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only regenerate the first N tasks"
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="script only: generate just the pilot's sample of N tasks (see pilot_seed_tasks)",
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        help="script only: skip tasks already cached with the current prompt (to resume a run)",
    )
    args = parser.parse_args()

    taxonomy = yaml.safe_load(LIBRARY_TAXONOMY_FILE.read_text())
    tasks = yaml.safe_load(TASKS_FILE.read_text())
    contexts = {**build_contexts(REPOS_DIR), **build_contexts(LIBRARY_REPOS_DIR)}
    contexts_by_id = {
        c.context_id: c for c in contexts.values() if c.language == "python"
    }

    prompt_types = ["simple", "script", "repo"] if args.type == "all" else [args.type]
    for prompt_type in prompt_types:
        if prompt_type == "script":
            regenerate_script_stage_cache(
                taxonomy=taxonomy,
                tasks=tasks,
                model=args.model,
                only=(
                    set(pilot_seed_tasks(taxonomy, tasks, args.pilot))
                    if args.pilot is not None
                    else None
                ),
                limit=args.limit,
                skip_cached=args.skip_cached,
            )
            continue

        cache = load_prompt_cache(prompt_type)
        processed = 0
        for entry in taxonomy.get("code", []):
            cid = entry["id"]
            banned_terms = entry["seed_libraries"]
            repo_context = contexts_by_id[entry["repo_hint"]]
            for number, task in tasks.get(cid, {}).items():
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1
                seed_task = f"{cid}.{number}"

                if prompt_type == "repo":
                    accepted = generate_repo_prompts(
                        task, repo_context, banned_terms, n=args.n, model=args.model
                    )
                else:
                    accepted = _GENERATORS[prompt_type](
                        task, banned_terms, n=args.n, model=args.model
                    )
                if accepted:
                    cache[seed_task] = accepted
                print(
                    f"{prompt_type}: {processed} | {seed_task} -> {len(accepted)} accepted"
                )
        write_prompt_cache(prompt_type, cache)
        print(
            f"{prompt_type}: wrote {len(cache)} cached task(s) to {CACHE_FILES[prompt_type]}"
        )


if __name__ == "__main__":
    main()
