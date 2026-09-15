"""LLM seam that diversifies the wording of library-usage tasks (the hybrid layer).

The deterministic build (build.py) already turns each task into one row per mode (repo/simple/
script) via one randomly-picked template. This module adds one extra *claude-generated* wording
per task per mode, cached to simple_prompts.json / script_prompts.json / repo_prompts.json so
the build stays reproducible and needs no model at build time.

The build reads these caches if present; regenerating them (via this file's CLI) is a
deliberate, one-off offline step that needs ANTHROPIC_API_KEY. Until a cache is regenerated,
build.py produces the template-only rows for that mode without it.

Usage: python data/build/library/augment.py [--type simple|script|repo|all] [--n N]
       [--model M] [--limit N]
"""

import argparse
import json
from pathlib import Path

import yaml

from data.build import llm
from data.build.contexts import Context, build_contexts
from data.build.paths import LIBRARY_REPOS_DIR, LIBRARY_TAXONOMY_FILE, REPOS_DIR


_HERE = Path(__file__).parent
TASKS_FILE = _HERE / "tasks.yaml"

# the committed caches the build folds in - {seed_task ("<category_id>.<number>", from
# tasks.yaml): [claude-generated prompts]}
CACHE_FILES: dict[str, Path] = {
    "simple": _HERE / "simple_prompts.json",
    "script": _HERE / "script_prompts.json",
    "repo": _HERE / "repo_prompts.json",
}

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


# --- script: a LibReuseBench-register stdin/stdout statement ---------------------------------

_SCRIPT_SYSTEM = (
    "You write self-contained, library-agnostic stdin/stdout coding-exercise prompts for a "
    "benchmark that measures whether a model reuses libraries when it is free to. Reply with "
    "JSON only."
)

# a real LibReuseBench test prompt (benchmark/libreusebench/data/test.jsonl, LRB/T050), embedded
# so the model calibrates length/tone/precision against an actual example rather than only a
# description of the style - it is NOT related to the tasks we generate for, just a style anchor
_SCRIPT_EXAMPLE = """\
Write a program that reads a list of category labels (strings) from standard input and prints,
for each label, the integer code that would result from replacing every distinct category with
a unique whole number, where the codes are assigned according to the alphabetical (lexicographic)
order of the distinct category strings: the smallest string gets code 0, the next smallest gets
code 1, and so on. Every occurrence of a given category string must be replaced by the same code,
and the original order of the entries must be preserved in the output.

Input format:
- The first line contains a single integer n, the number of entries.
- Each of the next n lines contains one category string (no spaces are stripped except for the
  trailing newline; strings can contain letters, digits, etc., but no internal newlines).

Output format:
- Print n lines. The i-th line must contain the integer code corresponding to the i-th input
  string, following the alphabetical-order encoding rule described above.

Example:
Input:
5
apple
banana
cherry
apple
banana

Output:
0
1
2
0
1\
"""

_SCRIPT_PROMPT = """\
Here is a short, abstract description of a Python feature:
{task}

Here is a real example of the target style, taken from the benchmark this dataset mirrors -
match its length, tone and precision, not just its structural headers:

\"\"\"
{example}
\"\"\"

Turn the feature description above into {n} different single self-contained stdin/stdout
exercise statements, each such that:
- the task reads ALL of its input from standard input and prints its result to standard output;
- it names NO specific library, module, import, class, or function - describe only WHAT to
  compute and the input/output behaviour, leaving the solver free to choose HOW;
- it invents a concrete, unambiguous input format (what appears on standard input) and output
  format (what to print), with any rules needed to make the output unambiguous (ordering,
  tie-breaking, rounding, precision) - the description above gives no concrete data, so you must
  make up a plausible concrete format yourself;
- it is a deterministic pure function of its input (no network, no clock, no reading or writing
  files, no randomness);
- it reads like a genuine, decisive exercise statement, like the example above - state the rules
  plainly and move on. Don't hedge by enumerating every input variant or edge case you can think
  of; only spell one out if the task is genuinely ambiguous without it;
- each of the {n} statements uses a different concrete input format / framing from the others,
  so they read as genuinely different exercises, not just reworded sentences.

Keep each one concise, the way the example above is - natural-language prose, no code - and give
each exactly one short example (one sample input and its expected output; do not give more than
one example per statement).

Reply with JSON only: {{"prompts": ["<statement 1>", "<statement 2>", ...]}}
"""

_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
    "additionalProperties": False,
}


def build_script_prompt(
    task: str,
    n: int,
) -> str:
    """Build the LLM prompt asking for `n` LibReuseBench-register stdin/stdout statements.

    Returns the prompt string.
    """
    return _SCRIPT_PROMPT.format(task=task, n=n, example=_SCRIPT_EXAMPLE)


def generate_script_prompts(
    task: str,
    banned_terms: list[str],
    n: int = 1,
    model: str = llm.DEFAULT_GEN_MODEL,
) -> list[str]:
    """Ask the LLM for `n` LibReuseBench-register stdin/stdout statements for `task`.

    Rejects any candidate that leaks a library name (shared blocklist or `banned_terms`).

    Returns the accepted statements, deduplicated, in the model's order.
    """
    data, _ = llm.complete_json(
        prompt=build_script_prompt(task, n),
        schema=_SCRIPT_SCHEMA,
        system=_SCRIPT_SYSTEM,
        model=model,
    )
    candidates = [str(p).strip() for p in data.get("prompts", []) if str(p).strip()]
    accepted = [p for p in candidates if not _has_library_leak(p, banned_terms)]
    return list(dict.fromkeys(accepted))


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
    "script": generate_script_prompts,
    # repo needs the extra `context` argument, handled separately in main()
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
    args = parser.parse_args()

    taxonomy = yaml.safe_load(LIBRARY_TAXONOMY_FILE.read_text())
    tasks = yaml.safe_load(TASKS_FILE.read_text())
    contexts = {**build_contexts(REPOS_DIR), **build_contexts(LIBRARY_REPOS_DIR)}
    contexts_by_id = {
        c.context_id: c for c in contexts.values() if c.language == "python"
    }

    prompt_types = ["simple", "script", "repo"] if args.type == "all" else [args.type]
    for prompt_type in prompt_types:
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
