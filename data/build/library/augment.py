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

# the committed caches the build folds in - {task text: [claude-generated prompts]}
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

    Returns the {task text: [claude-generated prompts]} mapping.
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


# --- simple: a BigCodeBench-style function-signature prompt ---------------------------------

_SIMPLE_SYSTEM = (
    "You write BigCodeBench-style function-signature prompts for a dataset that measures "
    "whether a model reaches for a library when solving a coding task. Reply with JSON only."
)

_SIMPLE_PROMPT = """\
A developer wants a Python function that does the following:
{task}

Write {n} different BigCodeBench-style prompts for this function. For each one:
- invent a plausible function name, argument names and (optionally) type hints - vary the
  signature and framing across the {n} versions
- write a docstring describing the task in your own words
- include exactly one made-up `>>> ` doctest example showing a call and its (illustrative, not
  necessarily exact) return value
- do NOT name any specific library, package, or tool anywhere in the prompt - describe only
  WHAT the function should do, never HOW

Reply with JSON only: {{"prompts": ["<prompt 1, a signature line followed by a docstring with
the description and doctest example>", "<prompt 2>", ...]}}
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
) -> str:
    """Build the LLM prompt asking for `n` BigCodeBench-style function-stub prompts for `task`.

    Returns the prompt string.
    """
    return _SIMPLE_PROMPT.format(task=task, n=n)


def generate_simple_prompts(
    task: str,
    banned_terms: list[str],
    n: int = 1,
    model: str = llm.DEFAULT_GEN_MODEL,
) -> list[str]:
    """Ask the LLM for `n` BigCodeBench-style function-stub prompts for `task`.

    Rejects any candidate that leaks a library name (shared blocklist or `banned_terms`).

    Returns the accepted prompts, deduplicated, in the model's order.
    """
    data, _ = llm.complete_json(
        prompt=build_simple_prompt(task, n),
        schema=_SIMPLE_SCHEMA,
        system=_SIMPLE_SYSTEM,
        model=model,
    )
    candidates = [str(p).strip() for p in data.get("prompts", []) if str(p).strip()]
    accepted = [p for p in candidates if not _has_library_leak(p, banned_terms)]
    return list(dict.fromkeys(accepted))


# --- script: a LibReuseBench-register stdin/stdout statement ---------------------------------

_SCRIPT_SYSTEM = (
    "You write self-contained, library-agnostic stdin/stdout coding-exercise prompts for a "
    "benchmark that measures whether a model reuses libraries when it is free to. Reply with "
    "JSON only."
)

_SCRIPT_PROMPT = """\
Here is a short, abstract description of a Python feature:
{task}

Turn this into {n} different single self-contained stdin/stdout exercise statements, each such
that:
- the task reads ALL of its input from standard input and prints its result to standard output;
- it names NO specific library, module, import, class, or function - describe only WHAT to
  compute and the input/output behaviour, leaving the solver free to choose HOW;
- it invents a concrete, unambiguous input format (what appears on standard input) and output
  format (what to print), with any rules needed to make the output unambiguous (ordering,
  tie-breaking, rounding, precision) - the description above gives no concrete data, so you must
  make up a plausible concrete format yourself;
- it is a deterministic pure function of its input (no network, no clock, no reading or writing
  files, no randomness);
- each of the {n} statements uses a different concrete input format / framing from the others,
  so they read as genuinely different exercises, not just reworded sentences.

Keep each one concise - the way a clear programming-exercise statement reads, natural-language
prose, no code - and give each exactly one short example (one sample input and its expected
output; do not give more than one example per statement).

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
    return _SCRIPT_PROMPT.format(task=task, n=n)


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

Rules:
- do NOT name any specific library, package, or tool - keep every phrasing tool-agnostic
- do NOT invent new file or module names beyond what the project layout above already shows
- vary the phrasing

Reply with JSON only: {{"prompts": ["<request 1>", "<request 2>", ...]}}
"""

_REPO_SCHEMA = {
    "type": "object",
    "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["prompts"],
    "additionalProperties": False,
}


def build_repo_prompt(
    task: str,
    context: Context,
    n: int,
) -> str:
    """Build the LLM prompt asking for `n` requests to add `task` to `context`'s project.

    Returns the prompt string.
    """
    return _REPO_PROMPT.format(context=context.summary, task=task, n=n)


def generate_repo_prompts(
    task: str,
    context: Context,
    banned_terms: list[str],
    n: int = 1,
    model: str = llm.DEFAULT_GEN_MODEL,
) -> list[str]:
    """Ask the LLM for `n` natural in-project requests for `task`, framed inside `context`.

    Rejects any candidate that leaks a library name (shared blocklist or `banned_terms`).

    Returns the accepted requests, deduplicated, in the model's order.
    """
    data, _ = llm.complete_json(
        prompt=build_repo_prompt(task, context, n),
        schema=_REPO_SCHEMA,
        system=_REPO_SYSTEM,
        model=model,
    )
    candidates = [str(p).strip() for p in data.get("prompts", []) if str(p).strip()]
    accepted = [p for p in candidates if not _has_library_leak(p, banned_terms)]
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
            banned_terms = entry["seed_libraries"]
            repo_context = contexts_by_id[entry["repo_hint"]]
            for task in tasks.get(entry["id"], []):
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1

                if prompt_type == "repo":
                    accepted = generate_repo_prompts(
                        task, repo_context, banned_terms, n=args.n, model=args.model
                    )
                else:
                    accepted = _GENERATORS[prompt_type](
                        task, banned_terms, n=args.n, model=args.model
                    )
                if accepted:
                    cache[task] = accepted
                print(
                    f"{prompt_type}: {processed} | {task[:60]!r} -> {len(accepted)} accepted"
                )
        write_prompt_cache(prompt_type, cache)
        print(
            f"{prompt_type}: wrote {len(cache)} cached task(s) to {CACHE_FILES[prompt_type]}"
        )


if __name__ == "__main__":
    main()
