"""Stage 03: an independent agent check of every task stage 02 kept.

Stage 02 proves a task runs: its two solutions agree on every test. It cannot tell whether the
task reads clearly, or whether its tests ask for something the text never says. This stage checks
that with fresh model calls, none of which sees how the task was generated:

1. two blind solves, from the task text alone. One is told to use only the standard library, the
   other is free to choose. Both are run on the task's tests. A failure usually means the text is
   unclear or the tests are unfair.
2. a rubric call, which sees the text, the tests, the gold solutions and how the blind solves
   did. It answers a few yes/no questions, and gives a verdict - keep, fix or reject - with a
   note. It judges whether the task is clear, fair and simply worded. It does not judge how much
   code the library saves: that is measured from the solutions, and decides a task's split.

One record per task is appended to benchmark/artefacts/03_agent_reviews.jsonl
(03_agent_reviews_control.jsonl for the control track), with the fingerprint of the task it
reviewed. A task whose current version is already reviewed is skipped, so the stage can be
stopped and rerun.

A task marked "fix" gets one regeneration. The stage ends by printing the command for it
(02_generate.py --seeds ...), which feeds the reviewer's note back to the generator. Run this
stage again afterwards, to review the regenerated tasks.

Blind solves run in the build venv, and the free one's packages are installed there while it
runs. So don't run this stage at the same time as stage 02.

Usage: python benchmark/pipeline/03_agent_review.py [--track reuse|control] [--seeds ID,ID]
       [--limit N] [--workers N] [--model M] [--effort E]
"""

import argparse
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from build import (
    BUILD_PYTHON,
    TRACKS,
    agent_reviews,
    generated_tasks,
    track_artefact_path,
)
from generation import DEFAULT_EFFORT, DEFAULT_MODEL, complete_json, usage_summary
from libreusebench.environment import temporary_packages
from libreusebench.functions import RAISES, evaluate_function_record
from libreusebench.libraries import analyse_source
from lure import pip_package

from src.utils.io import append_jsonl


# installing into the shared build venv isn't safe from two threads at once, and a blind solve
# must run while its own packages are the ones installed
_INSTALL_LOCK = threading.Lock()


# --- the blind solves ---------------------------------------------------------------------------

_SOLVE_SYSTEM = "You are an expert Python developer. Reply with JSON only."

_SOLVE = """\
{prompt}

{rule}

Reply with the complete solution: the function, with the imports and helpers it needs, and no
code that runs when the file is imported.
"""

# what each blind solve is told about libraries
_SOLVE_RULES = {
    "stdlib": "Use only the Python standard library: no third-party packages.",
    "free": (
        "Solve it whichever way you think is best. Third-party packages from PyPI are "
        "available if you want them."
    ),
}

_SOLVE_SCHEMA = {
    "type": "object",
    "properties": {"solution": {"type": "string"}},
    "required": ["solution"],
    "additionalProperties": False,
}


def run_solution(
    solution: str,
    task: dict,
) -> tuple[bool, str]:
    """Run a blind solve on a task's tests, in the build venv, as the benchmark would score it.

    Any third-party packages it imports are installed for the run, and removed afterwards.

    Returns a (passed, empty string or the failure message) tuple.
    """

    def run() -> tuple[bool, str]:
        return evaluate_function_record(
            code=solution,
            function_name=task["function_name"],
            test_inputs=task["test_inputs"],
            expected_outputs=task["expected_outputs"],
            python_executable=BUILD_PYTHON,
        )

    packages = sorted(map(pip_package, analyse_source(solution)[1]))
    if not packages:
        return run()
    with _INSTALL_LOCK, temporary_packages(BUILD_PYTHON, packages) as install:
        if install.failed:
            return False, f"could not install {sorted(install.failed)}"
        return run()


def blind_solve(
    task: dict,
    kind: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Ask for a solution from the task text alone, and run it on the task's tests.

    `kind` is "stdlib" (told to use only the standard library) or "free" (free to choose). A
    standard-library solve that imports a third-party library anyway counts as failed.

    Returns {passed, message, libraries, solution}.
    """
    data, _ = complete_json(
        prompt=_SOLVE.format(prompt=task["prompt"], rule=_SOLVE_RULES[kind]),
        schema=_SOLVE_SCHEMA,
        system=_SOLVE_SYSTEM,
        model=model,
        effort=effort,
    )
    solution = data.get("solution")
    if not solution:
        # no record is written for this task, so a rerun asks again
        raise RuntimeError(f"no usable {kind} blind solve")

    libraries = analyse_source(solution)[1]
    if kind == "stdlib" and libraries:
        passed, message = False, f"imported a third-party library: {libraries}"
    else:
        passed, message = run_solution(solution, task)
    return {
        "passed": passed,
        "message": message,
        "libraries": libraries,
        "solution": solution,
    }


# --- the rubric ---------------------------------------------------------------------------------

_RUBRIC_SYSTEM = (
    "You review programming tasks for a benchmark, strictly and independently. "
    "Reply with JSON only."
)

_RUBRIC = """\
You are reviewing one task for a benchmark that measures whether a coding model reaches for a
third-party library when it is free to choose. A task is a short, plain request for one Python
function, and names no library. {track_note}

Be strict: a task is only useful if a developer could solve it from its text alone, and if the
text reads like a simple request, not a specification.

## The task text

{prompt}

## Its tests

Each test calls the function with the arguments shown, and compares the result leniently: numbers
within a relative tolerance of 0.001, lists and tuples alike, whitespace in strings ignored.
"{raises}" means the call must raise an exception, of any kind.

{cases}

## The reference solutions
{solutions}
## Two blind attempts

Two developers were given only the task text. Neither saw the tests or the reference solutions.

{blind}

## Questions

Answer each with true or false:
- spec_complete: does the text pin down the result for every tested input, with nothing left to
  guess?
- example_consistent: is the example in the text right, and consistent with the tests?
- tests_fair: do the tests check only what the text asks for, with no requirement that is missing
  from the text?
- no_library_leak: is the text free of any hint at a particular library, module or function?
- library_fit: {library_question}
- text_simple: is the text short and plain, with no list of special rules, and no step-by-step
  account of how to compute the result?
- request_natural: does the text read like a request a developer would really make?

Then give a verdict:
- "keep": the task is sound as it is.
- "fix": the task is worth keeping, but something specific must change first.
- "reject": the task is unsound, or can't be made useful.

In the note, say in a few sentences what is wrong, and for a "fix" exactly what must change. A
failed blind attempt usually means the text is unclear or the tests are unfair: work out which,
and say so. Where a test needs a rule the text doesn't state, the fix is to drop that test, not
to add the rule to the text. Don't judge how much work the library does: a task where it does
little is still wanted. Leave the note empty for a clean "keep".
"""

_TRACK_NOTES = {
    "reuse": (
        "It must be solvable both with a library and with the standard library alone."
    ),
    "control": (
        "This one is a control task: one where no third-party library should help."
    ),
}

_LIBRARY_QUESTIONS = {
    "reuse": (
        "does the library solution use its library the way a developer who knew that "
        "library would, however small the library's part is?"
    ),
    "control": "is this a task where no third-party library would help?",
}

_SOLUTION = """
The {name}:
```python
{code}
```
"""

# the rubric's yes/no questions, in order
RUBRIC_QUESTIONS = [
    "spec_complete",
    "example_consistent",
    "tests_fair",
    "no_library_leak",
    "library_fit",
    "text_simple",
    "request_natural",
]

_RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        **{question: {"type": "boolean"} for question in RUBRIC_QUESTIONS},
        "verdict": {"type": "string", "enum": ["keep", "fix", "reject"]},
        "note": {"type": "string"},
    },
    "required": [*RUBRIC_QUESTIONS, "verdict", "note"],
    "additionalProperties": False,
}


def _describe_blind(
    name: str,
    solve: dict,
) -> str:
    """Describe how one blind solve did, with its code if it failed.

    Returns the description.
    """
    used = f" It used {', '.join(solve['libraries'])}." if solve["libraries"] else ""
    if solve["passed"]:
        return f"The {name} attempt passed every test.{used}"
    return (
        f"The {name} attempt FAILED: {solve['message']}{used}\n"
        f"Its code:\n```python\n{solve['solution'].strip()}\n```"
    )


def build_rubric_request(
    task: dict,
    blind: dict[str, dict],
    track: str = "reuse",
) -> str:
    """Build the rubric request for one task: its text, tests, solutions and blind solves.

    Returns the request text.
    """
    solutions = ""
    if track == "reuse":
        # only a reuse-track task has a library solution, and so a reuse value to show
        solutions += _SOLUTION.format(
            name=(
                f"library solution, using {task['primary_library']} "
                f"({task['extlib_lloc']} logical lines of code, against "
                f"{task['stdlib_lloc']} for the standard-library solution)"
            ),
            code=task["extlib_solution"].strip(),
        )
    solutions += _SOLUTION.format(
        name="standard-library solution",
        code=task["stdlib_solution"].strip(),
    )
    return _RUBRIC.format(
        track_note=_TRACK_NOTES[track],
        prompt=task["prompt"],
        raises=RAISES,
        cases="\n".join(
            f"Test {number}: {task['function_name']}(*{arguments}) -> {expected}"
            for number, (arguments, expected) in enumerate(
                zip(task["test_inputs"], task["expected_outputs"]),
                start=1,
            )
        ),
        solutions=solutions,
        blind="\n\n".join(
            [
                _describe_blind("standard-library-only", blind["stdlib"]),
                _describe_blind("free-choice", blind["free"]),
            ]
        ),
        library_question=_LIBRARY_QUESTIONS[track],
    )


def review_task(
    task: dict,
    track: str = "reuse",
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Review one task: two blind solves, then the rubric call that sees how they did.

    Returns the task's review record.
    """
    blind = {
        kind: blind_solve(task, kind, model=model, effort=effort)
        for kind in ("stdlib", "free")
    }
    data, resolved_model = complete_json(
        prompt=build_rubric_request(task, blind, track),
        schema=_RUBRIC_SCHEMA,
        system=_RUBRIC_SYSTEM,
        model=model,
        effort=effort,
    )
    if data.get("verdict") not in ("keep", "fix", "reject"):
        raise RuntimeError("no usable rubric reply")
    return {
        "seed_id": task["seed_id"],
        "task_hash": task["task_hash"],
        "track": track,
        "verdict": data["verdict"],
        "note": data.get("note", ""),
        "rubric": {question: data.get(question) for question in RUBRIC_QUESTIONS},
        "blind": blind,
        "model": resolved_model,
        "effort": effort,
    }


def seeds_to_fix(
    track: str = "reuse",
) -> list[str]:
    """The seeds whose current task the review marked "fix", and that can still be regenerated.

    A task gets one regeneration only: one that was itself regenerated for a fix (it has a
    `fixed_from`) is not sent back a second time, so a "fix" on it stands as a rejection.

    Returns the seed ids.
    """
    reviews = agent_reviews(track)
    return [
        task["seed_id"]
        for task in generated_tasks(track)
        if reviews.get(task["seed_id"], {}).get("verdict") == "fix"
        and not task.get("fixed_from")
    ]


def main() -> None:
    """Review each kept task that has no review of its current version yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="reuse",
        help="which track's tasks to review",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed ids or numbers to review, e.g. 723,BigCodeBench/340",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only review the first N tasks"
    )
    parser.add_argument("--workers", type=int, default=1, help="tasks reviewed at once")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="review model id")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, help="review effort (low to max)"
    )
    args = parser.parse_args()

    if not Path(BUILD_PYTHON).exists():
        sys.exit(
            f"no build venv at {BUILD_PYTHON} - create it with "
            "benchmark/pipeline/setup_build_venv.sh, or set LRB_BUILD_PYTHON"
        )

    output = track_artefact_path("03_agent_reviews.jsonl", args.track)
    reviewed = agent_reviews(args.track)
    todo = [
        task for task in generated_tasks(args.track) if task["seed_id"] not in reviewed
    ]
    if args.seeds:
        wanted = {
            seed if "/" in seed else f"BigCodeBench/{seed}"
            for seed in args.seeds.split(",")
        }
        todo = [task for task in todo if task["seed_id"] in wanted]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} {args.track} task(s) to review with {args.model}")

    def review(
        task: dict,
    ) -> dict[str, Any] | None:
        """Review one task, in a worker thread.

        Returns the review record, or None if the run itself failed (e.g. an api timeout).
        """
        try:
            return review_task(task, args.track, model=args.model, effort=args.effort)
        except Exception as error:
            # no record is written, so a rerun reviews this task again
            print(f"  {task['seed_id']}: {type(error).__name__}: {error}")
            return None

    # only this thread writes the artefact, as each review finishes
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(review, task): task["seed_id"] for task in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record is None:
                print(f"{count}/{len(todo)} | {futures[future]} -> run failed")
                continue
            append_jsonl(record, output)
            failed = [
                kind for kind, solve in record["blind"].items() if not solve["passed"]
            ]
            print(
                f"{count}/{len(todo)} | {record['seed_id']} -> {record['verdict']}"
                + (f" (blind solve failed: {', '.join(failed)})" if failed else "")
                + (f": {record['note']}" if record["note"] else "")
            )

    # where every task of the track now stands, counting earlier runs too
    verdicts = Counter(
        review["verdict"] for review in agent_reviews(args.track).values()
    )
    print(
        ", ".join(f"{verdict}: {count}" for verdict, count in sorted(verdicts.items()))
    )
    to_fix = seeds_to_fix(args.track)
    if to_fix:
        track_option = "" if args.track == "reuse" else f" --track {args.track}"
        print(
            f"{len(to_fix)} task(s) to regenerate once, then review again:\n"
            f"  python benchmark/pipeline/02_generate.py{track_option} "
            f"--seeds {','.join(seed.split('/')[1] for seed in to_fix)}"
        )
    print(usage_summary())


if __name__ == "__main__":
    main()
