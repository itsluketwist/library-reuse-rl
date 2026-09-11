"""Stage 04: produce a standard-library gold solution and mint the test cases from it.

For each reworded prompt we ask the model for a correct solution that uses ONLY the standard
library, together with a set of diverse test inputs, then run that solution to produce the
expected output for each input. A stdlib solution runs with no package installs, so minting is
reliable and fast, and requiring one to exist is already half of the dual-solvable gate (stage 05
then confirms an independent library solution reproduces the same outputs - the reuse path).

The run is resumable and supports --limit / --model. The gold solution is executed with the build
interpreter. --track selects the stage-03 pool to mint tests for; for the "control" track (no
external-library seeds), this stage's output is already the fully validated pool - there is no
stage-05 equivalent to run afterwards.

Apparent failures here are dominated by sampling variance, not model capability, so a task is
retried up to `--attempts` times, keeping the first that mints enough clean tests (mirroring stage
05's best-of-N). The shipped reuse track was built by running this stage (and stage 05) up to three
times with Sonnet, then up to three more with Opus on whatever was still not kept.

Usage: python benchmark/pipeline/04_make_tests.py [--track reuse|control] [--limit N] [--model M] [--attempts N]
"""

import argparse

from build import BUILD_PYTHON, succeeded_ids, track_artefact_path
from generation import DEFAULT_GEN_MODEL, complete_json
from libreusebench.environment import run_program
from libreusebench.libraries import analyse_source

from src.utils.io import append_jsonl, load_jsonl


# a problem needs at least this many cleanly-running inputs to be worth keeping
MIN_TESTS = 3

_GOLD_SYSTEM = (
    "You are an expert Python programmer. You write correct stdin/stdout solutions using ONLY the "
    "Python standard library, plus thorough test inputs. Reply with JSON only."
)

_GOLD_PROMPT = """\
Here is a programming task. Write a correct reference solution and a set of test inputs.

Task:
{prompt}

Provide:
1. "solution": a complete, correct Python program that reads from standard input and prints to
   standard output exactly as the task specifies, using ONLY the Python standard library (no
   third-party/pip packages such as numpy, pandas, requests, etc.).
2. "inputs": a list of 6 to 10 diverse test inputs. Each item is the FULL standard-input text for
   one run of the program. Cover typical cases and edge cases (empty, minimal, boundary values,
   duplicates, negatives, ...). Every input must be valid for the task's stated input format.

Reply with JSON only: {{"solution": "<python source>", "inputs": ["<stdin 1>", "<stdin 2>", ...]}}
"""

_GOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "solution": {"type": "string"},
        "inputs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["solution", "inputs"],
    "additionalProperties": False,
}


def build_gold_prompt(
    prompt: str,
) -> str:
    """Build the stdlib gold-solution + test-inputs prompt for one task.

    Returns the prompt string.
    """
    return _GOLD_PROMPT.format(prompt=prompt.strip())


def gold_solution(
    prompt: str,
    model: str = DEFAULT_GEN_MODEL,
) -> tuple[str, list[str], str] | None:
    """Generate a standard-library gold solution and candidate test inputs for a task.

    A single attempt - a failed task is retried at the run level (rerun, optionally with a
    stronger model).

    Returns a (solution, inputs, resolved model id) tuple, or None if nothing usable came back.
    """
    data, resolved_model = complete_json(
        prompt=build_gold_prompt(prompt),
        schema=_GOLD_SCHEMA,
        system=_GOLD_SYSTEM,
        model=model,
    )
    solution = str(data.get("solution", "")).strip()
    inputs = [str(i) for i in data.get("inputs", []) if isinstance(i, str)]
    if not solution or not inputs:
        return None
    return solution, inputs, resolved_model


def _mint(
    solution: str,
    inputs: list[str],
) -> tuple[list[str], list[str]]:
    """Run the gold solution on each input, keeping only inputs it handles cleanly and DETERMINISTICALLY.

    Each input is run twice; it is kept only if both runs succeed and produce the same output. This
    rejects a gold that depends on randomness, the clock, or anything non-reproducible - its
    outputs would be meaningless as test cases.

    Returns the (kept inputs, their expected outputs) tuple.
    """
    kept_inputs: list[str] = []
    expected: list[str] = []
    for stdin in inputs:
        ok1, out1 = run_program(solution, stdin, python_executable=BUILD_PYTHON)
        ok2, out2 = run_program(solution, stdin, python_executable=BUILD_PYTHON)
        if ok1 and ok2 and out1 == out2:
            kept_inputs.append(stdin)
            expected.append(out1)
    return kept_inputs, expected


def _attempt_once(
    prompt: str,
    model: str,
) -> dict:
    """Draw one stdlib gold solution for a task and mint its test cases.

    Returns a result dict: always carries 'passed'; on success the gold solution, its resolved
    model, and the minted (inputs, expected_outputs); on failure a drop_reason.
    """
    result = gold_solution(prompt, model=model)
    if result is None:
        return {"passed": False, "drop_reason": "no gold solution"}

    solution, candidate_inputs, gold_model = result
    # the gold must actually be standard-library-only (proves stdlib-solvability)
    if analyse_source(solution)[1]:
        return {"passed": False, "drop_reason": "gold not stdlib"}

    inputs, expected = _mint(solution, candidate_inputs)
    if len(inputs) < MIN_TESTS:
        return {"passed": False, "drop_reason": "too few clean tests"}

    return {
        "passed": True,
        "gold_stdlib": solution,
        "gold_stdlib_model": gold_model,
        "inputs": inputs,
        "expected_outputs": expected,
    }


def main() -> None:
    """Generate a stdlib gold solution per problem and mint its test cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["reuse", "control"],
        default="reuse",
        help="which stage-03 rewritten pool to mint tests for",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only solve the first N"
    )
    parser.add_argument(
        "--model", default=DEFAULT_GEN_MODEL, help="generation model id"
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="best-of-N: draw up to N gold solutions, keep the task on the first that passes",
    )
    args = parser.parse_args()

    rows = [
        row
        for row in load_jsonl(track_artefact_path("03_rewritten.jsonl", args.track))
        if row.get("kept")
    ]
    out_path = track_artefact_path("04_tests.jsonl", args.track)
    done = succeeded_ids(out_path)
    print(
        f"{len(rows)} rewritten problems"
        + (f", {len(done)} already done" if done else "")
        + (f", best-of-{args.attempts}" if args.attempts > 1 else "")
    )

    processed = kept = 0
    for row in rows:
        seed_id = row["seed_id"]
        if seed_id in done:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        # draw up to N solutions, stopping at the first that mints enough clean tests
        outcome: dict = {}
        for _ in range(max(1, args.attempts)):
            outcome = _attempt_once(row["prompt"], model=args.model)
            if outcome["passed"]:
                break

        fields = {key: value for key, value in outcome.items() if key != "passed"}
        if not outcome["passed"]:
            append_jsonl({"seed_id": seed_id, "kept": False} | fields, out_path)
            continue

        kept += 1
        append_jsonl(
            {
                "seed_id": seed_id,
                "category": row["category"],
                "seed_libs": row["seed_libs"],
                "prompt": row["prompt"],
                "kept": True,
            }
            | fields,
            out_path,
        )
        print(
            f"solved {processed} | kept {kept} | {len(fields['inputs'])} tests | last {seed_id}"
        )

    print(f"\ndone: processed {processed}, kept {kept}")


if __name__ == "__main__":
    main()
