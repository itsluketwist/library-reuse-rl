"""Stage 03: reword each candidate into a library-agnostic, natural-language stdin/stdout task.

For every kept/maybe candidate we give the model the original BigCodeBench task (its description,
reference solution, and tests, purely so it understands the required behaviour) and ask for a
single self-contained task statement in plain natural language: read the input from standard
input, print the result to standard output, name no library and no function. The rewritten prompt
is the only thing carried forward - the ground-truth test cases are produced later (stage 04) by
running an actual stdin/stdout solution, so there is no need for the model to write an adapter.

One model call per candidate, so the run is resumable and supports --limit / --model. --track
selects the stage-02 candidate pool to reword ("reuse" ext-lib seeds by default, or "control" the
stdlib-only seeds where library reuse isn't expected) - the reword prompt itself doesn't change.

Usage: python benchmark/pipeline/03_rewrite.py [--track reuse|control] [--limit N] [--model M]
"""

import argparse

from build import artefact_path, full_solution, succeeded_ids, track_artefact_path
from generation import DEFAULT_GEN_MODEL, complete_json

from src.utils.io import append_jsonl, load_jsonl


_REWRITE_SYSTEM = (
    "You reword coding problems into self-contained, library-agnostic tasks for a benchmark that "
    "measures whether a model reuses libraries when it is free to. Reply with JSON only."
)

_REWRITE_PROMPT = """\
Here is a coding problem. Its reference solution and tests are given only so you understand the
exact required behaviour.

Task:
{description}

Reference solution:
```python
{solution}
```

Tests:
```python
{test}
```

Reword this into a single self-contained task, described in plain natural language, such that:
- the task reads ALL of its input from standard input and prints its result to standard output;
- it names NO specific library, module, import, class, or function - describe only WHAT to
  compute and the input/output behaviour, leaving the solver free to choose HOW (do not narrate a
  step-by-step algorithm or hint at a particular implementation approach);
- it specifies the input format (what appears on standard input) and the output format (what to
  print), with any rules needed to make the output unambiguous (ordering, tie-breaking, rounding),
  precisely enough that each input has exactly one correct output;
- the task is a deterministic pure function of its input (no network, no clock, no reading or
  writing files). If the original solution only calls a random generator to manufacture its OWN
  example data (e.g. it generates a random list, then computes a statistic over it), do NOT carry
  that generation into the rewrite and do NOT take a seed as input either - reproducing a sequence
  from a seed still pins a specific library's generator. Instead, have the rewritten task read that
  already-generated data ITSELF directly from standard input (e.g. "read a list of integers from
  stdin", not "read a seed and regenerate the list"), so no randomness remains anywhere in the
  rewritten task.

Keep it concise. Write it the way a clear programming-exercise statement reads - natural-language
prose, no code - and include AT MOST ONE short example (one sample input and its expected output);
do not give more than one example.

Reply with JSON only: {{"prompt": "<the reworded task>"}}
"""

# structured-output schema - the reply is always a single natural-language prompt string
_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}


def build_rewrite_prompt(
    description: str,
    solution: str,
    test: str,
) -> str:
    """Build the rewrite prompt for one problem.

    Returns the prompt string.
    """
    return _REWRITE_PROMPT.format(
        description=description.strip(),
        solution=solution.strip(),
        test=test.strip(),
    )


def rewrite_problem(
    description: str,
    solution: str,
    test: str,
    model: str = DEFAULT_GEN_MODEL,
) -> str | None:
    """Reword one problem into a library-agnostic natural-language stdin/stdout task.

    A single attempt - a failed task is retried at the run level (rerun, optionally with a
    stronger model).

    Returns the reworded prompt, or None if the model returned nothing usable.
    """
    data, _ = complete_json(
        prompt=build_rewrite_prompt(description, solution, test),
        schema=_REWRITE_SCHEMA,
        system=_REWRITE_SYSTEM,
        model=model,
    )
    prompt = str(data.get("prompt", "")).strip()
    return prompt or None


def main() -> None:
    """Reword candidates into agnostic natural-language stdin/stdout task statements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["reuse", "control"],
        default="reuse",
        help="which stage-02 candidate pool to rewrite",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only rewrite the first N"
    )
    parser.add_argument(
        "--model", default=DEFAULT_GEN_MODEL, help="generation model id"
    )
    args = parser.parse_args()

    # every keep/maybe already has a real library-reuse path (reuse track) or is confirmed
    # deterministic/text-serialisable (control track) - either way it's ready to reword
    candidates = [
        row
        for row in load_jsonl(track_artefact_path("02_candidates.jsonl", args.track))
        if row["verdict"] in ("keep", "maybe")
    ]
    # the candidate records are slim (just the seed id + verdict), so re-join the full seed
    # problem from the matching stage-01 artefact by seed_id
    seed_file = (
        "01_bcb_extlibs.jsonl" if args.track == "reuse" else "01_bcb_stdlibs.jsonl"
    )
    seeds = {row["seed_id"]: row for row in load_jsonl(artefact_path(seed_file))}
    out_path = track_artefact_path("03_rewritten.jsonl", args.track)
    done = succeeded_ids(out_path)
    print(
        f"{len(candidates)} candidates carried forward"
        + (f", {len(done)} already done" if done else "")
    )

    processed = kept = 0
    for candidate in candidates:
        seed_id = candidate["seed_id"]
        if seed_id in done:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        seed = seeds[seed_id]
        prompt = rewrite_problem(
            description=seed["instruct_prompt"] or seed["complete_prompt"],
            solution=full_solution(seed["complete_prompt"], seed["canonical_solution"]),
            test=seed["test"],
            model=args.model,
        )
        if prompt is None:
            append_jsonl(
                {"seed_id": seed_id, "kept": False, "drop_reason": "unusable rewrite"},
                out_path,
            )
            continue

        kept += 1
        append_jsonl(
            {
                "seed_id": seed_id,
                "category": candidate.get("category", ""),
                "seed_libs": candidate["ext_libs"],
                "prompt": prompt,
                "kept": True,
                "source": "bigcodebenchreuse"
                if args.track == "reuse"
                else "bigcodebenchcontrol",
            },
            out_path,
        )
        print(f"rewrote {processed} | kept {kept} | last {seed_id}")

    print(f"\ndone: processed {processed}, kept {kept}")


if __name__ == "__main__":
    main()
