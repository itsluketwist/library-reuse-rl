"""Stage 02: filter BCB problems down to candidates for a library-agnostic eval.

Two passes. First a cheap, deterministic hard-exclude drops only the genuinely impossible
(live-network or plot/GUI-only problems). Everything else - including heavy-ML and stdlib-only -
goes to a permissive LLM judge that scores use-case fit (deterministic, text-serialisable, and for
the reuse track dual-solvable) as keep / maybe / drop. We carry keep AND maybe forward; the later
executable gates prune whatever cannot actually produce valid solutions.

--track selects which pool is judged: "reuse" (default) judges the ext-lib seeds for all three
criteria; "control" judges the stdlib-only seeds (BCB solutions with no external library - tasks
where library reuse isn't expected) for determinism and text-serialisability only, since they have
no library solution to be dual-solvable with.

The judge makes one model call per surviving problem, so the run is resumable: rows already in
the output artefact are skipped. Use --limit to judge only the first N (for a cheap smoke test)
and --dry-run to skip the model entirely.

Usage: python benchmark/pipeline/02_filter.py [--track reuse|control] [--limit N] [--dry-run] [--model M]
"""

import argparse
import os

from build import artefact_path, hard_exclude_reason, track_artefact_path
from generation import complete_json

from src.utils.io import append_jsonl, load_jsonl


# --- the use-case judge ---------------------------------------------------------------

# the judge is a simple classification, so it defaults to a cheaper model (env-overridable)
DEFAULT_JUDGE_MODEL = os.environ.get("EVAL_DATASET_JUDGE_MODEL", "claude-haiku-4-5")

_JUDGE_SYSTEM = (
    "You are curating coding problems for a benchmark that measures whether a model chooses "
    "to reuse a third-party library when it is free to. Judge each problem strictly."
)

# the judge sees the ORIGINAL problem and solution, but both get rewritten later (stage 03) into
# a library-agnostic stdin/stdout task - so it must judge the underlying problem's suitability
# for that rewrite, not its current library-specific wording or imports
_JUDGE_PROMPT = """\
We are building a benchmark of library-agnostic coding problems. Every problem below will be
REWRITTEN into a self-contained task that reads its input from standard input and prints its
result to standard output, naming no library and giving no function signature - so a model
solving it is free to choose how (a third-party library, the standard library, or by hand).

Here is the original problem and its reference solution. Both will be rewritten, so judge the
UNDERLYING problem, not its current wording, imports, or function signature.

Problem description:
{description}

Reference solution (currently uses libraries: {libs}):
```python
{solution}
```

Decide whether the underlying problem can be rewritten into such a stdin/stdout task meeting ALL:
1. deterministic, with NO randomness: the output is a pure function of the inputs (no network, no
   current date/time). Reject any problem that involves random number generation - EVEN if a seed
   is given - because reproducing a random sequence pins a specific library's generator, which
   defeats the library-agnostic goal.
2. text-serialisable: the rewrite reshapes the input/output, so judge the underlying DATA, not the
   original's file conventions. A single input file's CONTENTS simply become the stdin, and a
   result the original writes to a single file (or returns as a file path) can instead be PRINTED
   to stdout. The task fits when its inputs are one text stream and its full result is plain text
   (numbers, strings, lists, tables). Reject only when the result is genuinely non-text (an image,
   plot, trained model, or binary) or the task genuinely needs many files at once (scanning a
   directory / several input files, or producing several output files).{dual_solvable_criterion}

Use "keep" only if {all_criteria} clearly hold, "maybe" if plausible but uncertain, "drop"
otherwise. Give a short kebab-case category (e.g. datetime, html-parsing, stats) and a one-sentence
reason.
"""

# criterion 3 only applies to the reuse track - a control-track problem has no library solution to
# be dual-solvable with, so its judge prompt is these same two criteria without it
_DUAL_SOLVABLE_CRITERION = """
3. dual-solvable: it can be solved BOTH with a third-party library (the natural way) AND with
   only the Python standard library / hand-written code (possible, if more effort)."""

# structured-output schema, so the reply is always a valid verdict record
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["keep", "maybe", "drop"]},
        "category": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "category", "reason"],
    "additionalProperties": False,
}

_VALID_VERDICTS = {"keep", "maybe", "drop"}


def build_judge_prompt(
    description: str,
    solution: str,
    libs: list[str],
    require_dual_solvable: bool = True,
) -> str:
    """Build the judge prompt for one problem.

    `require_dual_solvable` is False for the control track, whose problems have no library
    solution to be dual-solvable with - the judge then scores only determinism and
    text-serialisability.

    Returns the prompt string.
    """
    return _JUDGE_PROMPT.format(
        description=description.strip(),
        solution=solution.strip(),
        libs=", ".join(libs) or "none",
        dual_solvable_criterion=_DUAL_SOLVABLE_CRITERION
        if require_dual_solvable
        else "",
        all_criteria="all three" if require_dual_solvable else "both",
    )


def normalise_verdict(
    data: dict,
) -> dict[str, str]:
    """Normalise a judge result dict, failing safe to "drop" on anything invalid.

    A missing/invalid verdict is treated as "drop" so a bad response never silently promotes a
    problem into the dataset.

    Returns a {"verdict", "category", "reason"} dict.
    """
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        return {"verdict": "drop", "category": "", "reason": "unparseable judge reply"}
    return {
        "verdict": verdict,
        "category": str(data.get("category", "")).strip(),
        "reason": str(data.get("reason", "")).strip(),
    }


def judge_problem(
    description: str,
    solution: str,
    libs: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
    require_dual_solvable: bool = True,
) -> dict[str, str]:
    """Run the use-case judge on a single problem via structured output.

    Returns the normalised {"verdict", "category", "reason"} record.
    """
    data, _ = complete_json(
        prompt=build_judge_prompt(description, solution, libs, require_dual_solvable),
        schema=_JUDGE_SCHEMA,
        system=_JUDGE_SYSTEM,
        model=model,
    )
    return normalise_verdict(data)


def _done_seed_ids(
    out_path,
) -> set[str]:
    """Seed ids already written to the output artefact, so a resumed run skips them.

    Returns the set of completed seed ids (empty if the artefact does not exist yet).
    """
    if not out_path.exists():
        return set()
    return {row["seed_id"] for row in load_jsonl(out_path)}


def _annotate(
    row: dict,
    dry_run: bool,
    model: str,
    require_dual_solvable: bool = True,
) -> dict:
    """Judge one BCB problem, returning only its seed id and the verdict annotation.

    The candidate record is deliberately slim - it references the seed problem by id rather than
    copying its prompt/solution, which later stages re-join from the stage-01 artefact.

    Returns a {seed_id, std_libs, ext_libs, stage, verdict, category, reason} record.
    """
    # libraries were classified at fetch (stage 01) and carried on the row
    std_libs, ext_libs = row["std_libs"], row["ext_libs"]
    annotation = {"seed_id": row["seed_id"], "std_libs": std_libs, "ext_libs": ext_libs}

    # pass 1: drop the genuinely impossible without spending a model call (a no-op for the
    # control track, whose seeds have no external libs to flag as network/plot-gui)
    reason = hard_exclude_reason(ext_libs)
    if reason is not None:
        return annotation | {
            "stage": "hard-exclude",
            "verdict": "drop",
            "category": "",
            "reason": reason,
        }

    # dry runs stop before the model: mark surviving problems as unjudged
    if dry_run:
        return annotation | {
            "stage": "judge",
            "verdict": "maybe",
            "category": "",
            "reason": "not judged",
        }

    # pass 2: the LLM use-case judge (description + reference solution)
    verdict = judge_problem(
        description=row["instruct_prompt"] or row["complete_prompt"],
        solution=row["complete_prompt"] + row["canonical_solution"],
        libs=ext_libs,
        model=model,
        require_dual_solvable=require_dual_solvable,
    )
    return annotation | {"stage": "judge", **verdict}


def main() -> None:
    """Filter the stage-01 rows into annotated candidates, streaming results for resumability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["reuse", "control"],
        default="reuse",
        help="reuse: ext-lib seeds, judged for dual-solvability too; "
        "control: stdlib-only seeds, no dual-solvable criterion",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only judge the first N problems"
    )
    parser.add_argument("--dry-run", action="store_true", help="skip the LLM judge")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="judge model id")
    args = parser.parse_args()

    # the reuse track judges ext-lib seeds; the control track judges the stdlib-only seeds
    # split off at stage 01 (never has a dual-solvable path, so that criterion is dropped)
    seed_file = (
        "01_bcb_extlibs.jsonl" if args.track == "reuse" else "01_bcb_stdlibs.jsonl"
    )
    rows = load_jsonl(artefact_path(seed_file))
    out_path = track_artefact_path("02_candidates.jsonl", args.track)
    done = _done_seed_ids(out_path)
    if done:
        print(f"resuming: {len(done)} problems already annotated")

    judged = 0
    for row in rows:
        if row["seed_id"] in done:
            continue
        if args.limit is not None and judged >= args.limit:
            break

        record = _annotate(
            row,
            dry_run=args.dry_run,
            model=args.model,
            require_dual_solvable=(args.track == "reuse"),
        )
        append_jsonl(record, out_path)

        # count only the problems that reached the (costly) judge, so --limit bounds model calls
        if record["stage"] == "judge" and not args.dry_run:
            judged += 1
        if (judged and judged % 25 == 0) or args.limit is not None:
            print(
                f"judged {judged} | last {record['seed_id']}: {record['verdict']} ({record['reason']})"
            )

    _summarise(out_path)


def _summarise(
    out_path,
) -> None:
    """Print verdict counts for the full output artefact."""
    from collections import Counter

    rows = load_jsonl(out_path)
    verdicts = Counter(r["verdict"] for r in rows)
    kept = sum(1 for r in rows if r["verdict"] in ("keep", "maybe"))
    print(f"\ntotal annotated: {len(rows)} -> carried forward (keep+maybe): {kept}")
    print(f"verdicts: {dict(verdicts)}")


if __name__ == "__main__":
    main()
