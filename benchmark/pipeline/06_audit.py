"""Stage 06: audit the build and write benchmark/docs/REPORT.md.

Reads whatever stage artefacts exist and reports, for the reuse track and the control track: how
many seeds survived each stage and why the rest were dropped, what the agent review and the human
review decided, and the spread of the shipped splits by reuse tier, library and domain. Purely a
reporting stage - it changes no data.

--provisional reports on the provisional splits (benchmark/artefacts/provisional/) in place of the
final ones (benchmark/libreusebench/data/).

Usage: python benchmark/pipeline/06_audit.py [--provisional]
"""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

from build import (
    DATA_DIR,
    PROVISIONAL_DIR,
    REPORT_PATH,
    SEED_FILES,
    agent_kept_tasks,
    agent_reviews,
    artefact_path,
    generated_tasks,
    latest_records,
    track_artefact_path,
)
from review import approved_tasks, pending, verdicts

from src.utils.io import load_jsonl


# the splits each track ships
TRACK_SPLITS = {"reuse": ("test",), "control": ("control",)}


def _md_counts(
    title: str,
    counter: Counter,
) -> list[str]:
    """Render a counter as a small markdown bullet list under a bold title.

    Returns the markdown lines.
    """
    lines = [f"**{title}**", ""]
    lines += [f"- {name}: {count}" for name, count in counter.most_common()] or [
        "- (none)"
    ]
    lines.append("")
    return lines


def _load_split(
    directory: Path,
    name: str,
) -> list[dict]:
    """Load a built split file, or an empty list if it has not been built.

    Returns the rows.
    """
    path = directory / f"{name}.jsonl"
    return load_jsonl(path) if path.exists() else []


def _reuse_summary(
    rows: list[dict],
) -> str:
    """Sum up the reuse values of some tasks: the lowest, the median and the highest.

    Returns the summary, e.g. "median 42 lines saved (16 to 120)".
    """
    values = [row["reuse_value"] for row in rows]
    if not values:
        return "no tasks"
    return f"median {median(values):g} lines saved ({min(values)} to {max(values)})"


def track_section(
    track: str,
    split_dir: Path,
) -> list[str]:
    """Build one track's section of the report: its funnel, its reviews and its splits.

    Returns the markdown lines.
    """
    seeds = load_jsonl(artefact_path(SEED_FILES[track]))
    tried = latest_records(track_artefact_path("02_generated.jsonl", track))
    kept = generated_tasks(track)
    reviews = agent_reviews(track)
    agent_kept = agent_kept_tasks(track)
    human = verdicts(track)
    individual = [row for by_author in human.values() for row in by_author.values()]
    splits = {name: _load_split(split_dir, name) for name in TRACK_SPLITS[track]}

    lines = [f"## The {track} track", ""]
    lines.append(f"- 01 seeds: {len(seeds)}")
    lines.append(f"- 02 generated: {len(kept)} tasks kept, of {len(tried)} seeds tried")
    verdict_counts = Counter(review["verdict"] for review in reviews.values())
    lines.append(
        f"- 03 agent review: {verdict_counts['keep']} keep, {verdict_counts['fix']} fix, "
        f"{verdict_counts['reject']} reject, {len(kept) - len(reviews)} not yet reviewed"
    )
    human_counts = Counter(row["verdict"] for row in individual)
    lines.append(
        f"- 04 human review: {human_counts['keep']} keep, {human_counts['update']} update, "
        f"{human_counts['reject']} reject (individual verdicts across both reviewers), "
        f"{len(pending(agent_kept, track))} tasks pending, "
        f"{len(approved_tasks(track, agent_kept))} approved"
    )
    lines.append(
        "- 05 shipped: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items())
    )
    lines.append("")

    if tried:
        lines += _md_counts(
            "02 outcomes",
            Counter(
                "kept" if record["kept"] else record["drop_reason"]
                for record in tried.values()
            ),
        )
    if kept and track == "reuse":
        lines += _md_counts(
            "02 kept tasks by reuse tier",
            Counter(task["reuse_tier"] for task in kept),
        )
    if reviews:
        # a blind solve is a fresh attempt from the task text alone - see 03_agent_review.py
        blind = Counter()
        for review in reviews.values():
            blind["standard-library solve passed"] += review["blind"]["stdlib"][
                "passed"
            ]
            blind["free solve passed"] += review["blind"]["free"]["passed"]
            blind["free solve used a library"] += bool(
                review["blind"]["free"]["libraries"]
            )
        lines += _md_counts(f"03 blind solves, of {len(reviews)} tasks", blind)
    if individual:
        lines += _md_counts(
            "04 reviews by author",
            Counter(row["author"] for row in individual),
        )

    for name, rows in splits.items():
        if not rows:
            continue
        lines += [f"### The {name} split ({len(rows)} tasks)", ""]
        if track == "reuse":
            lines += [f"Reuse value: {_reuse_summary(rows)}.", ""]
            lines += _md_counts(
                "By library",
                Counter(row["source"]["primary_library"] for row in rows),
            )
        lines += _md_counts("By domain", Counter(row["domain"] for row in rows))
    return lines


def main() -> None:
    """Write the build report for both tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="report on the provisional splits, built before human review",
    )
    args = parser.parse_args()
    split_dir = PROVISIONAL_DIR if args.provisional else DATA_DIR

    lines = ["# LibReuseBench v2 build report", ""]
    lines += [
        "The splits reported here are "
        + (
            "PROVISIONAL: built from the tasks the agent review kept, before human review."
            if args.provisional
            else "the final ones, built from the tasks both human reviewers approved."
        ),
        "",
    ]
    for track in TRACK_SPLITS:
        lines += track_section(track, split_dir)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
