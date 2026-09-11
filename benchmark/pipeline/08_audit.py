"""Stage 08: audit the build and write benchmark/docs/REPORT.md.

Reads whatever stage artefacts exist and reports the drop funnel (how many problems survived each
stage and why the rest were dropped), the author review outcome, plus coverage of the final
dataset by category and by seed library - for both the reuse track and the control track (the
no-reuse-expected pool sourced from BCB's stdlib-only seeds). Purely a reporting stage - it changes
no data.

Usage: python benchmark/pipeline/08_audit.py
"""

from collections import Counter

from build import DATA_DIR, latest_records, track_artefact_path
from review import validated_problems, verdicts

from src.utils.io import load_jsonl


REPORT_PATH = DATA_DIR.parent.parent / "docs" / "REPORT.md"


def _load(
    name: str,
    track: str = "reuse",
) -> list[dict]:
    """Load a stage artefact, deduplicated to the latest record per seed (reruns append retries).

    Returns the rows (empty if the artefact does not exist yet).
    """
    return list(latest_records(track_artefact_path(name, track)).values())


def _drop_reasons(
    rows: list[dict],
) -> Counter:
    """Count the drop reasons among rows that were not kept.

    Returns the reason counter.
    """
    return Counter(row.get("drop_reason", "?") for row in rows if not row.get("kept"))


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
    name: str,
) -> list[dict]:
    """Load a shipped split file from the data directory, or an empty list if not built yet.

    Returns the rows.
    """
    path = DATA_DIR / name
    return load_jsonl(path) if path.exists() else []


def _reuse_section() -> list[str]:
    """Build the "## Drop funnel" section for the reuse track.

    Returns the markdown lines.
    """
    extlibs = _load("01_bcb_extlibs.jsonl")
    stdlibs = _load("01_bcb_stdlibs.jsonl")
    candidates = _load("02_candidates.jsonl")
    rewritten = _load("03_rewritten.jsonl")
    minted = _load("04_tests.jsonl")
    validated = _load("05_validated.jsonl")

    test = _load_split("test.jsonl")
    dev = _load_split("dev.jsonl")
    final = test + dev

    lines = ["## Drop funnel (reuse track)", ""]
    lines.append(
        f"- 01 fetched: {len(extlibs)} ext-lib seeds (+ {len(stdlibs)} stdlib-only set aside)"
    )
    lines.append(
        f"- 02 candidates: {sum(1 for r in candidates if r['verdict'] in ('keep', 'maybe'))} "
        f"carried forward of {len(candidates)} annotated"
    )
    lines.append(
        f"- 03 rewritten (faithful): {sum(1 for r in rewritten if r.get('kept'))} of {len(rewritten)}"
    )
    lines.append(f"- 04 tests minted: {len(minted)}")
    lines.append(
        f"- 05 validated (dual-solvable): {sum(1 for r in validated if r.get('kept'))} of {len(validated)}"
    )

    # stage 06 is the human review pass over the validated problems
    reviewed = verdicts("reuse")
    n_validated = len(validated_problems("reuse"))
    counts = Counter(row.get("verdict", "?") for row in reviewed.values())
    n_pending = n_validated - len(reviewed)
    lines.append(
        f"- 06 reviewed: {counts['keep']} keep, {counts['update']} update, "
        f"{counts['reject']} reject, {n_pending} pending of {n_validated} "
        f"(shipping {counts['keep'] + counts['update']})"
    )
    lines.append(
        f"- **final dataset: {len(final)} problems** (test {len(test)}, dev {len(dev)})"
    )
    lines.append("")

    if candidates:
        lines += _md_counts("02 verdicts", Counter(r["verdict"] for r in candidates))
    if rewritten:
        lines += _md_counts("03 drop reasons", _drop_reasons(rewritten))
    if validated:
        lines += _md_counts("05 drop reasons", _drop_reasons(validated))
    if reviewed:
        lines += _md_counts(
            "06 review verdicts",
            Counter(row.get("verdict", "?") for row in reviewed.values()),
        )
        lines += _md_counts(
            "06 reviews by author",
            Counter(row.get("author") or "?" for row in reviewed.values()),
        )

    if final:
        lines += _md_counts(
            "Final by category", Counter(r.get("category", "?") for r in final)
        )
        # seed_libs live under the shipped row's nested source provenance block
        seed_libs = Counter(
            lib for r in final for lib in r.get("source", {}).get("seed_libs", [])
        )
        lines += _md_counts("Final by seed library", seed_libs)

    return lines


def _control_section() -> list[str]:
    """Build the "## Control track" section: the no-reuse-expected pool, no stage-05 equivalent.

    Returns the markdown lines.
    """
    stdlibs = _load("01_bcb_stdlibs.jsonl")
    candidates = _load("02_candidates.jsonl", "control")
    rewritten = _load("03_rewritten.jsonl", "control")
    minted = _load("04_tests.jsonl", "control")
    final = _load_split("control.jsonl")

    lines = ["## Control track", ""]
    lines.append(f"- 01 stdlib-only seeds: {len(stdlibs)}")
    lines.append(
        f"- 02 candidates: {sum(1 for r in candidates if r['verdict'] in ('keep', 'maybe'))} "
        f"carried forward of {len(candidates)} annotated"
    )
    lines.append(
        f"- 03 rewritten (faithful): {sum(1 for r in rewritten if r.get('kept'))} of {len(rewritten)}"
    )
    lines.append(
        f"- 04 tests minted (already the fully validated pool - no dual-solvable stage "
        f"for this track): {sum(1 for r in minted if r.get('kept'))} of {len(minted)}"
    )

    reviewed = verdicts("control")
    n_validated = len(validated_problems("control"))
    counts = Counter(row.get("verdict", "?") for row in reviewed.values())
    n_pending = n_validated - len(reviewed)
    lines.append(
        f"- 06 reviewed: {counts['keep']} keep, {counts['update']} update, "
        f"{counts['reject']} reject, {n_pending} pending of {n_validated} "
        f"(shipping {counts['keep'] + counts['update']})"
    )
    lines.append(f"- **final control dataset: {len(final)} problems**")
    lines.append("")

    if candidates:
        lines += _md_counts(
            "02 verdicts (control)", Counter(r["verdict"] for r in candidates)
        )
    if rewritten:
        lines += _md_counts("03 drop reasons (control)", _drop_reasons(rewritten))
    if reviewed:
        lines += _md_counts(
            "06 review verdicts (control)",
            Counter(row.get("verdict", "?") for row in reviewed.values()),
        )
        lines += _md_counts(
            "06 reviews by author (control)",
            Counter(row.get("author") or "?" for row in reviewed.values()),
        )
    if final:
        # every control seed's reference solution is stdlib-only, so seed-library coverage
        # would just be empty - only category coverage is informative here
        lines += _md_counts(
            "Control by category", Counter(r.get("category", "?") for r in final)
        )

    return lines


def main() -> None:
    """Compute the funnel and coverage stats for both tracks and write REPORT.md."""
    lines = ["# LibReuseBench build report", ""]
    lines += _reuse_section()
    lines += _control_section()

    REPORT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    n_final = len(_load_split("test.jsonl")) + len(_load_split("dev.jsonl"))
    n_control = len(_load_split("control.jsonl"))
    print(f"wrote {REPORT_PATH} ({n_final} reuse + {n_control} control final problems)")


if __name__ == "__main__":
    main()
