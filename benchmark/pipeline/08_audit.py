"""Stage 08: audit the build and write benchmark/docs/REPORT.md.

Reads whatever stage artefacts exist and reports the drop funnel (how many problems survived each
stage and why the rest were dropped), the author review outcome, plus coverage of the final
dataset by category and by seed library. Purely a reporting stage - it changes no data.

Usage: python benchmark/pipeline/08_audit.py
"""

from collections import Counter

from build import DATA_DIR, artefact_path, latest_records
from review import validated_problems, verdicts

from src.utils.io import load_jsonl


REPORT_PATH = DATA_DIR.parent.parent / "docs" / "REPORT.md"


def _load(
    name: str,
) -> list[dict]:
    """Load a stage artefact, deduplicated to the latest record per seed (reruns append retries).

    Returns the rows (empty if the artefact does not exist yet).
    """
    return list(latest_records(artefact_path(name)).values())


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


def main() -> None:
    """Compute the funnel and coverage stats and write REPORT.md."""
    extlibs = _load("01_bcb_extlibs.jsonl")
    stdlibs = _load("01_bcb_stdlibs.jsonl")
    candidates = _load("02_candidates.jsonl")
    rewritten = _load("03_rewritten.jsonl")
    minted = _load("04_tests.jsonl")
    validated = _load("05_validated.jsonl")

    test_path = DATA_DIR / "test.jsonl"
    dev_path = DATA_DIR / "dev.jsonl"
    test = load_jsonl(test_path) if test_path.exists() else []
    dev = load_jsonl(dev_path) if dev_path.exists() else []
    final = test + dev

    lines = ["# LibReuseBench build report", ""]

    lines += ["## Drop funnel", ""]
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
    reviewed = verdicts()
    n_validated = len(validated_problems())
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

    if final:
        lines += _md_counts(
            "Final by category", Counter(r.get("category", "?") for r in final)
        )
        # seed_libs live under the shipped row's nested source provenance block
        seed_libs = Counter(
            lib for r in final for lib in r.get("source", {}).get("seed_libs", [])
        )
        lines += _md_counts("Final by seed library", seed_libs)

    REPORT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {REPORT_PATH} ({len(final)} final problems)")


if __name__ == "__main__":
    main()
