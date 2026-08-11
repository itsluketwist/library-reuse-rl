"""Stage 01: fetch BigCodeBench, analyse each solution's libraries, and split the seed rows.

Downloads the pinned BigCodeBench split, keeps the fields later stages need, and classifies each
reference solution's imports (std vs external). The rows are written to two files: one for seeds
whose solution uses an external library (`01_bcb_extlibs.jsonl` - the pool the benchmark build
runs on, since those have a real library-reuse path) and one for stdlib-only seeds
(`01_bcb_stdlibs.jsonl` - kept aside in case we want a stdlib split later). This is the only stage
that touches the HuggingFace dataset.

Usage: python benchmark/pipeline/01_fetch_bcb.py
"""

import ast

from build import BCB_VERSION, analyse_libraries, artefact_path
from datasets import load_dataset

from src.utils.io import save_jsonl


# BCB's own id becomes our `seed_id` (the seed problem each benchmark task derives from); these
# other fields are carried forward, the rest of the BCB row is unused by the pipeline
_KEEP_FIELDS = (
    "complete_prompt",
    "instruct_prompt",
    "canonical_solution",
    "test",
    "entry_point",
)


def _parse_libs(
    raw: str | list[str],
) -> list[str]:
    """Normalise BCB's `libs` field (a stringified python list) into a list of names.

    Returns the library names, or an empty list if it cannot be parsed.
    """
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def main() -> None:
    """Download BigCodeBench, analyse each solution's libraries, and write the two split files."""
    dataset = load_dataset("bigcode/bigcodebench", split=BCB_VERSION)
    print(f"loaded BigCodeBench {BCB_VERSION}: {len(dataset)} problems")

    rows = []
    for row in dataset:
        record = (
            {"seed_id": row["task_id"]}
            | {field: row[field] for field in _KEEP_FIELDS}
            | {"libs": _parse_libs(row["libs"])}
        )
        # classify with the repo's own analyser so "external library" matches training
        std_libs, ext_libs = analyse_libraries(
            record["complete_prompt"], record["canonical_solution"]
        )
        rows.append(record | {"std_libs": std_libs, "ext_libs": ext_libs})

    # split by whether the reference solution reaches for an external library
    extlibs = [r for r in rows if r["ext_libs"]]
    stdlibs = [r for r in rows if not r["ext_libs"]]

    ext_path = artefact_path("01_bcb_extlibs.jsonl")
    std_path = artefact_path("01_bcb_stdlibs.jsonl")
    save_jsonl(extlibs, ext_path)
    save_jsonl(stdlibs, std_path)
    print(f"wrote {len(extlibs)} ext-lib rows -> {ext_path}")
    print(f"wrote {len(stdlibs)} stdlib-only rows -> {std_path}")


if __name__ == "__main__":
    main()
