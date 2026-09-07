"""Load the bundled LibReuseBench benchmark splits, and records from external files."""

import json
from pathlib import Path
from typing import Any

from llm_cgr import load_jsonl


# the split files are bundled alongside the package code (see pyproject package-data)
_DATA_DIR = Path(__file__).parent / "data"
_SPLITS = ("test", "dev", "control")


def load_records(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load a list of records from a .json (a JSON array) or .jsonl (one object per line) file.

    This is the shared, format-flexible primitive external results/response files are read
    with elsewhere in the repo - callers that need a particular record schema (e.g. keyed by
    task id, single vs multiple responses) normalise these raw records themselves.

    Returns the list of records, in file order.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_jsonl(str(path))
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(
        f"Unsupported file extension '{path.suffix}' (expected .json or .jsonl)"
    )


def load_dataset(
    split: str | None = None,
) -> dict[str, list[dict]]:
    """Load the benchmark splits from the bundled JSONL files.

    Pass a split name ("test", "dev", or "control") to load just one, or None for all available.
    "control" is a no-reuse-expected counterpart to "test"/"dev": BigCodeBench seeds whose
    reference solution used no external library, so a good solution is not expected to reach for
    one either.

    Returns a dict mapping split name to its list of task records.
    """
    requested = [split] if split is not None else list(_SPLITS)
    invalid = set(requested) - set(_SPLITS)
    if invalid:
        raise ValueError(f"invalid split(s) {sorted(invalid)}; valid: {list(_SPLITS)}")

    dataset: dict[str, list[dict]] = {}
    for name in requested:
        path = _DATA_DIR / f"{name}.jsonl"
        if path.exists():
            dataset[name] = load_jsonl(str(path))
    return dataset
