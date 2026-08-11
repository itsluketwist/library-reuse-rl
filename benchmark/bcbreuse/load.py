"""Load the bundled BigCodeBenchReuse benchmark splits."""

from pathlib import Path

from llm_cgr import load_jsonl


# the split files are bundled alongside the package code (see pyproject package-data)
_DATA_DIR = Path(__file__).parent / "data"
_SPLITS = ("test", "val")


def load_dataset(
    split: str | None = None,
) -> dict[str, list[dict]]:
    """Load the benchmark splits from the bundled JSONL files.

    Pass a split name ("test" or "val") to load just one, or None for all available.

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
