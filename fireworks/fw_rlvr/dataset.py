"""Convert the existing library-usage dataset into Fireworks RFT rows."""

from collections.abc import Iterable
from typing import Any


_REQUIRED_FIELDS = {
    "category_id",
    "local_modules",
    "prompt",
    "task_id",
    "task_type",
}


def prepare_record(
    record: dict[str, Any],
    system_prompt: str | None,
) -> dict[str, Any]:
    """Validate and convert one project row to the Fireworks prompt schema.

    Returns a prompt-only RFT row with reward metadata in `ground_truth`.
    """
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        task_id = record.get("task_id", "<unknown>")
        raise ValueError(f"Dataset row {task_id} is missing fields: {missing}")
    if record["task_type"] != "library_usage":
        raise ValueError(
            f"Expected task_type 'library_usage', got {record['task_type']!r}."
        )

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(record["prompt"])})

    # keep only the fields the structural evaluator requires.
    ground_truth = {
        "task_id": record["task_id"],
        "category_id": record["category_id"],
        "local_modules": list(record["local_modules"]),
        "phrasing_mode": record.get("phrasing_mode"),
        "dataset_version": record.get("dataset_version"),
    }
    return {"messages": messages, "ground_truth": ground_truth}


def prepare_records(
    records: Iterable[dict[str, Any]],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Convert project records while rejecting duplicate task identifiers.

    Returns Fireworks RFT rows in the same deterministic order as the input.
    """
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        task_id = str(record.get("task_id", ""))
        if not task_id:
            raise ValueError("Every dataset row must have a non-empty task_id.")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id in dataset: {task_id}")
        seen.add(task_id)
        prepared.append(prepare_record(record, system_prompt=system_prompt))
    return prepared


def representative_sample(
    records: list[dict[str, Any]],
    size: int,
) -> list[dict[str, Any]]:
    """Select a deterministic sample balanced across prompt phrasing modes.

    Returns at most `size` records without changing their contents.
    """
    if size < 1:
        raise ValueError("Sample size must be positive.")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        mode = str(record.get("ground_truth", {}).get("phrasing_mode") or "unknown")
        buckets.setdefault(mode, []).append(record)

    sample: list[dict[str, Any]] = []
    offsets = {mode: 0 for mode in buckets}
    while len(sample) < min(size, len(records)):
        added = False
        for mode in sorted(buckets):
            offset = offsets[mode]
            if offset >= len(buckets[mode]):
                continue
            sample.append(buckets[mode][offset])
            offsets[mode] += 1
            added = True
            if len(sample) == size:
                break
        if not added:
            break
    return sample
