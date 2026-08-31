"""CLI for adapting existing library-usage splits to Fireworks RFT JSONL."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FIREWORKS_ROOT = _REPOSITORY_ROOT / "fireworks"
for _path in (_REPOSITORY_ROOT, _FIREWORKS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fw_rlvr.dataset import prepare_records, representative_sample  # noqa: E402

from src.prompts import build_system_prompt  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON object records from one project split.

    Returns records in source order.
    """
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        records.append(value)
    return records


def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save records atomically as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset adapter CLI.

    Returns the configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/train/library_usage"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fireworks/rft/data"),
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated source splits to convert.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["base", "lib", "use", "std"],
        default="base",
    )
    parser.add_argument("--libraries-requirements-file", default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of validation rows copied to evaluator/development/sample.jsonl.",
    )
    return parser


def main() -> None:
    """Convert requested splits and build a small deterministic local-test sample."""
    args = build_parser().parse_args()
    system_prompt = build_system_prompt(
        args.libraries_requirements_file,
        style=args.prompt_style,
    )
    prepared_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        source_path = args.dataset_dir / f"{split}.jsonl"
        output_path = args.output_dir / f"{split}.jsonl"
        prepared = prepare_records(
            _load_jsonl(source_path),
            system_prompt=system_prompt,
        )
        _save_jsonl(prepared, output_path)
        prepared_by_split[split] = prepared
        print(f"Wrote {len(prepared)} rows to {output_path}.")

    sample_source = prepared_by_split.get("val")
    if sample_source is None:
        sample_source = prepare_records(
            _load_jsonl(args.dataset_dir / "val.jsonl"),
            system_prompt=system_prompt,
        )
    sample_path = (
        _REPOSITORY_ROOT
        / "fireworks"
        / "rft"
        / "evaluator"
        / "development"
        / "sample.jsonl"
    )
    sample = representative_sample(sample_source, size=args.sample_size)
    _save_jsonl(sample, sample_path)
    print(f"Wrote {len(sample)} local-test rows to {sample_path}.")


if __name__ == "__main__":
    main()
