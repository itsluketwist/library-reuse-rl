#!/usr/bin/env python3
"""Summarize eval result JSON files into per-task comparison tables.

Reads every base model's results (output/baselines/{model}/*.json) and every training
epoch's (output/{base_model}/{run_id}/epoch{n}/*.json) - see src/utils/paths.py - and
prints one table per dataset task, comparing models and prompt/decoding settings on that
task's key metrics. Also writes a combined summary.csv into the output directory.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.log import log
from src.utils.paths import OUTPUT_DIR


# metrics to show per task, as the path to each within the result's "analysis" block (see
# src/eval/scoring.py for where they are computed). "n_total" is always included first.
_TASK_METRICS: dict[str, dict[str, tuple[str, ...]]] = {
    "library_usage": {
        "valid_code_rate": ("valid_code_rate",),
        "external_lib_used_rate": ("external_lib_used_rate",),
        "appropriate_lib_rate": ("appropriate_lib_rate",),
    },
    "libreusebench": {
        "pass@1": ("pass@1",),
        "lib_used_rate": ("library_use", "used", "per_task", "using_lib_rate"),
    },
}


def _dig(
    analysis: dict[str, Any],
    path: tuple[str, ...],
) -> Any:
    """
    Follow a nested key path into an analysis block.

    Returns the value, or None if any step of the path is missing.
    """
    value: Any = analysis
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _result_files(output_dir: Path) -> list[tuple[str, Path]]:
    """
    Find every eval result file under `output_dir`, with the model it belongs to - a base
    model's folder name, or "{run_id}-epoch{n}" (its registry key) for a training epoch.

    Returns (model, path) pairs.
    """
    found = []
    for path in sorted(output_dir.glob("baselines/*/*.json")):
        found.append((path.parent.name, path))
    for path in sorted(output_dir.glob("*/*/epoch*/*.json")):
        found.append((f"{path.parent.parent.name}-{path.parent.name}", path))
    return found


def _load_results(output_dir: Path) -> list[dict]:
    """
    Load every result JSON under `output_dir` (see _result_files), pulling the run's
    identifiers and that task's metrics out of each file's "metadata"/"analysis" blocks.

    Returns one row per result file, as a flat dict.
    """
    rows = []
    for model, path in _result_files(output_dir):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        analysis = data["analysis"]
        dataset = metadata["dataset"]

        # filenames are "{dataset}-{decoding_profile}-{prompt_style}.json", with an
        # --adapter run appending its name to the style (see eval_results_path)
        decoding_profile, _, prompt_style = path.stem[len(dataset) + 1 :].partition("-")

        # metrics are keyed by the scorer's own task name, not the dataset name
        task = analysis.get("task")
        row = {
            "model": model,
            "dataset": dataset,
            "task": task,
            "decoding_profile": decoding_profile,
            "prompt_style": prompt_style,
            "n_total": analysis.get("n_total"),
        }
        for column, metric_path in _TASK_METRICS.get(task, {}).items():
            row[column] = _dig(analysis, metric_path)
        rows.append(row)
    return rows


def _format_rate(value: Any) -> str:
    """Format a rate as a percentage, or "-" when it is missing.

    Returns the formatted cell.
    """
    # pandas fills absent columns with NaN rather than None, and NaN is not None
    return "-" if value is None or pd.isna(value) else f"{value:.1%}"


def _print_task_table(rows: list[dict], dataset: str) -> None:
    """Print a comparison table for one dataset's rows."""
    subset = [row for row in rows if row["dataset"] == dataset]
    log(f"\n=== {dataset} ===")
    if not subset:
        log("(no results found)")
        return

    identifiers = ["model", "decoding_profile", "prompt_style"]
    rates = list(_TASK_METRICS.get(subset[0]["task"], {}))
    table = pd.DataFrame(subset)[[*identifiers, "n_total", *rates]]
    table = table.sort_values(identifiers).reset_index(drop=True)
    for column in rates:
        table[column] = table[column].map(_format_rate)
    log(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=OUTPUT_DIR,
        help="Output directory to summarize, in the layout described in "
        "src/utils/paths.py. Defaults to output/ (every model's results).",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        raise FileNotFoundError(
            f"{args.output_dir} does not exist or is not a directory"
        )

    rows = _load_results(args.output_dir)
    if not rows:
        log(f"No result JSON files found under {args.output_dir}")
        return

    log(f"Eval summary: {args.output_dir}")
    for dataset in sorted({row["dataset"] for row in rows}):
        _print_task_table(rows, dataset)

    csv_path = args.output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"\nSaved combined summary to {csv_path}")


if __name__ == "__main__":
    main()
