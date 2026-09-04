#!/usr/bin/env python3
"""Summarize an eval run's result JSON files into per-task comparison tables.

Reads every output/eval/{run_timestamp}/{model}/*.json file (see eval_results_path in
src/run_eval.py) under a given run directory, and prints one table per dataset task,
comparing models and prompt/decoding settings on that task's key metrics. Also writes a
combined summary.csv into the run directory.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.log import log


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


def _load_results(run_dir: Path) -> list[dict]:
    """
    Load every result JSON under `run_dir` (output/eval/{run_timestamp}/{model}/*.json),
    pulling the run's identifiers and that task's metrics out of each file's
    "metadata"/"analysis" blocks.

    Returns one row per result file, as a flat dict.
    """
    rows = []
    for path in sorted(run_dir.glob("*/*.json")):
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
            "model": path.parent.name,
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
        "run_dir",
        type=Path,
        help="output/eval/{run_timestamp} directory to summarize",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"{args.run_dir} does not exist or is not a directory")

    rows = _load_results(args.run_dir)
    if not rows:
        log(f"No result JSON files found under {args.run_dir}")
        return

    log(f"Eval summary: {args.run_dir}")
    for dataset in sorted({row["dataset"] for row in rows}):
        _print_task_table(rows, dataset)

    csv_path = args.run_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"\nSaved combined summary to {csv_path}")


if __name__ == "__main__":
    main()
