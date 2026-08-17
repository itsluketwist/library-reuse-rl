#!/usr/bin/env python3
"""Summarize an eval run's result JSON files into per-task comparison tables.

Reads every output/eval/{run_timestamp}/{model}/{dataset}-{decoding_profile}.json file
(see eval_results_path in src/run_eval.py) under a given run directory, and prints one
table per dataset task, comparing models and decoding profiles on that task's key metrics.
Also writes a combined summary.csv into the run directory.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# metrics to show per task (see src/eval/scoring.py for where these are computed);
# "n_total" is always included first
_TASK_METRICS = {
    "library_usage": [
        "valid_code_rate",
        "external_lib_used_rate",
        "appropriate_lib_rate",
    ],
    "bcbreuse": [
        "pass@1",
        "lib_use_rate",
    ],
}


def _load_results(run_dir: Path) -> list[dict]:
    """
    Load every result JSON under `run_dir` (output/eval/{run_timestamp}/{model}/*.json),
    pulling the model/dataset/decoding-profile identifiers and that task's key metrics
    out of each file's "metadata"/"analysis" blocks.

    Returns one row per result file, as a flat dict.
    """
    rows = []
    for path in sorted(run_dir.glob("*/*.json")):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        analysis = data["analysis"]
        dataset = metadata["dataset"]
        # filenames are "{dataset}-{decoding_profile}{suffix}.json" (see eval_results_path)
        decoding_profile = path.stem[len(dataset) + 1 :]

        row = {
            "model": path.parent.name,
            "dataset": dataset,
            "decoding_profile": decoding_profile,
            "n_total": analysis.get("n_total"),
        }
        if analysis.get("task") == "library_usage":
            row["valid_code_rate"] = analysis.get("valid_code_rate")
            row["external_lib_used_rate"] = analysis.get("external_lib_used_rate")
            row["appropriate_lib_rate"] = analysis.get("appropriate_lib_rate")
        elif analysis.get("task") == "bcbreuse":
            row["pass@1"] = analysis.get("pass@1")
            row["lib_use_rate"] = (
                analysis.get("library_use", {})
                .get("per_task", {})
                .get("using_lib_rate")
            )
        rows.append(row)
    return rows


def _print_task_table(rows: list[dict], dataset: str) -> None:
    """Print a model x decoding-profile comparison table for one dataset's rows."""
    subset = [r for r in rows if r["dataset"] == dataset]
    print(f"\n=== {dataset} ===")
    if not subset:
        print("(no results found)")
        return

    columns = ["n_total", *_TASK_METRICS.get(dataset, [])]
    table = pd.DataFrame(subset)[["model", "decoding_profile", *columns]]
    table = table.sort_values(["model", "decoding_profile"]).reset_index(drop=True)
    for column in columns:
        if column != "n_total":
            table[column] = table[column].map(
                lambda v: f"{v:.1%}" if v is not None else "-"
            )
    print(table.to_string(index=False))


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
        print(f"No result JSON files found under {args.run_dir}")
        return

    print(f"Eval summary: {args.run_dir}")
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        _print_task_table(rows, dataset)

    csv_path = args.run_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved combined summary to {csv_path}")


if __name__ == "__main__":
    main()
