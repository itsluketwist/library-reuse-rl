"""Feed Fireworks generation JSONL into the existing project scorers."""

import json
from pathlib import Path
from typing import Any

from src.eval.scoring import MULTI_SAMPLE_EVALUATORS, score_dataset
from src.utils.io import save_json


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON object records from a JSONL file.

    Returns records in file order.
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


def _stop_reason(finish_reason: str | None) -> str:
    """Map Fireworks finish reasons to the project's result vocabulary.

    Returns the normalised stop reason.
    """
    if finish_reason is None:
        return "unknown"
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)


def _result_with_fireworks_metadata(
    result: dict[str, Any],
    generation: dict[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    """Attach stable generation provenance to one project scoring result.

    Returns the enriched result record.
    """
    return {
        **result,
        "sample_index": sample_index,
        "fireworks": {
            "run_kind": generation.get("run_kind"),
            "requested_model": generation.get("requested_model"),
            "response_model": generation.get("response_model"),
            "response_id": generation.get("response_id"),
            "usage": generation.get("usage"),
            "generation_config": generation.get("generation_config"),
            "reasoning_content": generation.get("reasoning_content"),
        },
    }


def score_generation_file(
    *,
    dataset_file: str | Path,
    generations_file: str | Path,
    output_file: str | Path,
    evaluator: str,
    python_executable: str | None = None,
    environment_dir: str | None = None,
) -> dict[str, Any]:
    """Align external generations by task ID and invoke existing scorers.

    Returns and saves an evaluation envelope containing per-sample analyses.
    """
    dataset_path = Path(dataset_file)
    generations_path = Path(generations_file)
    dataset = [
        {**row, "idx": index} for index, row in enumerate(_load_jsonl(dataset_path))
    ]
    rows_by_task = {str(row["task_id"]): row for row in dataset}
    if len(rows_by_task) != len(dataset):
        raise ValueError(f"Dataset contains duplicate task IDs: {dataset_path}")

    generations = _load_jsonl(generations_path)
    by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for generation in generations:
        try:
            task_id = str(generation["task_id"])
            sample_index = int(generation["sample_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Every generation needs task_id and sample_index."
            ) from error
        if task_id not in rows_by_task:
            raise ValueError(f"Generation has unknown task_id: {task_id}")
        sample = by_sample.setdefault(sample_index, {})
        if task_id in sample:
            raise ValueError(
                f"Duplicate generation for task {task_id}, sample {sample_index}."
            )
        sample[task_id] = generation

    expected = set(rows_by_task)
    if not by_sample:
        raise ValueError(f"Generation file contains no records: {generations_path}")
    analyses: dict[str, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    ordered_task_ids = [str(row["task_id"]) for row in dataset]
    sample_indices = sorted(by_sample)
    for sample_index in sample_indices:
        present = set(by_sample[sample_index])
        if present != expected:
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            raise ValueError(
                f"Sample {sample_index} does not match dataset; "
                f"missing={missing}, extra={extra}."
            )
    if evaluator in MULTI_SAMPLE_EVALUATORS:
        responses_by_task = [
            [
                (
                    str(by_sample[sample_index][task_id].get("content") or ""),
                    _stop_reason(by_sample[sample_index][task_id].get("finish_reason")),
                )
                for sample_index in sample_indices
            ]
            for task_id in ordered_task_ids
        ]
        analysis_output, results = score_dataset(
            dataset=dataset,
            responses=responses_by_task,
            evaluator=evaluator,
            python_executable=python_executable,
            environment_dir=environment_dir,
        )
        for result in results:
            task_id = str(result["task_id"])
            sample_offset = int(result["sample_idx"])
            sample_index = sample_indices[sample_offset]
            generation = by_sample[sample_index][task_id]
            all_results.append(
                _result_with_fireworks_metadata(
                    result,
                    generation,
                    sample_index,
                )
            )
    else:
        for sample_index in sample_indices:
            ordered_generations = [
                by_sample[sample_index][task_id] for task_id in ordered_task_ids
            ]
            # score_dataset now always takes one list of samples per row; each
            # sample_index here is its own independent trial, so that's a single-sample list
            responses = [
                [
                    (
                        str(generation.get("content") or ""),
                        _stop_reason(generation.get("finish_reason")),
                    )
                ]
                for generation in ordered_generations
            ]
            analysis, results = score_dataset(
                dataset=dataset,
                responses=responses,
                evaluator=evaluator,
                python_executable=python_executable,
                environment_dir=environment_dir,
            )
            analyses[str(sample_index)] = analysis
            for result, generation in zip(results, ordered_generations):
                all_results.append(
                    _result_with_fireworks_metadata(
                        result,
                        generation,
                        sample_index,
                    )
                )

        if len(analyses) == 1:
            analysis_output = next(iter(analyses.values()))
        else:
            analysis_output = {
                "task": evaluator,
                "n_samples": len(analyses),
                "by_sample": analyses,
            }
    output = {
        "metadata": {
            "source": "fireworks",
            "dataset_file": str(dataset_path),
            "generations_file": str(generations_path),
            "evaluator": evaluator,
            "python_executable": python_executable,
            "environment_dir": environment_dir,
        },
        "analysis": analysis_output,
        "results": all_results,
    }
    save_json(output, output_file)
    return output
