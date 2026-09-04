"""Score externally generated responses (e.g. from fireworks/) with the project's evaluators."""

from pathlib import Path
from typing import Any

from src.data import load_eval_dataset
from src.eval.scoring import score_dataset
from src.response.stop_reason import map_stop_reason
from src.utils.io import load_jsonl, save_json
from src.utils.log import log


# generation fields (see fireworks/fw_rlvr/inference.py's generation records) worth
# keeping alongside each scored result for provenance; excludes task_id/sample_index/
# content/finish_reason (already used for scoring) and the large raw sdk payloads
_GENERATION_RESULT_FIELDS = (
    "run_kind",
    "dataset",
    "requested_model",
    "response_model",
    "usage",
    "generation_config",
    "reasoning_content",
)


def _group_generations_by_sample(
    generations: list[dict[str, Any]],
    task_ids: set[str],
) -> dict[int, dict[str, dict[str, Any]]]:
    """Index generation records by sample_index then task_id, validating full coverage.

    Every sample_index present must carry exactly one generation for every task in
    `task_ids` (score_dataset needs each sample list to line up with the dataset).

    Returns {sample_index: {task_id: generation}}.
    """
    by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for generation in generations:
        try:
            task_id = str(generation["task_id"])
            sample_index = int(generation["sample_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Every generation needs task_id and sample_index."
            ) from error
        if task_id not in task_ids:
            raise ValueError(f"Generation has unknown task_id: {task_id}")
        sample = by_sample.setdefault(sample_index, {})
        if task_id in sample:
            raise ValueError(
                f"Duplicate generation for task {task_id}, sample {sample_index}."
            )
        sample[task_id] = generation

    if not by_sample:
        raise ValueError("Generations file contains no records.")
    for sample_index, present in by_sample.items():
        missing = task_ids - set(present)
        if missing:
            raise ValueError(
                f"Sample {sample_index} is missing generations for: {sorted(missing)}"
            )
    return by_sample


def score_generations(
    dataset_file: str | Path,
    generations_file: str | Path,
    output_file: str | Path,
    evaluator: str,
    python_executable: str | None = None,
    environment_root: str | None = None,
) -> dict[str, Any]:
    """
    Score a file of externally generated responses (see fireworks/fw_rlvr/inference.py)
    the same way evaluate_model scores vllm-generated ones, aligning them to `dataset_file`
    by task_id/sample_index rather than generating them in-process.

    `dataset_file` is loaded with load_eval_dataset (no system prompt - only its
    reward-reference columns matter for scoring), so it is normalised exactly as the
    rest of eval expects. `generations_file` is a JSONL file with at least task_id,
    sample_index, content, and finish_reason per record, one record per (task, sample).
    score_dataset (see src/eval/scoring.py) decides whether `evaluator` scores every
    sample or just the first, same as in-process evaluation.

    Returns and saves an evaluation envelope: {metadata, analysis, results}, matching
    evaluate_model's output shape (plus a "generation" block per result carrying
    _GENERATION_RESULT_FIELDS for provenance).
    """
    dataset = load_eval_dataset(dataset_file=dataset_file, system_prompt=None)
    task_ids = [str(row["task_id"]) for row in dataset]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Dataset contains duplicate task IDs: {dataset_file}")

    generations = load_jsonl(generations_file)
    by_sample = _group_generations_by_sample(generations, set(task_ids))
    sample_indices = sorted(by_sample)

    responses = [
        [
            (
                str(by_sample[sample_index][task_id].get("content") or ""),
                map_stop_reason(by_sample[sample_index][task_id].get("finish_reason")),
            )
            for sample_index in sample_indices
        ]
        for task_id in task_ids
    ]

    analysis, results = score_dataset(
        dataset=dataset,
        responses=responses,
        evaluator=evaluator,
        python_executable=python_executable,
        environment_root=environment_root,
    )

    for result in results:
        # multi-sample evaluators record which sample they scored as sample_idx; the
        # rest (score_dataset's single-sample path) only ever score the first
        sample_offset = result.get("sample_idx", 0)
        sample_index = sample_indices[sample_offset]
        generation = by_sample[sample_index][str(result["task_id"])]
        result["sample_index"] = sample_index
        result["generation"] = {
            field: generation[field]
            for field in _GENERATION_RESULT_FIELDS
            if field in generation
        }

    output = {
        "metadata": {
            "source": "external_generations",
            "dataset_file": str(dataset_file),
            "generations_file": str(generations_file),
            "evaluator": evaluator,
            "python_executable": python_executable,
            "environment_root": environment_root,
        },
        "analysis": analysis,
        "results": results,
    }
    save_json(output, output_file)
    log(f"Scored {len(results)} response(s) into {output_file}")
    return output
