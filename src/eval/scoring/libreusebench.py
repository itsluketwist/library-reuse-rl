"""Score libreusebench eval responses: delegate to the official benchmark evaluator."""

import tempfile
from typing import Iterable

from libreusebench.evaluate import evaluate_responses as evaluate_lrb_responses


def score_libreusebench(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Run the complete official LibReuseBench evaluation workflow.

    Each row may carry multiple sampled responses (see eval_decoding_profiles'
    n_samples); every sample is submitted for the task, so the official evaluator's
    pass@k (see benchmark/libreusebench/evaluate.py's _pass_at_k) reflects them.

    The official evaluator discovers packages from the responses, installs them into
    its cached environment, executes the selected code, and aggregates the results.
    Eval runs are unattended (including on HPC/Slurm), so this passes `confirm=None`
    to install the discovered package plan automatically rather than prompting on stdin.

    Returns the official analysis and per-response results.
    """
    del python_executable
    rows = list(dataset)
    response_texts = {
        row["task_id"]: [response for response, _stop_reason in samples]
        for row, samples in zip(rows, responses)
    }
    stop_reasons = {
        (row["task_id"], sample_idx): stop_reason
        for row, samples in zip(rows, responses)
        for sample_idx, (_response, stop_reason) in enumerate(samples)
    }

    # the responses are already in memory, so they go straight to evaluate_responses;
    # only its own results file (which this discards, keeping just `official` below) and
    # a possible env-fixup file need somewhere to go, hence the still-disposable output_dir
    with tempfile.TemporaryDirectory(prefix="libreusebench-eval-") as temp_dir:
        official = evaluate_lrb_responses(
            responses=response_texts,
            output_dir=temp_dir,
            # without a cache dir the evaluator builds a disposable venv per run and
            # deletes it, so concurrently running eval jobs can never corrupt a shared one
            env_dir=environment_cache_dir,
            env_root=environment_root,
            confirm=None,
        )

    if not official:
        raise RuntimeError(
            "LibReuseBench could not prepare the generated responses' package "
            "environment; see the evaluator log above for failed packages."
        )

    # the official results number each task's samples as "idx" (0, 1, ...), which is
    # how they are matched back to the stop reason and full text of that sample
    results = official.pop("results")
    for result in results:
        result["stop_reason"] = stop_reasons[(result["task_id"], result["idx"])]
        result["response"] = response_texts[result["task_id"]][result["idx"]]

    return {
        "task": "libreusebench",
        "n_total": official["n_responses"],
        **official,
    }, results
