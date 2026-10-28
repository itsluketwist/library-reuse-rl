"""Scoring the val and test splits during training, to track how the reward generalises.

Every few optimizer steps, the model answers each val and test prompt a few times, and each
answer is judged with the run's own reward components (see SplitEvalCallback in
src/grpo/train_model.py, which does the generating). This module holds the parts that need
no GPU - which answers each GPU generates, how one answer is judged, and how the judgements
become logged rates - so they can be tested (and read) in the core development environment,
as src/grpo/checkpoints.py is.
"""

from typing import Any

from src.grpo.rewards import LibraryReward
from src.response.stop_reason import stop_reason_from_tokens


def split_eval_entries(
    splits: dict[str, list[dict[str, Any]]],
    n_samples: int,
    process_index: int,
    num_processes: int,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Choose the answers this GPU generates in one round: its share of every (split, row)
    pair, each repeated `n_samples` times.

    The answers are dealt out in turn, so the GPUs get near-equal shares and every GPU
    gets at least one whenever there are at least as many answers as GPUs.

    Returns (split name, dataset row) pairs, one per answer to generate.
    """
    entries = [
        (split, row)
        for split, rows in splits.items()
        for row in rows
        for _ in range(n_samples)
    ]
    return entries[process_index::num_processes]


def judge_completion(
    reward: LibraryReward,
    split: str,
    row: dict[str, Any],
    text: str,
    completion_ids: list[int],
) -> dict[str, Any]:
    """
    Judge one generated answer exactly as the training reward would, component by
    component.

    The row gets the completion's token ids, as trl's reward functions do, so the stop
    reason (and with it correct_format) is worked out the same way.

    Returns the split, each component's verdict, the reward under the run's own weights,
    the answer's length in tokens, and whether it was cut off at the length limit.
    """
    components = reward.components(text, {**row, "completion_ids": completion_ids})
    return {
        "split": split,
        **components,
        "reward": sum(
            reward.weights.get(name, 0.0)
            for name, satisfied in components.items()
            if satisfied
        ),
        "length": len(completion_ids),
        "truncated": stop_reason_from_tokens(completion_ids, reward.eos_token_id)
        == "length",
    }


def summarise_split_eval(
    judgements: list[dict[str, Any]],
    components: list[str],
) -> dict[str, float]:
    """
    Turn one round's judgements (from every GPU) into the metrics to log.

    Each split gets its own keys, e.g. "eval_val/external_lib_used" is the share of val
    answers that used an external library. The "eval_" prefix is what transformers' wandb
    logging files under "eval/", and what src/grpo/train_model.py collects into
    eval_results.json.

    Returns the metrics, keyed "eval_{split}/{metric}".
    """
    metrics: dict[str, float] = {}
    for split in sorted({judgement["split"] for judgement in judgements}):
        rows = [judgement for judgement in judgements if judgement["split"] == split]
        n = len(rows)
        prefix = f"eval_{split}"
        metrics[f"{prefix}/n_completions"] = n
        # each component's rate - the same rates the training logs per step
        for component in components:
            metrics[f"{prefix}/{component}"] = sum(row[component] for row in rows) / n
        metrics[f"{prefix}/reward"] = sum(row["reward"] for row in rows) / n
        metrics[f"{prefix}/clipped_ratio"] = sum(row["truncated"] for row in rows) / n
        metrics[f"{prefix}/mean_length"] = sum(row["length"] for row in rows) / n
    return metrics
