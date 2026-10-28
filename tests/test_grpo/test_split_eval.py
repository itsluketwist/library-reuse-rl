"""Tests for the GPU-free parts of the in-training split eval."""

import pytest

from src.grpo.rewards import build_reward, load_reward_profile
from src.grpo.split_eval import (
    judge_completion,
    split_eval_entries,
    summarise_split_eval,
)


_EOS = 2

# the import-only reward: library credit for any import, used or not
_REWARD = build_reward(
    "library_usage",
    load_reward_profile("lu-import"),
    eos_token_id=_EOS,
)

_ROW = {"local_modules": [], "target_libs": ["numpy"]}


def _completion(code: str) -> str:
    """Wrap python code as a completion with reasoning and a fenced block."""
    return f"reasoning</think>\n```python\n{code}\n```"


def test_entries_cover_every_answer_once_across_gpus():
    """Every (row, sample) answer goes to exactly one gpu, in near-equal shares."""
    splits = {"val": [{"id": 0}, {"id": 1}], "test": [{"id": 2}]}

    shares = [
        split_eval_entries(splits, n_samples=5, process_index=index, num_processes=4)
        for index in range(4)
    ]

    assert sorted(len(share) for share in shares) == [3, 4, 4, 4]
    answers = [(split, row["id"]) for share in shares for split, row in share]
    assert sorted(answers) == sorted(
        [("val", 0)] * 5 + [("val", 1)] * 5 + [("test", 2)] * 5
    )


def test_judge_matches_the_training_reward():
    """An unused import earns full import-only reward, and its use is still recorded."""
    judgement = judge_completion(
        reward=_REWARD,
        split="val",
        row=_ROW,
        text=_completion("import numpy\nprint(sum([1, 2]))"),
        completion_ids=[5, 6, _EOS],
    )

    assert judgement["split"] == "val"
    assert judgement["external_lib_imported"] and not judgement["external_lib_used"]
    assert judgement["target_lib_imported"] and not judgement["target_lib_used"]
    assert judgement["reward"] == pytest.approx(2.0)
    assert (judgement["length"], judgement["truncated"]) == (3, False)


def test_judge_spots_a_cut_off_answer():
    """An answer that never reached the eos token was cut off, so loses correct_format."""
    judgement = judge_completion(
        reward=_REWARD,
        split="test",
        row=_ROW,
        text=_completion("import numpy\nnumpy.sum([1])"),
        completion_ids=[5, 6, 7],
    )

    assert judgement["truncated"]
    assert not judgement["correct_format"]


def test_summary_gives_each_split_its_own_rates():
    """Rates are per split, keyed for the eval logs."""
    judgements = [
        {
            "split": "val",
            "external_lib_used": True,
            "reward": 2.0,
            "truncated": False,
            "length": 10,
        },
        {
            "split": "val",
            "external_lib_used": False,
            "reward": 1.0,
            "truncated": True,
            "length": 30,
        },
        {
            "split": "test",
            "external_lib_used": False,
            "reward": 0.0,
            "truncated": False,
            "length": 5,
        },
    ]

    metrics = summarise_split_eval(judgements, components=["external_lib_used"])

    assert metrics == {
        "eval_test/n_completions": 1,
        "eval_test/external_lib_used": 0.0,
        "eval_test/reward": 0.0,
        "eval_test/clipped_ratio": 0.0,
        "eval_test/mean_length": 5.0,
        "eval_val/n_completions": 2,
        "eval_val/external_lib_used": 0.5,
        "eval_val/reward": 1.5,
        "eval_val/clipped_ratio": 0.5,
        "eval_val/mean_length": 20.0,
    }
