"""Tests for conversion of project rows to Fireworks RFT rows."""

import pytest
from fw_rlvr.dataset import prepare_records, representative_sample


def _record(task_id: str, phrasing_mode: str = "bare") -> dict:
    """Build a minimal valid source record.

    Returns a library-usage row for adapter tests.
    """
    return {
        "task_id": task_id,
        "prompt": "Write Python code.",
        "task_type": "library_usage",
        "category_id": "tables",
        "local_modules": ["helpers"],
        "useful_libs": ["pandas"],
        "phrasing_mode": phrasing_mode,
        "dataset_version": "test",
    }


def test_prepare_records_preserves_prompt_and_reward_metadata() -> None:
    """The adapter keeps source prompts and only adds required RFT structure."""
    prepared = prepare_records(
        [_record("task-1")],
        system_prompt="Prefer reusable libraries.",
    )

    assert prepared[0]["messages"] == [
        {"role": "system", "content": "Prefer reusable libraries."},
        {"role": "user", "content": "Write Python code."},
    ]
    assert prepared[0]["ground_truth"]["task_id"] == "task-1"
    assert prepared[0]["ground_truth"]["local_modules"] == ["helpers"]
    assert "useful_libs" not in prepared[0]["ground_truth"]
    assert "solution" not in prepared[0]


def test_prepare_records_rejects_duplicates_and_wrong_task_type() -> None:
    """Invalid source data fails before any remote upload can occur."""
    with pytest.raises(ValueError, match="Duplicate task_id"):
        prepare_records([_record("same"), _record("same")], system_prompt=None)

    wrong = _record("wrong")
    wrong["task_type"] = "product_choice"
    with pytest.raises(ValueError, match="Expected task_type"):
        prepare_records([wrong], system_prompt=None)


def test_representative_sample_round_robins_phrasing_modes() -> None:
    """The development sample contains both dataset prompt variants."""
    records = prepare_records(
        [
            _record("bare-1", "bare"),
            _record("bare-2", "bare"),
            _record("repo-1", "repo"),
            _record("repo-2", "repo"),
        ],
        system_prompt=None,
    )

    sample = representative_sample(records, size=3)

    assert [row["ground_truth"]["task_id"] for row in sample] == [
        "bare-1",
        "repo-1",
        "bare-2",
    ]
