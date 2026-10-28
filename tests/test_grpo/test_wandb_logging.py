"""Tests for keeping one wandb run across a training run's slurm jobs."""

import os

import pytest

from src.grpo.wandb_logging import WANDB_RUN_ID_FILE, resume_wandb_run


@pytest.fixture(autouse=True)
def _clean_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no wandb run settings, and leaves none behind."""
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)


def test_a_requeued_job_continues_the_same_wandb_run(tmp_path) -> None:
    """The first job saves its run id; a requeued job reads it back."""
    output_dir = tmp_path / "qwen3-8b-ab12"

    first_job = resume_wandb_run(output_dir)
    del os.environ["WANDB_RUN_ID"]
    requeued_job = resume_wandb_run(output_dir)

    assert requeued_job == first_job
    assert first_job.startswith("qwen3-8b-ab12-")
    assert (output_dir / WANDB_RUN_ID_FILE).read_text().strip() == first_job
    assert os.environ["WANDB_RUN_ID"] == first_job
    assert os.environ["WANDB_RESUME"] == "allow"


def test_a_fresh_output_directory_gets_a_new_wandb_run(tmp_path) -> None:
    """A debug run cleaned out and started again is a new wandb run."""
    first = resume_wandb_run(tmp_path / "run-a")
    second = resume_wandb_run(tmp_path / "run-b")

    assert first != second
