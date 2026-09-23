"""Tests for reading and tidying a training run's epoch checkpoints."""

import json
from pathlib import Path

from src.grpo.checkpoints import epoch_checkpoints, prune_resume_state


def _checkpoint(
    output_dir: Path,
    step: int,
    epoch: float | None = None,
) -> Path:
    """Build a checkpoint directory holding weights and a full resume state."""
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    # what evaluation needs
    (checkpoint / "model.safetensors").write_text("weights")
    (checkpoint / "config.json").write_text("{}")
    # what only resuming needs
    (checkpoint / "optimizer.bin").write_text("optimizer")
    (checkpoint / "scheduler.pt").write_text("scheduler")
    (checkpoint / "rng_state_0.pth").write_text("rng")
    if epoch is not None:
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"epoch": epoch, "global_step": step})
        )
    return checkpoint


class TestEpochCheckpoints:
    """epoch_checkpoints maps each epoch to the checkpoint it ended at."""

    def test_reads_the_epoch_recorded_in_each_checkpoint(self, tmp_path) -> None:
        """The epoch comes from the checkpoint's own state, not its position."""
        first = _checkpoint(tmp_path, step=28, epoch=1.0)
        second = _checkpoint(tmp_path, step=56, epoch=2.0)

        assert epoch_checkpoints(tmp_path) == {1: first, 2: second}

    def test_orders_by_epoch_not_by_directory_name(self, tmp_path) -> None:
        """Checkpoint names sort as text ("checkpoint-100" before "checkpoint-28")."""
        _checkpoint(tmp_path, step=28, epoch=1.0)
        _checkpoint(tmp_path, step=100, epoch=2.0)
        _checkpoint(tmp_path, step=140, epoch=3.0)

        assert list(epoch_checkpoints(tmp_path)) == [1, 2, 3]

    def test_ignores_a_checkpoint_without_a_recorded_epoch(self, tmp_path) -> None:
        """A half-written checkpoint is skipped rather than failing the run."""
        first = _checkpoint(tmp_path, step=28, epoch=1.0)
        _checkpoint(tmp_path, step=56, epoch=None)

        assert epoch_checkpoints(tmp_path) == {1: first}

    def test_no_checkpoints_is_not_an_error(self, tmp_path) -> None:
        """A run stopped by --max-steps saves nothing, and just registers nothing."""
        assert epoch_checkpoints(tmp_path) == {}


class TestPruneResumeState:
    """prune_resume_state keeps every epoch's weights but one resume state."""

    def test_older_checkpoints_keep_weights_and_lose_resume_state(
        self, tmp_path
    ) -> None:
        """Five copies of an 8b optimizer state would cost hundreds of gigabytes."""
        older = _checkpoint(tmp_path, step=28, epoch=1.0)
        newest = _checkpoint(tmp_path, step=56, epoch=2.0)

        prune_resume_state(tmp_path)

        # the older epoch is still evaluable
        assert (older / "model.safetensors").exists()
        assert (older / "config.json").exists()
        # but no longer resumable
        assert not (older / "optimizer.bin").exists()
        assert not (older / "scheduler.pt").exists()
        assert not (older / "rng_state_0.pth").exists()
        # the newest checkpoint is what a requeued job resumes from, so it keeps it all
        assert (newest / "optimizer.bin").exists()
        assert (newest / "rng_state_0.pth").exists()

    def test_removes_resume_state_saved_as_a_directory(self, tmp_path) -> None:
        """Some fsdp state dict types write the optimizer state as a directory."""
        older = _checkpoint(tmp_path, step=28, epoch=1.0)
        (older / "optimizer_0").mkdir()
        (older / "optimizer_0" / "shard.pt").write_text("shard")
        _checkpoint(tmp_path, step=56, epoch=2.0)

        prune_resume_state(tmp_path)

        assert not (older / "optimizer_0").exists()

    def test_a_single_checkpoint_is_left_alone(self, tmp_path) -> None:
        """After the first epoch there is nothing older to prune."""
        only = _checkpoint(tmp_path, step=28, epoch=1.0)

        prune_resume_state(tmp_path)

        assert (only / "optimizer.bin").exists()
