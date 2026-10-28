"""Reading and tidying the epoch checkpoints a training run leaves behind.

Kept apart from src/grpo/train_model.py so it imports nothing from the GPU stack, and can
therefore be tested (and read) in the core development environment.
"""

import shutil
from pathlib import Path

from src.utils.io import load_json


# what the trainer writes into a checkpoint purely so a run can resume: the optimizer, lr
# scheduler and random states, and fsdp's own copy of the weights (pytorch_model_fsdp.bin,
# read only to resume - evaluation loads model.safetensors). the safetensors weights, config
# and tokenizer alongside them are what evaluation needs, and are always kept.
_RESUME_ONLY_FILES = ("optimizer*", "scheduler*", "rng_state*", "pytorch_model_fsdp*")


def _checkpoint_step(checkpoint: Path) -> int:
    """
    Read the global step a checkpoint directory is named after.

    Returns the step number.
    """
    return int(checkpoint.name.split("-")[-1])


def sorted_checkpoints(output_dir: str | Path) -> list[Path]:
    """
    List a run's checkpoint directories, oldest first.

    Returns the checkpoint paths in step order.
    """
    return sorted(Path(output_dir).glob("checkpoint-*"), key=_checkpoint_step)


def last_resumable_checkpoint(output_dir: str | Path) -> Path | None:
    """
    Find the newest checkpoint a run can resume from.

    A checkpoint is only resumable once the trainer has finished saving it - its
    trainer_state.json is written last - and while it still holds its optimizer state
    (prune_resume_state removes that from all but the newest). A job killed partway through
    saving (e.g. at the time limit) leaves a newer, incomplete checkpoint, which is skipped.

    Returns the checkpoint directory, or None if there is nothing to resume from.
    """
    for checkpoint in reversed(sorted_checkpoints(output_dir)):
        saved = (checkpoint / "trainer_state.json").exists()
        has_optimizer = any(checkpoint.glob("optimizer*"))
        if saved and has_optimizer:
            return checkpoint
    return None


def epoch_checkpoints(output_dir: str | Path) -> dict[int, Path]:
    """
    Find the checkpoint each epoch ended at (save_strategy is "epoch", see the recipe).

    Reads the epoch recorded inside each checkpoint's trainer_state.json instead of
    assuming checkpoints line up with epochs, so a run that was requeued and resumed
    still maps correctly. A checkpoint saved partway through an epoch (the trainer saves
    one when --max-steps stops a run early) belongs to no epoch, so is skipped.

    Returns each epoch number mapped to its checkpoint directory, in epoch order.
    """
    found: dict[int, Path] = {}
    for checkpoint in sorted_checkpoints(output_dir):
        state_file = checkpoint / "trainer_state.json"
        if not state_file.exists():
            continue
        epoch = load_json(state_file).get("epoch")
        if epoch is None:
            continue
        # epoch is a float that lands on a whole number at an epoch-end save (within
        # float rounding - one step is ~1/37 of an epoch, far larger than the tolerance)
        whole_epoch = round(epoch)
        if whole_epoch >= 1 and abs(epoch - whole_epoch) < 1e-3:
            found[whole_epoch] = checkpoint
    return dict(sorted(found.items()))


def prune_resume_state(output_dir: str | Path) -> None:
    """
    Tidy a run's checkpoints after a save, so only the newest one stays resumable.

    The newest complete checkpoint (the one just saved) is kept whole, since a requeued job
    resumes from it. Every older epoch checkpoint keeps its weights, so each epoch can still
    be evaluated, but loses its resume state - eight copies of an 8b model's optimizer state
    would add hundreds of gigabytes. Every older checkpoint that isn't an epoch's is deleted
    outright: the extra ones saved partway through an epoch only so a requeue loses less work
    (see resume_save_steps in the recipe), and any a job was killed partway through saving.
    """
    checkpoints = sorted_checkpoints(output_dir)
    complete = [
        checkpoint
        for checkpoint in checkpoints
        if (checkpoint / "trainer_state.json").exists()
    ]
    if not complete:
        return
    newest_step = _checkpoint_step(complete[-1])
    epoch_ends = set(epoch_checkpoints(output_dir).values())

    for checkpoint in checkpoints:
        # the newest complete checkpoint is kept whole. a half-saved one from a later step
        # (a job killed while saving it) is a step the run will reach and save again, over it
        if _checkpoint_step(checkpoint) >= newest_step:
            continue
        if checkpoint not in epoch_ends:
            shutil.rmtree(checkpoint)
            continue
        for pattern in _RESUME_ONLY_FILES:
            for path in checkpoint.glob(pattern):
                # fsdp writes some of these as directories rather than files
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
