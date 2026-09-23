"""Reading and tidying the epoch checkpoints a training run leaves behind.

Kept apart from src/grpo/train_model.py so it imports nothing from the GPU stack, and can
therefore be tested (and read) in the core development environment.
"""

import shutil
from pathlib import Path

from src.utils.io import load_json


# what the trainer writes into a checkpoint purely so a run can resume. the weights,
# config and tokenizer alongside them are what evaluation needs, and are always kept.
_RESUME_ONLY_FILES = ("optimizer*", "scheduler*", "rng_state*")


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


def epoch_checkpoints(output_dir: str | Path) -> dict[int, Path]:
    """
    Find the checkpoint each epoch ended at (save_strategy is "epoch", see the recipe).

    Reads the epoch recorded inside each checkpoint's trainer_state.json instead of
    assuming checkpoints line up with epochs, so a run that was requeued and resumed
    still maps correctly.

    Returns each epoch number mapped to its checkpoint directory, in epoch order.
    """
    found: dict[int, Path] = {}
    for checkpoint in sorted_checkpoints(output_dir):
        state_file = checkpoint / "trainer_state.json"
        if not state_file.exists():
            continue
        epoch = load_json(state_file).get("epoch")
        if epoch is not None:
            # epoch is a float that lands on a whole number at an epoch-end save
            found[round(epoch)] = checkpoint
    return dict(sorted(found.items()))


def prune_resume_state(output_dir: str | Path) -> None:
    """
    Delete the resume state from every checkpoint except the newest one.

    Each epoch's weights are kept so every epoch can be evaluated, but only the newest
    checkpoint stays resumable - five copies of an 8b model's optimizer state would add
    hundreds of gigabytes for no benefit.
    """
    for checkpoint in sorted_checkpoints(output_dir)[:-1]:
        for pattern in _RESUME_ONLY_FILES:
            for path in checkpoint.glob(pattern):
                # fsdp writes some of these as directories rather than files
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
