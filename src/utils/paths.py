"""Where training runs and evaluations write their results, under output/.

output/
    baselines/{model}/              eval results for each base model, before any training
    {base_model}/{run_id}/          everything from one training run of that base model:
        grpo/                       its training metrics (trainer_state.json, ...)
        epoch1/, epoch2/, ...       each epoch checkpoint's eval results
    {iteration}/                    a numbered round of experiments (01/, 02/, ...), written
        README.md                   up in its README - its runs are moved here from their
        {run_id}/                   base model's folder, in the same layout
    debug/                          debug (--debug) eval results, in the same layout as above

Eval results are two files per dataset (see src/eval/evaluate.py). The checkpoints
themselves are far larger, and live on project storage instead (see --output-root in
src/run_grpo.py).
"""

from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("output")


def debug_output_dir(output_dir: Path) -> Path:
    """
    The folder a debug (truncated) eval's results go in, kept apart from real results so
    they are never mistaken for them - a real eval skips datasets that already have a result
    file. Mirrors the normal layout under output/debug/.

    Returns output/debug/{the normal folder, relative to output/}.
    """
    return OUTPUT_DIR / "debug" / output_dir.relative_to(OUTPUT_DIR)


def baseline_dir(model: str) -> Path:
    """
    The folder a base model's eval results go in.

    Returns output/baselines/{model}.
    """
    return OUTPUT_DIR / "baselines" / model


def training_run_dir(
    base_model: str,
    run_id: str,
    iteration: str | None = None,
) -> Path:
    """
    The folder everything from one training run goes in: its training metrics (grpo/)
    and each epoch's eval results (epoch1/, epoch2/, ...). A new run is grouped under the
    base model it trained. Once it is written up as part of an iteration of experiments
    (an "iteration" property in its registry entries), it lives in that iteration's folder.

    Returns output/{iteration}/{run_id} for a run in an iteration, otherwise
    output/{base_model}/{run_id}.
    """
    if iteration:
        return OUTPUT_DIR / iteration / run_id
    return OUTPUT_DIR / base_model / run_id


def model_output_dir(
    model_key: str,
    model_settings: dict[str, Any],
) -> Path:
    """
    The folder a model's eval results go in, given its registry entry (see
    load_model_settings in src/utils/config.py): a base or api-served model (no
    checkpoint path) goes in its baseline folder, and a training run's epoch checkpoint
    in that run's folder, as epoch{n}/.

    Returns the folder path.
    """
    if not model_settings.get("path"):
        return baseline_dir(model_key)

    # the run id, base model and epoch are recorded when a checkpoint is registered (see
    # src/run_grpo.py). entries registered before that are recovered from their
    # "{run_id}-epoch{n}" key, and the "{base_model}-{suffix}" run id inside it
    properties = model_settings.get("properties") or {}
    key_run_id, _, key_epoch = model_key.rpartition("-epoch")
    run_id = properties.get("run_id", key_run_id)
    epoch = properties.get("epoch", key_epoch)
    base_model = properties.get("base_model", run_id.rpartition("-")[0])
    if not (run_id and epoch and base_model):
        raise ValueError(
            f"Can't tell which training run '{model_key}' belongs to - expected a "
            "'{run_id}-epoch{n}' key or run_id/base_model/epoch properties."
        )
    run_dir = training_run_dir(
        base_model=base_model,
        run_id=run_id,
        iteration=properties.get("iteration"),
    )
    return run_dir / f"epoch{epoch}"
