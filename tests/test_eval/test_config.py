"""Tests for experiment-critical evaluation and training configuration."""

from src.utils.config import load_full_yaml


def test_every_eval_dataset_has_an_explicit_evaluator() -> None:
    """Every configured dataset declares how it must be scored."""
    datasets = load_full_yaml("config/eval_datasets.yaml")

    assert datasets
    assert all({"path", "evaluator"} <= set(config) for config in datasets.values())
    assert datasets["libreusebench_test"]["evaluator"] == "libreusebench"


def test_every_decoding_profile_declares_its_thinking_mode() -> None:
    """Reasoning mode is a decoding-profile setting, not a hardcoded default - an eval
    silently switching it changes every result, so each profile must state it."""
    profiles = load_full_yaml("config/training.yaml")["eval_decoding_profiles"]

    assert profiles
    assert all(isinstance(p.get("enable_thinking"), bool) for p in profiles.values())


def test_training_uses_eight_epochs_and_sixteen_rollouts() -> None:
    """The main recipe retains experiment 01's epoch and rollout counts (8 epochs, as run
    8672 was still improving at epoch 6, and 16 rollouts, for more library contrast per
    group), with 32 prompts per optimizer step across the 4 gpus."""
    training = load_full_yaml("config/training.yaml")

    assert training["num_train_epochs"] == 8
    assert training["num_generations"] == 16
    completions_per_step = (
        training["per_device_train_batch_size"]
        * 4
        * training["gradient_accumulation_steps"]
    )
    assert completions_per_step // training["num_generations"] == 32


def test_training_scores_val_and_test_every_five_steps() -> None:
    """Reward generalisation is tracked every 5 steps, with 5 answers per prompt."""
    training = load_full_yaml("config/training.yaml")

    assert training["split_eval_steps"] == 5
    assert training["split_eval_samples"] == 5


def test_training_saves_a_checkpoint_every_epoch() -> None:
    """Each epoch is evaluated afterwards, so each epoch must leave a checkpoint."""
    training = load_full_yaml("config/training.yaml")

    assert training["save_strategy"] == "epoch"
    assert training["save_total_limit"] is None
    # for now only epoch checkpoints, and a fresh training process after each save, which
    # leaves gpu memory taken
    assert training["resume_save_steps"] is None
    assert training["restart_after_save"] is True


def test_training_keeps_master_weights_in_fp32() -> None:
    """At lr 1e-6 an update to a bf16 weight rounds to nothing, so full fine-tuning
    must hold the weights in fp32 and only compute in bf16."""
    training = load_full_yaml("config/training.yaml")

    assert training["model_init_kwargs"]["dtype"] == "float32"
    assert training["bf16"] is True
    assert training["learning_rate"] <= 1.0e-05
