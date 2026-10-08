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


def test_training_uses_three_epochs_and_eight_rollouts() -> None:
    """The main recipe retains the experiment's epoch and rollout counts (3 epochs since run
    4796's gains plateaued after epoch 2)."""
    training = load_full_yaml("config/training.yaml")

    assert training["num_train_epochs"] == 3
    assert training["num_generations"] == 8


def test_training_saves_a_checkpoint_every_epoch() -> None:
    """Each epoch is evaluated afterwards, so each epoch must leave a checkpoint."""
    training = load_full_yaml("config/training.yaml")

    assert training["save_strategy"] == "epoch"
    assert training["save_total_limit"] is None


def test_training_keeps_master_weights_in_fp32() -> None:
    """At lr 1e-6 an update to a bf16 weight rounds to nothing, so full fine-tuning
    must hold the weights in fp32 and only compute in bf16."""
    training = load_full_yaml("config/training.yaml")

    assert training["model_init_kwargs"]["dtype"] == "float32"
    assert training["bf16"] is True
    assert training["learning_rate"] <= 1.0e-05
