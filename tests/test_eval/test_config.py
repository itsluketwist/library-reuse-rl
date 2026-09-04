"""Tests for experiment-critical evaluation and training configuration."""

from src.utils.config import load_full_yaml


def test_every_eval_dataset_has_an_explicit_evaluator() -> None:
    """Every configured dataset declares how it must be scored."""
    datasets = load_full_yaml("config/eval_datasets.yaml")

    assert datasets
    assert all({"path", "evaluator"} <= set(config) for config in datasets.values())
    assert datasets["libreusebench"]["evaluator"] == "libreusebench"


def test_every_decoding_profile_declares_its_thinking_mode() -> None:
    """Reasoning mode is a decoding-profile setting, not a hardcoded default - an eval
    silently switching it changes every result, so each profile must state it."""
    profiles = load_full_yaml("config/training.yaml")["eval_decoding_profiles"]

    assert profiles
    assert all(isinstance(p.get("enable_thinking"), bool) for p in profiles.values())


def test_training_uses_one_epoch_and_eight_rollouts() -> None:
    """The main recipe retains the experiment's epoch and rollout counts."""
    training = load_full_yaml("config/training.yaml")

    assert training["num_train_epochs"] == 1
    assert training["num_generations"] == 8
