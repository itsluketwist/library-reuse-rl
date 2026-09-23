"""Tests for the config-loading helpers."""

import pytest
import yaml

from src.utils.config import check_config_keys, register_model, unregister_models


class TestCheckConfigKeys:
    """check_config_keys turns silently ignored settings into a startup failure."""

    def test_known_keys_pass(self) -> None:
        check_config_keys(
            config={"learning_rate": 1e-6, "beta": 0.01},
            known={"learning_rate", "beta", "seed"},
            source="config/training.yaml",
        )

    def test_a_renamed_setting_is_named_in_the_error(self) -> None:
        """A recipe key nothing reads means the run quietly used the default instead -
        transformers 5 renamed warmup_ratio to warmup_steps, for example."""
        with pytest.raises(ValueError, match="warmup_ratio"):
            check_config_keys(
                config={"learning_rate": 1e-6, "warmup_ratio": 0.02},
                known={"learning_rate", "warmup_steps"},
                source="config/training.yaml",
            )


class TestUnregisterModels:
    """unregister_models removes the trained entries debug runs leave behind."""

    @staticmethod
    def _registry(tmp_path) -> str:
        """A registry holding a base model plus two debug runs and one real run."""
        config_file = str(tmp_path / "models.yaml")
        with open(config_file, "w") as f:
            yaml.safe_dump(
                {"qwen3-debug": {"base": "Qwen/Qwen3-0.6B", "path": None}}, f
            )
        for model_key in (
            "qwen3-debug-smoke-epoch1",
            "qwen3-8b-scale-epoch0",
            "qwen3-8b-a1b2-epoch1",
        ):
            register_model(
                model_key=model_key,
                base="Qwen/Qwen3-8B",
                path=f"/checkpoints/{model_key}",
                properties={},
                config_file=config_file,
            )
        return config_file

    def test_removes_only_matching_trained_entries(self, tmp_path) -> None:
        config_file = self._registry(tmp_path)

        removed = unregister_models(
            prefixes=["qwen3-debug-", "qwen3-8b-scale-"],
            config_file=config_file,
        )

        assert removed == ["qwen3-debug-smoke-epoch1", "qwen3-8b-scale-epoch0"]
        with open(config_file) as f:
            remaining = yaml.safe_load(f)
        # the base model and the real run are untouched
        assert set(remaining) == {"qwen3-debug", "qwen3-8b-a1b2-epoch1"}

    def test_never_removes_a_base_model(self, tmp_path) -> None:
        """A base model has no weights path, even if its name matches a prefix."""
        config_file = self._registry(tmp_path)

        removed = unregister_models(prefixes=["qwen3-debug"], config_file=config_file)

        assert "qwen3-debug" not in removed
