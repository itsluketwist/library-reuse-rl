"""Tests for the config-loading helpers."""

import pytest
import yaml

from src.utils.config import check_config_keys, load_model_settings, register_model


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


class TestModelRegistry:
    """The registry is split across base, trained and debug files, read as one."""

    @staticmethod
    def _registry_files(tmp_path) -> tuple[str, str, str]:
        """A hand-written base file, plus trained and debug files no run has made yet."""
        base_file = str(tmp_path / "models.yaml")
        with open(base_file, "w") as f:
            yaml.safe_dump(
                {"qwen3-8b": {"base": "Qwen/Qwen3-8B", "path": None}},
                f,
            )
        return (
            base_file,
            str(tmp_path / "trained_models.yaml"),
            str(tmp_path / "debug_models.yaml"),
        )

    def test_register_model_creates_a_missing_registry_file(self, tmp_path) -> None:
        """The first run to register creates the trained (or debug) file."""
        _, trained_file, _ = self._registry_files(tmp_path)

        register_model(
            model_key="qwen3-8b-a1b2-epoch1",
            base="Qwen/Qwen3-8B",
            path="/checkpoints/qwen3-8b-a1b2/checkpoint-28",
            properties={"epoch": 1},
            config_file=trained_file,
        )

        with open(trained_file) as f:
            entry = yaml.safe_load(f)["qwen3-8b-a1b2-epoch1"]
        assert entry["path"] == "/checkpoints/qwen3-8b-a1b2/checkpoint-28"

    def test_looks_up_base_and_trained_models_alike(self, tmp_path) -> None:
        """Eval names a base model or a trained epoch the same way."""
        registry_files = self._registry_files(tmp_path)
        register_model(
            model_key="qwen3-8b-a1b2-epoch1",
            base="Qwen/Qwen3-8B",
            path="/checkpoints/qwen3-8b-a1b2/checkpoint-28",
            properties={},
            config_file=registry_files[1],
        )

        base = load_model_settings("qwen3-8b", registry_files=registry_files)
        trained = load_model_settings(
            "qwen3-8b-a1b2-epoch1",
            registry_files=registry_files,
        )

        assert base["path"] is None
        assert trained["base"] == "Qwen/Qwen3-8B"

    def test_missing_registry_files_are_skipped(self, tmp_path) -> None:
        """Before any run, only the base file exists - lookups still work."""
        registry_files = self._registry_files(tmp_path)

        assert load_model_settings("qwen3-8b", registry_files=registry_files)

    def test_an_unknown_model_raises(self, tmp_path) -> None:
        registry_files = self._registry_files(tmp_path)

        with pytest.raises(KeyError, match="qwen3-typo"):
            load_model_settings("qwen3-typo", registry_files=registry_files)
