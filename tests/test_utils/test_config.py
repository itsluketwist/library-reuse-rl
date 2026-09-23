"""Tests for the config-loading helpers."""

import pytest

from src.utils.config import check_config_keys


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
