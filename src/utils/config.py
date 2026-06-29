"""Configuration file loading utilities."""

import fcntl
import os
from typing import Any

import yaml


def _load_yaml(config_file: str) -> dict[str, Any]:
    """Load and parse a YAML config file.

    Returns the parsed dict, or an empty dict if the file is blank.
    """
    with open(file=config_file) as f:
        return yaml.safe_load(stream=f) or {}


def load_yaml_config(
    config_file: str,
    key: str,
) -> dict[str, Any]:
    """Load a specific key from a yaml config file.

    Returns the config dict for the given key.
    Raises KeyError if the key is not found.
    """
    config = _load_yaml(config_file=config_file)

    if key in config:
        return config[key]

    raise KeyError(f"key '{key}' not found in {config_file}")


def load_full_yaml(config_file: str) -> dict[str, Any]:
    """Load the full contents of a yaml config file.

    Returns the complete parsed dict.
    """
    return _load_yaml(config_file=config_file)


def register_model(
    model_key: str,
    base: str,
    path: str,
    config_file: str,
) -> None:
    """
    Add a trained model entry to the model registry (config/models.yaml).

    Creates or overwrites the entry for `model_key`, recording the base model it was
    trained from (`base`) and where the trained output was saved (`path`). Uses an
    exclusive file lock so concurrent slurm jobs cannot corrupt the file by writing at
    the same time.
    """
    # separate lock file, so we hold the lock without truncating the yaml itself
    lock_path = config_file + ".lock"

    with open(file=lock_path, mode="a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # block until we are the only writer
        try:
            # load existing config inside the lock so we see all prior writes
            config = _load_yaml(config_file=config_file)

            config[model_key] = {
                "base": base,
                "path": path,
            }

            # write to a temp file then rename, so readers never see a half-written file
            tmp_path = config_file + ".tmp"
            with open(file=tmp_path, mode="w") as f:
                yaml.dump(
                    data=config,
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            os.replace(src=tmp_path, dst=config_file)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    print(f"registered model '{model_key}' in {config_file}")
