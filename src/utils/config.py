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
    section: str | None = None,
) -> dict[str, Any]:
    """Load a specific key from a yaml config file, optionally nested under a section.

    `section`, when given, is looked up first (e.g. "eval_decoding_profiles" in
    config/training.yaml, which shares that file with training's own top-level
    keys), and `key` is then looked up inside it rather than at the top level.

    Returns the config dict for the given key.
    Raises KeyError if the section or key is not found.
    """
    config = _load_yaml(config_file=config_file)

    if section is not None:
        if section not in config:
            raise KeyError(f"section '{section}' not found in {config_file}")
        config = config[section]

    if key in config:
        return config[key]

    where = f"section '{section}' of {config_file}" if section else config_file
    raise KeyError(f"key '{key}' not found in {where}")


def load_full_yaml(config_file: str) -> dict[str, Any]:
    """Load the full contents of a yaml config file.

    Returns the complete parsed dict.
    """
    return _load_yaml(config_file=config_file)


def register_model(
    model_key: str,
    base: str,
    path: str,
    properties: dict[str, Any],
    config_file: str,
) -> None:
    """
    Add a trained model entry to the model registry (config/models.yaml).

    Creates or overwrites the entry for `model_key`, recording the base model it was
    trained from (`base`), where the trained output (the lora adapter) was saved
    (`path`), and the training `properties` - the knobs the run was launched with (see
    src/run_grpo.py). The properties are stored because `model_key` is a short, non-
    descriptive run id, so they are what makes a registered model recognisable without
    having to reconstruct the run from its id. Uses an exclusive file lock so concurrent
    slurm jobs cannot corrupt the file by writing at the same time.
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
                "properties": properties,
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
