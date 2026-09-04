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


# re-added at the top of the registry each rewrite, since yaml.dump drops comments - so
# anything worth keeping about the file's format has to live here, not in the yaml itself
_MODEL_REGISTRY_HEADER = """\
# model registry - base models we start from, plus any models trained from them.
# "path" is blank for base models, and filled in (by register_model) once a model
# has actually been trained, pointing at that run's lora adapter. trained entries also
# carry a "properties" block recording the knobs the run was launched with, since the
# run-id key is non-descriptive - see src/utils/config.py and src/run_grpo.py
#
# entries default to "provider: vllm" (local weights). "provider: api" entries are served
# over an OpenAI-compatible api instead, and carry "api_key_env" (the .env variable holding
# the key) plus optional "base_url", "extra_params" and "max_tokens_param" - see the README
# and src/eval/generate_api.py"""


class _RegistryDumper(yaml.SafeDumper):
    """yaml dumper that writes None as a blank value (e.g. `path:`) rather than `null`,
    matching the hand-written base-model entries in the registry."""


_RegistryDumper.add_representer(
    type(None),
    lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)


def _dump_model_registry(config: dict[str, Any]) -> str:
    """Serialise the model registry: the header comment, then one entry per model with a
    blank line between them (yaml.dump alone drops comments and packs entries together).

    Returns the file contents as a string.
    """
    entries = [
        yaml.dump(
            {model_key: entry},
            Dumper=_RegistryDumper,
            default_flow_style=False,
            sort_keys=False,
        )
        for model_key, entry in config.items()
    ]
    return _MODEL_REGISTRY_HEADER + "\n\n" + "\n".join(entries)


def register_model(
    model_key: str,
    base: str,
    path: str,
    properties: dict[str, Any],
    config_file: str,
) -> None:
    """
    Add a trained model entry to the model registry (config/models.yaml).

    Creates or overwrites the entry for `model_key`, recording the `base` it was trained
    from, the adapter `path`, and the training `properties` (the knobs the run was
    launched with) - stored because `model_key` is a non-descriptive run id, so they are
    what make the model recognisable later. Uses an exclusive file lock so concurrent
    slurm jobs cannot corrupt the file.
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
                f.write(_dump_model_registry(config))
            os.replace(src=tmp_path, dst=config_file)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    print(f"registered model '{model_key}' in {config_file}")
