"""Configuration file loading utilities."""

import fcntl
import os
from collections.abc import Callable
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


def check_config_keys(
    config: dict[str, Any],
    known: set[str],
    source: str,
) -> None:
    """Check every key in `config` is a setting something actually reads.

    Parsers that accept extra keys (TrlParser's allow_extra_keys) drop unrecognised ones
    silently, so a renamed or misspelled setting would leave the run quietly using the
    default instead - transformers 5, for instance, renamed warmup_ratio to warmup_steps.

    Raises ValueError naming any key that is not in `known`.
    """
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            f"{source} contains setting(s) nothing reads: {unknown}. They would be "
            "ignored silently - remove them, or correct the name."
        )


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

    def add_entry(config: dict[str, Any]) -> None:
        config[model_key] = {
            "base": base,
            "path": path,
            "properties": properties,
        }

    _update_model_registry(config_file=config_file, update=add_entry)


def unregister_models(
    prefixes: list[str],
    config_file: str,
) -> list[str]:
    """
    Remove every trained model whose key starts with one of `prefixes` from the model
    registry (config/models.yaml), e.g. the entries debug runs leave behind.

    Only trained entries (those with a weights path) are removed, so a base model whose
    name happens to match a prefix is never touched.

    Returns the keys that were removed.
    """
    removed: list[str] = []

    def remove_entries(config: dict[str, Any]) -> None:
        for model_key in list(config):
            is_trained = bool((config[model_key] or {}).get("path"))
            if is_trained and model_key.startswith(tuple(prefixes)):
                del config[model_key]
                removed.append(model_key)

    _update_model_registry(config_file=config_file, update=remove_entries)
    return removed


def _update_model_registry(
    config_file: str,
    update: Callable[[dict[str, Any]], None],
) -> None:
    """
    Apply `update` to the model registry and write it back, holding an exclusive file
    lock throughout so concurrent slurm jobs cannot corrupt the file.
    """
    # separate lock file, so we hold the lock without truncating the yaml itself
    lock_path = config_file + ".lock"

    with open(file=lock_path, mode="a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # block until we are the only writer
        try:
            # load existing config inside the lock so we see all prior writes
            config = _load_yaml(config_file=config_file)
            update(config)

            # write to a temp file then rename, so readers never see a half-written file
            tmp_path = config_file + ".tmp"
            with open(file=tmp_path, mode="w") as f:
                f.write(_dump_model_registry(config))
            os.replace(src=tmp_path, dst=config_file)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
