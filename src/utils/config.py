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


# the model registry is split so each file has a single owner, and a model is looked up
# in all three (see load_model_settings):
# - models.yaml holds the base and api-served models, written by hand - code never
#   rewrites it, so its comments and formatting survive
# - trained_models.yaml holds every epoch of every real training run, written by
#   register_model
# - debug_models.yaml holds the same for debug runs. it is git-ignored, and
#   `bash scripts/submit_debug.sh clean` simply deletes it
BASE_MODELS_FILE = "config/models.yaml"
TRAINED_MODELS_FILE = "config/trained_models.yaml"
DEBUG_MODELS_FILE = "config/debug_models.yaml"
MODEL_REGISTRY_FILES = (BASE_MODELS_FILE, TRAINED_MODELS_FILE, DEBUG_MODELS_FILE)


def load_model_settings(
    model_key: str,
    registry_files: tuple[str, ...] = MODEL_REGISTRY_FILES,
) -> dict[str, Any]:
    """
    Look a model up across the model registry files (base, trained and debug models).

    A registry file that does not exist yet - no run has registered anything - is
    skipped.

    Returns the model's registry entry.
    Raises KeyError if no registry file holds `model_key`.
    """
    for config_file in registry_files:
        if not os.path.exists(config_file):
            continue
        config = _load_yaml(config_file=config_file)
        if model_key in config:
            return config[model_key]
    raise KeyError(f"model '{model_key}' not found in any of {list(registry_files)}")


# re-added at the top of a trained-model registry each rewrite, since yaml.dump drops
# comments - so anything worth keeping about the file has to live here, not in the yaml
_TRAINED_REGISTRY_HEADER = """\
# trained models - one entry per epoch checkpoint of each training run, keyed
# "{run_id}-epoch{n}". written by register_model (src/utils/config.py), which rewrites
# this whole file on every registration, so hand edits to comments will not survive.
# "path" points at the checkpoint, and "properties" records the knobs the run was
# launched with, since the run-id key is non-descriptive. base models, and the format
# every entry shares, are described in config/models.yaml"""


class _RegistryDumper(yaml.SafeDumper):
    """yaml dumper that writes None as a blank value (e.g. `path:`) rather than `null`,
    matching the hand-written base-model entries in config/models.yaml."""


_RegistryDumper.add_representer(
    type(None),
    lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)


def _dump_model_registry(config: dict[str, Any]) -> str:
    """Serialise a trained-model registry: the header comment, then one entry per model
    with a blank line between them (yaml.dump alone drops comments and packs entries
    together).

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
    return _TRAINED_REGISTRY_HEADER + "\n\n" + "\n".join(entries)


def register_model(
    model_key: str,
    base: str,
    path: str,
    properties: dict[str, Any],
    config_file: str = TRAINED_MODELS_FILE,
) -> None:
    """
    Add a trained model entry to a trained-model registry: TRAINED_MODELS_FILE for real
    runs, DEBUG_MODELS_FILE for debug runs. The file is created if it does not exist.

    Creates or overwrites the entry for `model_key`, recording the `base` it was trained
    from, the checkpoint `path`, and the training `properties` (the knobs the run was
    launched with) - stored because `model_key` is a non-descriptive run id, so they are
    what make the model recognisable later. Uses an exclusive file lock so concurrent
    slurm jobs cannot corrupt the file.
    """
    # separate lock file, so we hold the lock without truncating the yaml itself
    lock_path = config_file + ".lock"

    with open(file=lock_path, mode="a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # block until we are the only writer
        try:
            # load existing entries inside the lock so we see all prior writes
            config = {}
            if os.path.exists(config_file):
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
