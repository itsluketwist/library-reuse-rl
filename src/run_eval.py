"""Evaluation of untrained or pretrained models - a placeholder, since inference isn't wired up yet."""

from src.utils.config import load_full_yaml, load_yaml_config
from src.utils.log import log, log_header


def run(
    model: str,
    datasets: list[str] | None = None,
    model_config: str = "config/models.yaml",
    decoding_profile: str = "greedy",
    decoding_config: str = "config/decoding.yaml",
    eval_datasets_config: str = "config/eval_datasets.yaml",
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key from `model_config` (see config/models.yaml), which holds both
    base models (with a blank "path") and trained models (registered by register_model
    in src/utils/config.py once training produces one) - this is what lets the
    post-training eval job (queued by scripts/submit_train.sh) point at the model a
    training run just produced, by name, the same way it points at a base model.

    `datasets` is a list of nice names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml). If not given, every dataset in that file is run.

    Placeholder: resolves the model, decoding, and dataset configs, then just reports
    them for each dataset in turn. Will eventually load each dataset, generate
    responses from the model, and score them with the response pipeline (see
    src/response/pipeline.py).
    """
    model_settings = load_yaml_config(config_file=model_config, key=model)
    log(f"Model path: {model_settings['base']}")
    if model_settings.get("path"):
        log(f"Trained output path: {model_settings['path']}")

    decoding_settings = load_yaml_config(
        config_file=decoding_config, key=decoding_profile
    )
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")

    available_datasets = load_full_yaml(eval_datasets_config)
    dataset_names = datasets if datasets is not None else list(available_datasets)

    for name in dataset_names:
        log_header(f"DATASET: {name}")
        log(f"Dataset file: {available_datasets[name]}")
