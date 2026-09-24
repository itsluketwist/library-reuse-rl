"""The command-line entry point and orchestration for evaluating untrained, pretrained, or trained models."""

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal, get_args

from src.eval import evaluate_model
from src.eval.generate_api import DEFAULT_MAX_TOKENS_PARAM, ApiSettings
from src.prompts import PromptStyle, library_prompt_suffix
from src.utils.config import load_full_yaml, load_model_settings, load_yaml_config
from src.utils.log import log, log_header, silence_library_warnings
from src.utils.paths import model_output_dir


# tiny model (see config/models.yaml) used to smoke-test --debug runs
DEBUG_MODEL_KEY = "qwen3-debug"

# environment variable overriding where a dataset's execution environments are built (see
# environment_root in config/eval_datasets.yaml), so HPC jobs can put them on scratch
ENVIRONMENT_ROOT_ENV_VAR = "EVAL_ENVIRONMENT_ROOT"

# how a model entry is served. "provider" is absent on every local/trained entry, so it
# defaults to "vllm"; "api" entries are called over an OpenAI-compatible endpoint instead
Provider = Literal["vllm", "api"]
DEFAULT_PROVIDER: Provider = "vllm"


def build_api_settings(
    model_settings: dict[str, Any],
) -> ApiSettings | None:
    """
    Read a model entry's API configuration (see config/models.yaml).

    Returns the settings for an "api" entry, or None for a local one served by vllm.
    """
    provider = model_settings.get("provider", DEFAULT_PROVIDER)
    # caught here rather than silently falling back to vllm, which would try to load an
    # api model id ("deepseek-v4-flash") as local weights and fail confusingly
    if provider not in get_args(Provider):
        raise ValueError(
            f"unknown provider '{provider}' - expected one of {get_args(Provider)}."
        )

    if provider != "api":
        return None

    api_key_env = model_settings.get("api_key_env")
    if api_key_env is None:
        raise ValueError(
            "API-served models must set 'api_key_env' in config/models.yaml, naming "
            "the environment variable that holds the API key."
        )

    return ApiSettings(
        api_key_env=api_key_env,
        base_url=model_settings.get("base_url"),
        extra_params=model_settings.get("extra_params"),
        max_tokens_param=model_settings.get(
            "max_tokens_param", DEFAULT_MAX_TOKENS_PARAM
        ),
    )


# create the main argument parser
parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default=None,
    help=(
        "Model key - a base model from config/models.yaml, or a trained model "
        "registered by training (config/trained_models.yaml, or "
        "config/debug_models.yaml for debug runs). Not required if --debug is set."
    ),
)

parser.add_argument(
    "-d",
    "--datasets",
    type=str,
    default=None,
    help=(
        "Comma-separated dataset names from eval_datasets.yaml. "
        "Defaults to every dataset in that file."
    ),
)

parser.add_argument(
    "--decoding-profile",
    type=str,
    default="greedy",
    help="Decoding profile name within the decoding config file's eval_decoding_profiles section.",
)

parser.add_argument(
    "--decoding-config",
    type=str,
    default="config/training.yaml",
    help=(
        "Path to decoding config file. Defaults to the training recipe, which also "
        "carries eval's decoding profiles (eval_decoding_profiles), so training and "
        "inference share one place for sampling settings."
    ),
)

parser.add_argument(
    "--eval-datasets-config",
    type=str,
    default="config/eval_datasets.yaml",
    help="Path to eval datasets config file.",
)

parser.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help=(
        "How many completions to sample per prompt (e.g. for LibReuseBench's pass@k or "
        "library-usage's per-response reuse stats). Eval-only - training rollouts use "
        "num_generations instead. Defaults to 1 if not given; with greedy decoding, more "
        "than 1 just repeats the same output."
    ),
)

parser.add_argument(
    "--prompt-style",
    type=str,
    choices=["lib", "use", "std", "base"],
    default="lib",
    help=(
        "System-prompt wording (see src/prompts.py): 'lib' lists the available "
        "libraries, 'use' also pushes the model to prefer them (both need the recipe's "
        "libraries_requirements_file), 'std' pushes the model to prefer the standard "
        "library (independent of that file), 'base' skips the system prompt. Sets the "
        "output suffix."
    ),
)

parser.add_argument(
    "--debug",
    action="store_true",
    help=(
        f"Run a tiny smoke test: defaults the model to '{DEBUG_MODEL_KEY}' if --model "
        "isn't given, and truncates each dataset to a few samples, to check the eval "
        "code runs end-to-end."
    ),
)

parser.add_argument(
    "-u",
    "--update",
    action="store_true",
    default=False,
    help=(
        "Regenerate results even if the output file already exists (see "
        "eval_results_path). Without this, a dataset whose result file is already there "
        "is skipped, so resubmitting the same models/datasets only fills in what is "
        "missing."
    ),
)


def eval_results_path(
    output_dir: Path,
    dataset: str,
    decoding_profile: str,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
) -> Path:
    """
    Compute the output path an evaluation run saves its results to.

    `output_dir` is the model's folder (see model_output_dir in src/utils/paths.py -
    output/baselines/{model}/ for a base model, a training run's epoch{n}/ folder for a
    checkpoint), and the file name encodes the dataset and decoding profile. A rerun of
    the same model/dataset/decoding-profile/prompt-style combination overwrites the
    previous result file at this same path - see the `update` flag on `run_eval`, which
    skips regenerating a result that already exists here unless told otherwise. A suffix
    ("-libv1"/"-usev1", else the style name) separates runs under different system-prompt
    settings (see library_prompt_suffix).

    Returns the output file path.
    """
    lib_suffix = library_prompt_suffix(libraries_requirements_file, style=prompt_style)
    # "base", and "lib"/"use" without a requirements file, all give an empty suffix - fall
    # back to the style name so those runs can't share one path and overwrite each other
    if not lib_suffix:
        lib_suffix = f"-{prompt_style}"
    return output_dir / f"{dataset}-{decoding_profile}{lib_suffix}.json"


def apply_environment_root(
    dataset_configs: dict[str, dict[str, str]],
    environment_root: str | None,
) -> dict[str, dict[str, str]]:
    """
    Move each dataset's execution environments to `environment_root`, when one is set.

    The environments are venvs built per run and thrown away, so on the HPC they belong on
    scratch rather than inside the repo (see ENVIRONMENT_ROOT_ENV_VAR). Datasets that
    never build one (no environment_root of their own) are left untouched.

    Returns the dataset configs, with environment_root replaced where it applies.
    """
    if not environment_root:
        return dataset_configs
    return {
        name: {**config, "environment_root": environment_root}
        if config.get("environment_root")
        else config
        for name, config in dataset_configs.items()
    }


def run_eval(
    model: str | None,
    datasets: list[str] | None = None,
    decoding_profile: str = "greedy",
    decoding_config: str = "config/training.yaml",
    eval_datasets_config: str = "config/eval_datasets.yaml",
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
    update: bool = False,
    n_samples: int | None = None,
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key in the model registry (see load_model_settings): released base
    models (blank "path") and API-served models ("provider: api" - DeepSeek, OpenAI,
    ...) in config/models.yaml, and each epoch of a training run in
    config/trained_models.yaml (config/debug_models.yaml for debug runs), registered once
    training produces them so the post-training eval jobs can point at a fresh
    checkpoint by name. A local model is loaded from the entry's "path" when it has one,
    and from its "base" otherwise. API models have no local weights and skip vllm
    entirely (see evaluate_model).

    `datasets` is a list of names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml); if not given, every dataset in that file is run.

    The system prompt's libraries_requirements_file is read from `decoding_config` (the
    recipe training uses), so the model is evaluated under the library list it trained
    on. `prompt_style` is still a free choice - "base" forces no system prompt at all.

    If `debug` is set, each dataset is truncated to a few samples (a quick end-to-end
    check) and `model` defaults to DEBUG_MODEL_KEY; otherwise `model` is required.

    A dataset whose result file (see `eval_results_path`) already exists is skipped unless
    `update` is set, so resubmitting the same models/datasets only fills in what is
    missing rather than redoing everything. If every requested dataset already has
    results, the model is never loaded at all.

    `n_samples` overrides how many completions are sampled per prompt for this run - it is
    eval-only (training rollouts use num_generations instead, see config/training.yaml),
    so it is a run_eval choice rather than a decoding-profile field; leave unset to sample
    just 1.

    Resolves the model, decoding settings, and dataset/output paths, then runs
    evaluation (see evaluate_model) against each dataset, saving results under the paths
    from `eval_results_path`.
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

    model_settings = load_model_settings(model_key=model)
    # a trained entry's registered weights (a training checkpoint), else the base model
    base = model_settings["base"]
    weights_path = model_settings["path"] or None
    api = build_api_settings(model_settings)
    log(f"Base: {base}")
    log(f"Provider: {'api' if api is not None else 'vllm'}")
    log(f"Weights: {weights_path or 'base model'}")
    log(f"Prompt style: {prompt_style}")
    log(f"Debug: {debug}")

    recipe = load_full_yaml(decoding_config)
    libraries_requirements_file = recipe.get("libraries_requirements_file")
    log(f"Libraries requirements file: {libraries_requirements_file or 'none'}")
    if api is None:
        log("Model dtype: bfloat16")

    decoding_settings = load_yaml_config(
        config_file=decoding_config,
        key=decoding_profile,
        section="eval_decoding_profiles",
    )
    if n_samples is not None:
        decoding_settings = {**decoding_settings, "n_samples": n_samples}
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")
    log(f"n_samples: {decoding_settings.get('n_samples', 1)}")

    available_datasets = load_full_yaml(eval_datasets_config)
    dataset_names = datasets if datasets is not None else list(available_datasets)
    dataset_configs = {name: available_datasets[name] for name in dataset_names}
    for name, dataset_config in dataset_configs.items():
        if not isinstance(dataset_config, dict):
            raise ValueError(
                f"Dataset '{name}' in {eval_datasets_config} must define path and evaluator."
            )
        missing = {"path", "evaluator"} - set(dataset_config)
        if missing:
            raise ValueError(
                f"Dataset '{name}' in {eval_datasets_config} is missing {sorted(missing)}."
            )
        dataset_path = Path(dataset_config["path"])
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset '{name}' does not exist at {dataset_path}."
            )

    # scripts/setup_job.sh points this at scratch on the HPC (see apply_environment_root)
    environment_root = os.environ.get(ENVIRONMENT_ROOT_ENV_VAR)
    if environment_root:
        log(f"Execution environment root: {environment_root}")
    dataset_configs = apply_environment_root(dataset_configs, environment_root)

    output_dir = model_output_dir(model_key=model, model_settings=model_settings)
    log(f"Results folder: {output_dir}")
    output_paths = {
        name: eval_results_path(
            output_dir=output_dir,
            dataset=name,
            decoding_profile=decoding_profile,
            libraries_requirements_file=libraries_requirements_file,
            prompt_style=prompt_style,
        )
        for name in dataset_names
    }

    # skip a dataset whose result file already exists unless told to regenerate it - debug
    # runs are cheap smoke tests, so they always rerun regardless of what is on disk
    datasets_to_run = {}
    for name, path in output_paths.items():
        if update or debug or not path.exists():
            datasets_to_run[name] = path
        else:
            log(f"  - {name}: results already exist at {path}, skipping")
    if not datasets_to_run:
        log(
            "All requested datasets already have results - nothing to do (use --update to regenerate)."
        )
        return

    evaluate_model(
        base=base,
        weights_path=weights_path,
        dataset_configs={name: dataset_configs[name] for name in datasets_to_run},
        decoding_settings=decoding_settings,
        output_paths=datasets_to_run,
        libraries_requirements_file=libraries_requirements_file,
        prompt_style=prompt_style,
        debug=debug,
        api=api,
    )


def main() -> None:
    args = parser.parse_args()
    silence_library_warnings()
    datasets = args.datasets.split(",") if args.datasets else None

    # --- startup banner ---
    log_header("EVALUATION")
    log(f"Model: {args.model}")
    log(f"Datasets: {datasets or 'all'}")
    log(f"Decoding: {args.decoding_config} [{args.decoding_profile}]")
    log(
        f"n_samples: {args.n_samples if args.n_samples is not None else '(default: 1)'}"
    )
    log(f"Prompt style: {args.prompt_style}")
    log(f"Debug: {args.debug}")
    log(f"Update: {args.update}")
    log()

    run_eval(
        model=args.model,
        datasets=datasets,
        decoding_profile=args.decoding_profile,
        decoding_config=args.decoding_config,
        eval_datasets_config=args.eval_datasets_config,
        prompt_style=args.prompt_style,
        debug=args.debug,
        update=args.update,
        n_samples=args.n_samples,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
