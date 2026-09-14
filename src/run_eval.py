"""The command-line entry point and orchestration for evaluating untrained, pretrained, or trained models."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

from src.eval import evaluate_model
from src.eval.generate_api import DEFAULT_MAX_TOKENS_PARAM, ApiSettings
from src.prompts import PromptStyle, library_prompt_suffix
from src.utils.config import load_full_yaml, load_yaml_config
from src.utils.log import log, log_header


# strftime format for run_timestamp, matching the `date +%y%m%dT%H%M%S` the shell launcher
# scripts (scripts/submit_eval.sh, scripts/run_eval_local.sh) use for their log directory,
# so a run's output and log folders share the same name when they pass theirs through.
RUN_TIMESTAMP_FORMAT = "%y%m%dT%H%M%S"


# tiny model (see config/models.yaml) used to smoke-test --debug runs
DEBUG_MODEL_KEY = "qwen3-debug"

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
        "Model key from models.yaml - a base model, or a trained model registered by "
        "training. Not required if --debug is set."
    ),
)

parser.add_argument(
    "--adapter",
    type=str,
    default=None,
    help=(
        "Path to a trained lora adapter to apply on top of the model's base. Overrides "
        "the adapter registered for --model. Leave unset to use the registered adapter, "
        "or the base model alone if it has none."
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
    "-mc",
    "--model-config",
    type=str,
    default="config/models.yaml",
    help="Path to model config file.",
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
    "--run-timestamp",
    type=str,
    default=None,
    help=(
        "Timestamp (RUN_TIMESTAMP_FORMAT, e.g. 260816T174336) grouping this run's output "
        "files (see eval_results_path). The launcher scripts generate one per invocation "
        "and pass it to every run_eval call in that batch, so results and the matching "
        "logs/<timestamp>/ folder share a name. Defaults to the current time if omitted."
    ),
)


def eval_results_path(
    run_timestamp: str,
    model: str,
    dataset: str,
    decoding_profile: str,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    adapter_override: str | None = None,
) -> Path:
    """
    Compute the output path an evaluation run saves its results to.

    Output lives under output/eval/{run_timestamp}/{model}/, with the file name encoding
    the dataset and decoding profile. Grouping by `run_timestamp` first means rerunning a
    model/decoding-profile/prompt-style combination never overwrites a previous run's
    results - each invocation gets its own folder (see RUN_TIMESTAMP_FORMAT). A suffix
    ("-libv1"/"-usev1", else the style name) separates runs under different system-prompt
    settings (see library_prompt_suffix).

    `adapter_override` is the --adapter path when given: its directory name is appended
    so trying an arbitrary adapter under a base-model key doesn't overwrite that model's
    own results. Registered runs need no suffix, as their --model key is already unique.

    Returns the output file path.
    """
    lib_suffix = library_prompt_suffix(libraries_requirements_file, style=prompt_style)
    # "base", and "lib"/"use" without a requirements file, all give an empty suffix - fall
    # back to the style name so those runs can't share one path and overwrite each other
    if not lib_suffix:
        lib_suffix = f"-{prompt_style}"
    adapter_suffix = (
        f"-adapter-{Path(adapter_override).name}" if adapter_override else ""
    )
    return (
        Path("output")
        / "eval"
        / run_timestamp
        / model
        / f"{dataset}-{decoding_profile}{lib_suffix}{adapter_suffix}.json"
    )


def run_eval(
    model: str | None,
    datasets: list[str] | None = None,
    model_config: str = "config/models.yaml",
    decoding_profile: str = "greedy",
    decoding_config: str = "config/training.yaml",
    eval_datasets_config: str = "config/eval_datasets.yaml",
    prompt_style: PromptStyle = "lib",
    adapter: str | None = None,
    debug: bool = False,
    run_timestamp: str | None = None,
    n_samples: int | None = None,
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key from `model_config` (see config/models.yaml), holding local base
    models (blank "path") and trained models (registered once training produces one), so
    the post-training eval job can point at a fresh model by name, as well as API-served
    models ("provider: api" - DeepSeek, OpenAI, ...). For local models, eval loads the
    entry's "base" and applies its adapter on top; `adapter` overrides which adapter is
    applied (e.g. to try one on a base-model key); with neither set, the base model is
    evaluated alone. API models have no adapter and skip vllm entirely (see evaluate_model).

    `datasets` is a list of names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml); if not given, every dataset in that file is run.

    The system prompt's libraries_requirements_file is read from `decoding_config` (the
    recipe training uses), so the model is evaluated under the library list it trained
    on. `prompt_style` is still a free choice - "base" forces no system prompt at all.

    If `debug` is set, each dataset is truncated to a few samples (a quick end-to-end
    check) and `model` defaults to DEBUG_MODEL_KEY; otherwise `model` is required.

    `run_timestamp` groups this run's output files under output/eval/{run_timestamp}/
    (see eval_results_path); defaults to the current time (RUN_TIMESTAMP_FORMAT) when not
    given, so a standalone invocation still gets its own folder.

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

    if run_timestamp is None:
        run_timestamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    log(f"Run timestamp: {run_timestamp}")

    model_settings = load_yaml_config(config_file=model_config, key=model)
    # --adapter override wins, else the entry's registered adapter, else base alone
    base = model_settings["base"]
    adapter_path = adapter or model_settings["path"] or None
    api = build_api_settings(model_settings)
    log(f"Base: {base}")
    log(f"Provider: {'api' if api is not None else 'vllm'}")
    log(f"Adapter: {adapter_path or 'none'}")
    log(f"Prompt style: {prompt_style}")
    log(f"Debug: {debug}")

    recipe = load_full_yaml(decoding_config)
    libraries_requirements_file = recipe.get("libraries_requirements_file")
    log(f"Libraries requirements file: {libraries_requirements_file or 'none'}")
    if api is None:
        log("Model dtype: auto (vLLM default)")

    # size vllm's adapter slot from the recipe's training rank (assumes same recipe)
    max_lora_rank = recipe.get("lora_r")

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
    output_paths = {
        name: eval_results_path(
            run_timestamp=run_timestamp,
            model=model,
            dataset=name,
            decoding_profile=decoding_profile,
            libraries_requirements_file=libraries_requirements_file,
            prompt_style=prompt_style,
            adapter_override=adapter,
        )
        for name in dataset_names
    }

    evaluate_model(
        base=base,
        adapter_path=adapter_path,
        dataset_configs=dataset_configs,
        decoding_settings=decoding_settings,
        output_paths=output_paths,
        max_lora_rank=max_lora_rank,
        libraries_requirements_file=libraries_requirements_file,
        prompt_style=prompt_style,
        debug=debug,
        api=api,
    )


def main() -> None:
    args = parser.parse_args()
    datasets = args.datasets.split(",") if args.datasets else None

    # --- startup banner ---
    log_header("EVALUATION")
    log(f"Model: {args.model}")
    log(f"Adapter: {args.adapter or '(registered)'}")
    log(f"Datasets: {datasets or 'all'}")
    log(f"Decoding: {args.decoding_config} [{args.decoding_profile}]")
    log(
        f"n_samples: {args.n_samples if args.n_samples is not None else '(default: 1)'}"
    )
    log(f"Prompt style: {args.prompt_style}")
    log(f"Debug: {args.debug}")
    log()

    run_eval(
        model=args.model,
        datasets=datasets,
        model_config=args.model_config,
        decoding_profile=args.decoding_profile,
        decoding_config=args.decoding_config,
        eval_datasets_config=args.eval_datasets_config,
        prompt_style=args.prompt_style,
        adapter=args.adapter,
        debug=args.debug,
        run_timestamp=args.run_timestamp,
        n_samples=args.n_samples,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
