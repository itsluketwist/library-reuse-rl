"""The command-line entry point and orchestration for evaluating untrained, pretrained, or trained models."""

from argparse import ArgumentParser
from pathlib import Path

from src.eval import evaluate_model
from src.prompts import PromptStyle, library_prompt_suffix
from src.utils.config import load_full_yaml, load_yaml_config
from src.utils.log import log, log_header


# tiny model (see config/models.yaml) used to smoke-test --debug runs
DEBUG_MODEL_KEY = "qwen3-debug"


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


def eval_results_path(
    model: str,
    dataset: str,
    decoding_profile: str,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    adapter_override: str | None = None,
) -> Path:
    """
    Compute the output path an evaluation run saves its results to.

    Output lives under output/eval/{model}/, with the file name encoding the dataset and
    decoding profile (the model name is already the directory). When
    `libraries_requirements_file` is set, a suffix ("-libv1"/"-usev1", per `prompt_style`)
    separates runs under different library prompts (see library_prompt_suffix).

    `adapter_override` is the --adapter path when given: its directory name is appended
    so trying an arbitrary adapter under a base-model key doesn't overwrite that model's
    own results. Registered runs need no suffix, as their --model key is already unique.

    Returns the output file path.
    """
    lib_suffix = library_prompt_suffix(libraries_requirements_file, style=prompt_style)
    adapter_suffix = (
        f"-adapter-{Path(adapter_override).name}" if adapter_override else ""
    )
    return (
        Path("output")
        / "eval"
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
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key from `model_config` (see config/models.yaml), holding both base
    models (blank "path") and trained models (registered once training produces one), so
    the post-training eval job can point at a fresh model by name. Training saves lora
    adapters, so eval always loads the entry's "base" and applies its adapter on top.
    `adapter` overrides which adapter is applied (e.g. to try one on a base-model key);
    with neither set, the base model is evaluated alone.

    `datasets` is a list of names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml); if not given, every dataset in that file is run.

    The system prompt's libraries_requirements_file is read from `decoding_config` (the
    recipe training uses), so the model is evaluated under the library list it trained
    on. `prompt_style` is still a free choice - "base" forces no system prompt at all.

    If `debug` is set, each dataset is truncated to a few samples (a quick end-to-end
    check) and `model` defaults to DEBUG_MODEL_KEY; otherwise `model` is required.

    Resolves the model, decoding settings, and dataset/output paths, then runs
    evaluation (see evaluate_model) against each dataset, saving results under the paths
    from `eval_results_path`.
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

    model_settings = load_yaml_config(config_file=model_config, key=model)
    # --adapter override wins, else the entry's registered adapter, else base alone
    base = model_settings["base"]
    adapter_path = adapter or model_settings["path"] or None
    log(f"Base: {base}")
    log(f"Adapter: {adapter_path or 'none'}")
    log(f"Prompt style: {prompt_style}")
    log(f"Debug: {debug}")

    recipe = load_full_yaml(decoding_config)
    libraries_requirements_file = recipe.get("libraries_requirements_file")
    model_dtype = recipe.get("dtype", "auto")
    log(f"Libraries requirements file: {libraries_requirements_file or 'none'}")
    log(f"Model dtype: {model_dtype}")

    # size vllm's adapter slot from the recipe's training rank (assumes same recipe)
    max_lora_rank = recipe.get("lora_r")

    decoding_settings = load_yaml_config(
        config_file=decoding_config,
        key=decoding_profile,
        section="eval_decoding_profiles",
    )
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")

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
        model_dtype=model_dtype,
        max_lora_rank=max_lora_rank,
        libraries_requirements_file=libraries_requirements_file,
        prompt_style=prompt_style,
        debug=debug,
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
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
