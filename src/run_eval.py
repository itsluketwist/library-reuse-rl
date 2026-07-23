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
        "Model key from models.yaml - either a base model, or a trained model "
        "registered by register_model in src/utils/config.py. Not required if "
        "--debug is set."
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
    default="config/train/demo.yaml",
    help=(
        "Path to decoding config file - defaults to the training recipe file, which "
        "also carries eval's decoding profiles (see its eval_decoding_profiles "
        "section), so training and inference share exactly one place for sampling "
        "settings."
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
    choices=["lib", "use", "base"],
    default="lib",
    help=(
        "Wording used for the library system prompt, when the recipe file's "
        "libraries_requirements_file is set (see "
        "src/prompts.py::build_system_prompt): 'lib' just lists the available "
        "libraries, 'use' also pushes the model to prefer them over writing its own "
        "implementations. Changes the output suffix from e.g. '-libv1' to '-usev1'. "
        "'base' skips the system prompt entirely, regardless of "
        "libraries_requirements_file."
    ),
)

parser.add_argument(
    "--debug",
    action="store_true",
    help=(
        f"Run a tiny smoke test: defaults the model to '{DEBUG_MODEL_KEY}' (see "
        "config/models.yaml) if --model isn't given, and truncates each dataset to "
        "a handful of samples, to quickly check the eval code runs end-to-end."
    ),
)


def eval_results_path(
    model: str,
    dataset: str,
    decoding_profile: str,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
) -> Path:
    """
    Compute the output path an evaluation run saves its results to.

    Output lives under output/{model}/eval/ so training and eval results for the same
    model share a directory. The file name encodes the dataset and decoding profile
    (which produce genuinely different results) but not the model name, since that is
    already captured by the directory. When `libraries_requirements_file` is set, a
    suffix (e.g. "-libv1" or "-usev1", depending on `prompt_style`) is appended to
    separate runs under different library system-prompt settings (see
    src/prompts.py::library_prompt_suffix).

    Returns the output file path.
    """
    lib_suffix = library_prompt_suffix(libraries_requirements_file, style=prompt_style)
    return (
        Path("output")
        / model
        / "eval"
        / f"{dataset}-{decoding_profile}{lib_suffix}.json"
    )


def run_eval(
    model: str | None,
    datasets: list[str] | None = None,
    model_config: str = "config/models.yaml",
    decoding_profile: str = "greedy",
    decoding_config: str = "config/train/demo.yaml",
    eval_datasets_config: str = "config/eval_datasets.yaml",
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Run evaluation of an untrained, pretrained, or trained model against one or more
    datasets.

    `model` is a key from `model_config` (see config/models.yaml), which holds both
    base models (with a blank "path") and trained models (registered by register_model
    in src/utils/config.py once training produces one) - this is what lets the
    post-training eval job (queued by scripts/submit_grpo.sh) point at the model a
    training run just produced, by name, the same way it points at a base model.

    `datasets` is a list of nice names looked up in `eval_datasets_config` (see
    config/eval_datasets.yaml). If not given, every dataset in that file is run.

    The system prompt's libraries_requirements_file is read straight from
    `decoding_config` (the same recipe file training uses - see
    src/prompts.py::build_system_prompt and eval_results_path), so the model is
    always evaluated under the library list it was trained on. `prompt_style` is
    still a free choice, since it isn't part of the recipe file either - setting it to
    "base" forces no system prompt at all, regardless of the recipe file.

    If `debug` is set, src/eval/evaluate.py::evaluate_model truncates each dataset to
    a handful of samples - useful to quickly check the eval code runs end-to-end.
    `model` defaults to `DEBUG_MODEL_KEY` in that case, but an explicit `model` is
    still respected, so a bigger model can be smoke-tested too. Without `debug`,
    `model` is required.

    Resolves the model, decoding settings, and dataset files/output paths, then runs
    evaluation (see src/eval/evaluate.py::evaluate_model, which loads the model and
    handles generation, scoring, and saving itself) against each dataset, saving
    results under the paths returned by `eval_results_path`.
    """
    if model is None:
        if not debug:
            raise ValueError("--model is required unless --debug is set.")
        model = DEBUG_MODEL_KEY

    model_settings = load_yaml_config(config_file=model_config, key=model)
    # prefer a trained checkpoint's path if one's registered, otherwise the base model
    model_name_or_path = model_settings["path"] or model_settings["base"]
    log(f"Model: {model_name_or_path}")
    log(f"Prompt style: {prompt_style}")
    log(f"Debug: {debug}")

    recipe = load_full_yaml(decoding_config)
    libraries_requirements_file = recipe.get("libraries_requirements_file")
    log(f"Libraries requirements file: {libraries_requirements_file or 'none'}")

    decoding_settings = load_yaml_config(
        config_file=decoding_config,
        key=decoding_profile,
        section="eval_decoding_profiles",
    )
    log(f"Decoding profile '{decoding_profile}': {decoding_settings}")

    available_datasets = load_full_yaml(eval_datasets_config)
    dataset_names = datasets if datasets is not None else list(available_datasets)
    dataset_files = {name: available_datasets[name] for name in dataset_names}
    output_paths = {
        name: eval_results_path(
            model=model,
            dataset=name,
            decoding_profile=decoding_profile,
            libraries_requirements_file=libraries_requirements_file,
            prompt_style=prompt_style,
        )
        for name in dataset_names
    }

    evaluate_model(
        model_name_or_path=model_name_or_path,
        dataset_files=dataset_files,
        decoding_settings=decoding_settings,
        output_paths=output_paths,
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
        debug=args.debug,
    )

    # --- summary ---
    log_header("COMPLETE")
    log("Done.")


if __name__ == "__main__":
    main()
