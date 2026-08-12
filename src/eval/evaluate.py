"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

from pathlib import Path

from vllm import LLM
from vllm.lora.request import LoRARequest

from src.data import load_eval_dataset
from src.prompts import PromptStyle, build_system_prompt
from src.response.evaluate import resolve_eval_python
from src.utils.io import save_json
from src.utils.log import log, log_header, log_timer

from .generate import generate_responses
from .scoring import score_dataset


# sample count used by --debug mode (src/run_eval.py) for a quick end-to-end check
DEBUG_EVAL_SAMPLES = 4

# generation length cap for --debug runs
DEBUG_MAX_NEW_TOKENS = 512


def evaluate_model(
    base: str,
    adapter_path: str | None,
    dataset_configs: dict[str, dict[str, str]],
    decoding_settings: dict,
    output_paths: dict[str, Path],
    model_dtype: str = "auto",
    max_lora_rank: int | None = None,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Generate responses from the `base` model (optionally with a trained lora adapter
    from `adapter_path`) against each dataset in `dataset_configs`, analyse them with the
    response pipeline, and save results.

    `base` is loaded as the vllm model; when `adapter_path` is given, that adapter is
    applied on top for every generation, otherwise the base model runs alone (this is
    how a trained model is evaluated). `max_lora_rank` sizes vllm's adapter slot and
    should be at least the adapter's rank (only used when `adapter_path` is set).

    `dataset_configs` and `output_paths` are keyed by dataset name (see
    config/eval_datasets.yaml); every dataset config names its file and evaluator. The
    model is loaded once and reused across all of them.
    For each dataset, generates a response per prompt and scores it the way that dataset's
    task is verified - executing tested-implementation code, or the deterministic prose
    pick / import classification for the new library-reuse splits (see score_dataset) -
    then saves one record per prompt.

    `libraries_requirements_file` and `prompt_style` build the system prompt as training
    does (see build_system_prompt) - match these to the training recipe so the model is
    evaluated as it trained. When the file is None, no system prompt is used.

    If `debug` is set, each dataset is truncated to DEBUG_EVAL_SAMPLES rows for a quick
    end-to-end smoke test.
    """
    system_prompt = build_system_prompt(libraries_requirements_file, style=prompt_style)

    # validate execution environments before loading a large model or generating anything.
    for dataset_config in dataset_configs.values():
        python_executable = dataset_config.get("python_executable")
        if python_executable is not None:
            resolve_eval_python(python_executable)

    # cap generation length for the debug smoke test (copy, don't mutate the shared dict)
    if debug:
        decoding_settings = {
            **decoding_settings,
            "max_new_tokens": DEBUG_MAX_NEW_TOKENS,
        }

    log("*** Loading model ***")
    llm_kwargs = {"model": base, "dtype": model_dtype, "enforce_eager": debug}

    if adapter_path is not None:
        llm = LLM(**llm_kwargs, enable_lora=True, max_lora_rank=max_lora_rank)
        lora_request = LoRARequest(
            lora_name="trained-adapter",
            lora_int_id=1,
            lora_path=adapter_path,
        )
    else:
        llm = LLM(**llm_kwargs)
        lora_request = None

    for name, dataset_config in dataset_configs.items():
        dataset_file = dataset_config["path"]
        evaluator = dataset_config["evaluator"]
        python_executable = dataset_config.get("python_executable")
        environment_root = dataset_config.get("environment_root")
        environment_dir = (
            str(
                Path(environment_root)
                / output_paths[name].parent.name
                / output_paths[name].stem
            )
            if environment_root is not None
            else None
        )
        log_header(f"DATASET: {name}")
        log(f"Dataset file: {dataset_file}")
        log(f"Evaluator: {evaluator}")

        dataset = load_eval_dataset(
            dataset_file=dataset_file,
            system_prompt=system_prompt,
            n=DEBUG_EVAL_SAMPLES if debug else None,
        )

        with log_timer("generation"):
            responses = generate_responses(
                llm=llm,
                prompts=list(dataset["prompt"]),
                decoding_settings=decoding_settings,
                lora_request=lora_request,
            )

        # score the responses the way this dataset's task is verified (execution for the
        # tested split, prose pick / import classification for the new splits)
        analysis, results = score_dataset(
            dataset=dataset,
            responses=responses,
            evaluator=evaluator,
            python_executable=python_executable,
            environment_dir=environment_dir,
        )
        log(f"Scored {analysis['n_total']} responses ({analysis['task']})")

        output = {
            "metadata": {
                "base": base,
                "adapter": adapter_path,
                "dataset": name,
                "dataset_file": dataset_file,
                "evaluator": evaluator,
                "python_executable": python_executable,
                "environment_dir": environment_dir,
                "model_dtype": model_dtype,
                "system_prompt": system_prompt,
                # lib/use need the requirements file to matter; std applies regardless
                "prompt_style": (
                    prompt_style
                    if libraries_requirements_file or prompt_style == "std"
                    else None
                ),
                "decoding_settings": decoding_settings,
                "debug": debug,
            },
            "analysis": analysis,
            "results": results,
        }

        output_path = output_paths[name]
        save_json(output, output_path)
        log(f"Saved results to {output_path}")
