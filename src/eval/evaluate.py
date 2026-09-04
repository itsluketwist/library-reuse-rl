"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

from pathlib import Path

from src.data import load_eval_dataset
from src.eval.generate_api import ApiSettings, generate_api_responses
from src.eval.scoring import score_dataset
from src.prompts import PromptStyle, build_system_prompt
from src.response.evaluate import resolve_eval_python
from src.utils.io import save_json
from src.utils.log import log, log_header, log_timer


# sample count used by --debug mode (src/run_eval.py) for a quick end-to-end check
DEBUG_EVAL_SAMPLES = 4

# generation length cap for --debug runs
DEBUG_MAX_NEW_TOKENS = 512

# name of the venv --debug runs reuse, under a dataset's environment_root
DEBUG_ENVIRONMENT_NAME = "debug"


def evaluate_model(
    base: str,
    adapter_path: str | None,
    dataset_configs: dict[str, dict[str, str]],
    decoding_settings: dict,
    output_paths: dict[str, Path],
    max_lora_rank: int | None = None,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
    api: ApiSettings | None = None,
) -> None:
    """
    Generate responses from `base` against each dataset in `dataset_configs`, analyse
    them with the response pipeline, and save results.

    `api`, when given, serves `base` as a hosted chat model over an OpenAI-compatible API
    (see src/eval/generate_api.py) and vllm is never imported; adapters aren't meaningful
    for a hosted model, so `adapter_path` must be None. Otherwise `base` is loaded as
    local weights via vllm, optionally with a trained lora adapter from `adapter_path`
    applied on top for every generation (this is how a trained model is evaluated) -
    `max_lora_rank` sizes vllm's adapter slot and should be at least the adapter's rank.

    `dataset_configs` and `output_paths` are keyed by dataset name (see
    config/eval_datasets.yaml); every dataset config names its file and evaluator. The
    model is loaded/connected once and reused across all of them.
    For each dataset, generates a response per prompt and scores it the way that dataset's
    task is verified - executing tested-implementation code, or the deterministic prose
    pick / import classification for the new library-reuse splits (see score_dataset) -
    then saves one record per prompt.

    `libraries_requirements_file` and `prompt_style` build the system prompt as training
    does (see build_system_prompt) - match these to the training recipe so the model is
    evaluated as it trained. When the file is None, no system prompt is used.

    If `debug` is set, each dataset is truncated to DEBUG_EVAL_SAMPLES rows for a quick
    end-to-end smoke test, and execution environments are reused between runs rather than
    rebuilt (see DEBUG_ENVIRONMENT_NAME).
    """
    if api is not None and adapter_path is not None:
        raise ValueError(
            "API-served models do not support adapters - the model key's registered "
            "path must be blank, and --adapter must not be set."
        )

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

    if api is not None:
        log("*** Using API provider (no local model load) ***")
        log(f"API base url: {api.base_url or '(default)'}")
        llm = None
        lora_request = None
    else:
        # imported here, not at module scope, so API-only evaluation never needs vllm
        # (and therefore torch) installed at all
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        log("*** Loading model ***")
        # leave dtype unset so vllm uses its `auto` default from the model checkpoint.
        llm_kwargs = {"model": base, "enforce_eager": debug}

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
        # the evaluator builds a disposable venv under this root per run and deletes it
        # afterwards, so concurrent eval jobs can never corrupt a shared one
        environment_root = dataset_config.get("environment_root")
        # debug runs are repeated smoke tests, never concurrent, so they reuse one named
        # venv instead - no reinstalling the same packages, and it's there to inspect after
        environment_cache_dir = (
            str(Path(environment_root) / DEBUG_ENVIRONMENT_NAME)
            if debug and environment_root is not None
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
            if api is not None:
                responses = generate_api_responses(
                    model=base,
                    api=api,
                    prompts=list(dataset["prompt"]),
                    decoding_settings=decoding_settings,
                )
            else:
                # deferred alongside the vllm import above, for the same reason
                from src.eval.generate_vllm import generate_vllm_responses

                responses = generate_vllm_responses(
                    llm=llm,
                    prompts=list(dataset["prompt"]),
                    decoding_settings=decoding_settings,
                    lora_request=lora_request,
                )

        # score the responses the way this dataset's task is verified (execution for the
        # tested split, prose pick / import classification for the new splits); score_dataset
        # takes care of which evaluators want every sample vs just the first
        analysis, results = score_dataset(
            dataset=dataset,
            responses=responses,
            evaluator=evaluator,
            python_executable=python_executable,
            environment_root=environment_root,
            environment_cache_dir=environment_cache_dir,
        )
        log(f"Scored {analysis['n_total']} responses ({analysis['task']})")

        output = {
            "metadata": {
                "base": base,
                "adapter": adapter_path,
                "provider": "api" if api is not None else "vllm",
                "dataset": name,
                "dataset_file": dataset_file,
                "evaluator": evaluator,
                "python_executable": python_executable,
                "environment_root": environment_root,
                "environment_cache_dir": environment_cache_dir,
                "model_dtype": None if api is not None else "auto",
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
