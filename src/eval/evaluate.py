"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

from pathlib import Path

from src.data import load_eval_dataset
from src.prompts import PromptStyle, build_system_prompt
from src.response.evaluate import resolve_eval_python
from src.utils.io import save_json
from src.utils.log import log, log_header, log_timer

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
    max_lora_rank: int | None = None,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
    provider: str = "vllm",
    base_url: str | None = None,
    api_key_env: str | None = None,
    extra_params: dict | None = None,
    max_tokens_param: str = "max_tokens",
) -> None:
    """
    Generate responses from `base` against each dataset in `dataset_configs`, analyse
    them with the response pipeline, and save results.

    `provider` selects how `base` is served (see the "provider" key in
    config/models.yaml): "vllm" (default) loads it as local weights via vllm, optionally
    with a trained lora adapter from `adapter_path` applied on top for every generation
    (this is how a trained model is evaluated) - `max_lora_rank` sizes vllm's adapter slot
    and should be at least the adapter's rank. "api" instead calls it as a hosted chat
    model over an OpenAI-compatible API (`base_url`, `api_key_env`, `extra_params`,
    `max_tokens_param` - see src/eval/generate_api.py); adapters aren't meaningful for a
    hosted model, so `adapter_path` must be None. `extra_params` is forwarded verbatim to
    every API request, for provider-specific fields the standard decoding profile doesn't
    cover (e.g. DeepSeek's "thinking"). `max_tokens_param` names the request field the
    generation-length cap is sent under - some newer OpenAI models reject the legacy
    "max_tokens" and require "max_completion_tokens" instead.

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
    end-to-end smoke test.
    """
    if provider == "api":
        if adapter_path is not None:
            raise ValueError(
                "API-served models do not support adapters - the model key's registered "
                "path must be blank, and --adapter must not be set."
            )
        if api_key_env is None:
            raise ValueError(
                "API-served models must set 'api_key_env' in config/models.yaml, naming "
                "the environment variable that holds the API key."
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

    # vllm (and its lora request type) is only imported for the local-weights path, so
    # API-only evaluation never needs vllm/torch installed at all.
    if provider == "api":
        log("*** Using API provider (no local model load) ***")
        log(f"API base url: {base_url or '(default)'}")
        llm = None
        lora_request = None
    else:
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
            if provider == "api":
                from .generate_api import generate_api_responses

                # validated non-None above (api_key_env is required when provider == "api")
                assert api_key_env is not None
                responses = generate_api_responses(
                    model=base,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    prompts=list(dataset["prompt"]),
                    decoding_settings=decoding_settings,
                    extra_params=extra_params,
                    max_tokens_param=max_tokens_param,
                )
            else:
                from .generate import generate_responses

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
                "provider": provider,
                "dataset": name,
                "dataset_file": dataset_file,
                "evaluator": evaluator,
                "python_executable": python_executable,
                "environment_dir": environment_dir,
                "model_dtype": "auto" if provider != "api" else None,
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
