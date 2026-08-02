"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

import dataclasses
from pathlib import Path

from vllm import LLM
from vllm.lora.request import LoRARequest

from src.data import load_eval_dataset
from src.prompts import PromptStyle, build_system_prompt
from src.response.pipeline import process_response
from src.utils.io import save_json
from src.utils.log import log, log_header, log_timer

from .generate import generate_responses


# dtype used to load the model for generation, matching config/training.yaml
EVAL_DTYPE = "bfloat16"

# sample count used by --debug mode (src/run_eval.py) for a quick end-to-end check
DEBUG_EVAL_SAMPLES = 4

# generation length cap for --debug runs
DEBUG_MAX_NEW_TOKENS = 512


def _average(
    values: list[float],
) -> float | None:
    """
    Average a list of numbers.

    Returns the mean, or None if there are no values to average.
    """
    return sum(values) / len(values) if values else None


def evaluate_model(
    base: str,
    adapter_path: str | None,
    dataset_files: dict[str, str],
    decoding_settings: dict,
    output_paths: dict[str, Path],
    max_lora_rank: int | None = None,
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Generate responses from the `base` model (optionally with a trained lora adapter
    from `adapter_path`) against each dataset in `dataset_files`, analyse them with the
    response pipeline, and save results.

    `base` is loaded as the vllm model; when `adapter_path` is given, that adapter is
    applied on top for every generation, otherwise the base model runs alone (this is
    how a trained model is evaluated). `max_lora_rank` sizes vllm's adapter slot and
    should be at least the adapter's rank (only used when `adapter_path` is set).

    `dataset_files` and `output_paths` are keyed by dataset name (see
    config/eval_datasets.yaml); the model is loaded once and reused across all of them.
    For each dataset, generates a response per prompt, parses and evaluates it against
    its test cases (see process_response), and saves one record per prompt.

    `libraries_requirements_file` and `prompt_style` build the system prompt as training
    does (see build_system_prompt) - match these to the training recipe so the model is
    evaluated as it trained. When the file is None, no system prompt is used.

    If `debug` is set, each dataset is truncated to DEBUG_EVAL_SAMPLES rows for a quick
    end-to-end smoke test.
    """
    system_prompt = build_system_prompt(libraries_requirements_file, style=prompt_style)

    # cap generation length for the debug smoke test (copy, don't mutate the shared dict)
    if debug:
        decoding_settings = {
            **decoding_settings,
            "max_new_tokens": DEBUG_MAX_NEW_TOKENS,
        }

    log("*** Loading model ***")
    llm_kwargs = {"model": base, "dtype": EVAL_DTYPE, "enforce_eager": debug}

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

    for name, dataset_file in dataset_files.items():
        log_header(f"DATASET: {name}")
        log(f"Dataset file: {dataset_file}")

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

        results = []
        n_passed = 0
        for row, (response, stop_reason) in zip(dataset, responses):
            processed = process_response(
                response=response,
                record={
                    "inputs": row["inputs"],
                    "expected_outputs": row["expected_outputs"],
                },
            )
            n_passed += processed.passed
            # store idx (original row number) to identify the problem without the prompt
            results.append(
                {
                    "idx": row["idx"],
                    "stop_reason": stop_reason,
                    **dataclasses.asdict(processed),
                }
            )

        n_total = len(results)
        n_valid_code = sum(1 for r in results if r.get("valid") is True)
        n_overflow = sum(1 for r in results if r.get("stop_reason") == "length")
        n_reasoning = sum(1 for r in results if r.get("reasoning"))
        n_answer = sum(1 for r in results if r.get("answer"))

        # average over valid-code responses only, so missing/unparseable code doesn't skew
        valid_results = [r for r in results if r.get("valid") is True]
        avg_std_libs = _average([len(r["std_libs"]) for r in valid_results])
        avg_ext_libs = _average([len(r["ext_libs"]) for r in valid_results])
        avg_ll = _average(
            [
                r["logical_lines"]
                for r in valid_results
                if r["logical_lines"] is not None
            ]
        )
        avg_mi = _average(
            [
                r["maintainability_index"]
                for r in valid_results
                if r["maintainability_index"] is not None
            ]
        )

        log(f"Passed {n_passed}/{n_total}")

        output = {
            "metadata": {
                "base": base,
                "adapter": adapter_path,
                "dataset": name,
                "dataset_file": dataset_file,
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
            "analysis": {
                "n_total": n_total,
                "n_passed": n_passed,
                "accuracy": n_passed / n_total if n_total > 0 else 0.0,
                "n_reasoning": n_reasoning,
                "n_answer": n_answer,
                "n_overflow": n_overflow,
                "n_valid_code": n_valid_code,
                "avg_std_libs": avg_std_libs,
                "avg_ext_libs": avg_ext_libs,
                "avg_ll": avg_ll,
                "avg_mi": avg_mi,
            },
            "results": results,
        }

        output_path = output_paths[name]
        save_json(output, output_path)
        log(f"Saved results to {output_path}")
