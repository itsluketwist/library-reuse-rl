"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

import dataclasses
from pathlib import Path

from vllm import LLM

from src.data import load_eval_dataset
from src.prompts import PromptStyle, build_system_prompt
from src.response.pipeline import process_response
from src.utils.io import save_json
from src.utils.log import log, log_header, log_timer

from .generate import generate_responses


# dtype used to load the model for generation - matching config/training.yaml's
# model arguments, since eval isn't driven by its own recipe file. no attention-
# implementation setting here (unlike training) - vllm picks its own backend
EVAL_DTYPE = "bfloat16"

# sample count used by --debug mode (src/run_eval.py) - just enough to smoke-test
# generation and scoring end-to-end without waiting on a full dataset
DEBUG_EVAL_SAMPLES = 4


def _average(
    values: list[float],
) -> float | None:
    """
    Average a list of numbers.

    Returns the mean, or None if there are no values to average.
    """
    return sum(values) / len(values) if values else None


def evaluate_model(
    model_name_or_path: str,
    dataset_files: dict[str, str],
    decoding_settings: dict,
    output_paths: dict[str, Path],
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Generate responses from `model_name_or_path` against each dataset in
    `dataset_files`, analyse them with the response pipeline, and save results.

    `dataset_files` and `output_paths` are both keyed by dataset name (see
    config/eval_datasets.yaml) - the model is loaded once and reused across every
    dataset. For each dataset, generates a response per prompt (see
    generate_responses), parses and evaluates each response against its test cases
    (see src/response/pipeline.py::process_response - reused rather than
    reimplementing scoring), and saves one record per prompt to that dataset's
    `output_paths` entry.

    `libraries_requirements_file` and `prompt_style` build the system prompt the same
    way training does (see src/prompts.py::build_system_prompt) - match these to the
    training run's recipe config so the model is evaluated under the same conditions
    it trained on. When `libraries_requirements_file` is None, no system prompt is
    used and `prompt_style` is ignored.

    If `debug` is set, each dataset is truncated to DEBUG_EVAL_SAMPLES rows, to
    quickly smoke-test that evaluation runs end-to-end.
    """
    system_prompt = build_system_prompt(libraries_requirements_file, style=prompt_style)

    log("*** Loading model ***")
    llm = LLM(model=model_name_or_path, dtype=EVAL_DTYPE)

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
                prompts=dataset["prompt"],
                decoding_settings=decoding_settings,
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
            # store idx (original row number in the dataset file) to identify the
            # problem without repeating the full prompt text in every result
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

        # averages computed over responses with valid code only, so missing or
        # unparseable code doesn't skew the library and complexity numbers
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
                "model": model_name_or_path,
                "dataset": name,
                "dataset_file": dataset_file,
                "system_prompt": system_prompt,
                "prompt_style": prompt_style if libraries_requirements_file else None,
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
