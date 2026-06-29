"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

import dataclasses
from pathlib import Path

from trl import ModelConfig

from src.data import load_eval_dataset
from src.models import get_model, get_tokenizer
from src.response.pipeline import process_response
from src.utils.io import save_jsonl
from src.utils.log import log, log_header, log_timer

from .generate import generate_responses


# settings used to load the model for generation - matching config/train/demo.yaml's
# model arguments, since eval isn't driven by its own recipe file
EVAL_TORCH_DTYPE = "bfloat16"
EVAL_ATTN_IMPLEMENTATION = "sdpa"

# matches config/train/demo.yaml's max_completion_length - the cap on how many tokens
# a response is allowed to generate
DEFAULT_MAX_NEW_TOKENS = 2048


def evaluate_model(
    model_name_or_path: str,
    dataset_files: dict[str, str],
    decoding_settings: dict,
    output_paths: dict[str, Path],
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
    """
    model_args = ModelConfig(
        model_name_or_path=model_name_or_path,
        torch_dtype=EVAL_TORCH_DTYPE,
        attn_implementation=EVAL_ATTN_IMPLEMENTATION,
    )

    log("*** Loading tokenizer and model ***")
    tokenizer = get_tokenizer(model_args)
    model = get_model(model_args, gradient_checkpointing=False)
    model.eval()

    for name, dataset_file in dataset_files.items():
        log_header(f"DATASET: {name}")
        log(f"Dataset file: {dataset_file}")

        dataset = load_eval_dataset(dataset_file=dataset_file, system_prompt=None)

        with log_timer("generation"):
            responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=dataset["prompt"],
                decoding_settings=decoding_settings,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            )

        results = []
        n_passed = 0
        for row, response in zip(dataset, responses):
            processed = process_response(
                response=response,
                record={
                    "inputs": row["inputs"],
                    "expected_outputs": row["expected_outputs"],
                },
            )
            n_passed += processed.passed
            results.append(
                {
                    "source": row["source"],
                    "prompt": row["prompt"],
                    "response": response,
                    **dataclasses.asdict(processed),
                }
            )

        log(f"Passed {n_passed}/{len(results)}")

        output_path = output_paths[name]
        save_jsonl(results, output_path)
        log(f"Saved results to {output_path}")
