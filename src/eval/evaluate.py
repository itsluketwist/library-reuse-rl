"""Evaluate a model against one or more datasets: generation, analysis, and saving."""

import traceback
from pathlib import Path

import lure

from src.data import load_eval_dataset
from src.eval.generate_api import (
    FAILED_STOP_REASON,
    ApiSettings,
    generate_api_responses,
)
from src.eval.scoring import score_dataset
from src.prompts import (
    FIXED_SYSTEM_PROMPTS,
    TASK_SUFFIXES,
    PromptStyle,
    build_system_prompt,
)
from src.response.evaluate import resolve_eval_python
from src.utils.io import load_jsonl, save_json, save_jsonl
from src.utils.log import log, log_header, log_timer
from src.utils.vllm_settings import VLLM_ADDITIONAL_CONFIG


# sample count used by --debug mode (src/run_eval.py) for a quick end-to-end check
DEBUG_EVAL_SAMPLES = 4

# generation length cap for --debug runs - long enough for qwen3's reasoning to finish,
# so debug responses actually contain code to score
DEBUG_MAX_NEW_TOKENS = 2048

# name of the venv --debug runs reuse, under a dataset's environment_root
DEBUG_ENVIRONMENT_NAME = "debug"


def unscored_generations_path(output_path: Path) -> Path:
    """
    Where a dataset's generations are kept until they have been scored, beside its results.

    Returns the path of the .unscored.jsonl file for `output_path`.
    """
    return output_path.with_suffix(".unscored.jsonl")


def save_unscored_generations(
    responses: list[list[tuple[str, str, int | None]]],
    path: Path,
) -> None:
    """
    Save a dataset's generated responses (one line per prompt, with every sample's text,
    stop reason and token count) before they are scored, so a scoring failure never loses
    them.
    """
    save_jsonl(
        [
            {
                "samples": [
                    {
                        "response": response,
                        "stop_reason": stop_reason,
                        "completion_tokens": completion_tokens,
                    }
                    for response, stop_reason, completion_tokens in samples
                ],
            }
            for samples in responses
        ],
        path,
    )


def load_unscored_generations(
    path: Path,
    n_prompts: int,
    n_samples: int,
) -> list[list[tuple[str, str, int | None]]] | None:
    """
    Load generations a previous run saved but couldn't score, so they can be scored without
    generating them again. They are only reused when they match this run: one line per prompt
    in the dataset, each with the expected number of samples.

    Returns the responses (as generate_*_responses returns them), or None if there are no
    saved generations or they don't match.
    """
    if not path.exists():
        return None

    # a file saved before token counts were kept has none, so its samples load as None
    responses = [
        [
            (
                sample["response"],
                sample["stop_reason"],
                sample.get("completion_tokens"),
            )
            for sample in row["samples"]
        ]
        for row in load_jsonl(path)
    ]
    if len(responses) != n_prompts or any(len(s) != n_samples for s in responses):
        log(f"Ignoring saved generations that don't match this run: {path}")
        return None
    return responses


def count_failed_samples(
    responses: list[list[tuple[str, str, int | None]]],
) -> int:
    """
    Count the samples whose API request failed (e.g. a rate limit, or an empty balance).

    Returns the number of failed samples.
    """
    return sum(
        stop_reason == FAILED_STOP_REASON
        for samples in responses
        for _response, stop_reason, _tokens in samples
    )


def retry_failed_api_samples(
    responses: list[list[tuple[str, str, int | None]]],
    prompts: list[list[dict]],
    model: str,
    api: ApiSettings,
    decoding_settings: dict,
) -> list[list[tuple[str, str, int | None]]]:
    """
    Ask the API again, once, for every sample whose request failed, keeping the others.

    A failed request holds no answer, and scoring it would count it as a wrong one. So only
    those samples are requested again, which also lets a run that stopped partway (e.g. when
    the account ran out of credit) continue without paying again for what already succeeded.

    Returns the responses, with each failed sample replaced by its retry.
    """
    failed = [
        (row, sample)
        for row, samples in enumerate(responses)
        for sample, (_response, stop_reason, _tokens) in enumerate(samples)
        if stop_reason == FAILED_STOP_REASON
    ]
    if not failed:
        return responses

    log(f"Asking again for the {len(failed)} sample(s) whose request failed")
    retried = generate_api_responses(
        model=model,
        api=api,
        prompts=[prompts[row] for row, _sample in failed],
        decoding_settings={**decoding_settings, "n_samples": 1},
        max_workers=api.max_workers,
    )
    filled = [list(samples) for samples in responses]
    for (row, sample), retry in zip(failed, retried):
        filled[row][sample] = retry[0]
    return filled


def evaluate_model(
    base: str,
    weights_path: str | None,
    dataset_configs: dict[str, dict[str, str]],
    decoding_settings: dict,
    output_paths: dict[str, Path],
    libraries_requirements_file: str | None = None,
    prompt_style: PromptStyle = "lib",
    debug: bool = False,
    api: ApiSettings | None = None,
) -> None:
    """
    Generate responses from `base` against each dataset in `dataset_configs`, analyse
    them with the response pipeline, and save results.

    `api`, when given, serves `base` as a hosted chat model over an OpenAI-compatible API
    (see src/eval/generate_api.py) and vllm is never imported; a hosted model has no local
    weights, so `weights_path` must be None. Otherwise the model is loaded by vllm from
    `weights_path` when given - a training run's epoch checkpoint (see src/run_grpo.py) -
    and from `base` otherwise, which is how an untrained model is evaluated.

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

    Each dataset's generations are saved before they are scored (see
    unscored_generations_path). If scoring fails, they are kept, the error is logged, and the
    remaining datasets still run - a later run then scores the saved generations instead of
    generating them again. Once every dataset has been tried, an error names any that failed.
    """
    if api is not None and weights_path is not None:
        raise ValueError(
            "API-served models have no local weights - the model key's registered "
            "path must be blank."
        )

    system_prompt = build_system_prompt(libraries_requirements_file, style=prompt_style)
    # the prompt-engineering styles add a sentence to each task instead
    task_suffix = TASK_SUFFIXES.get(prompt_style)

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
    else:
        # imported here, not at module scope, so API-only evaluation never needs vllm
        # (and therefore torch) installed at all
        import torch
        from vllm import LLM

        log("*** Loading model ***")
        # training checkpoints store fp32 weights, which vllm's "auto" dtype would serve
        # as fp16 - the wrong range for these models. every model here is natively
        # bf16, so ask for it explicitly and both cases are right.
        llm = LLM(
            model=weights_path or base,
            dtype="bfloat16",
            enforce_eager=debug,
            additional_config=dict(VLLM_ADDITIONAL_CONFIG),
            # one engine split across every gpu the job has - one, except for models too
            # big or slow for one (see EVAL_GPUS in scripts/submit_eval.sh)
            tensor_parallel_size=torch.cuda.device_count(),
        )

    # datasets whose scoring failed, reported once every dataset has been tried
    failed_datasets: list[str] = []

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
            task_suffix=task_suffix,
        )
        prompts = list(dataset["prompt"])

        # generations a previous run saved but couldn't score are scored, not regenerated
        output_path = output_paths[name]
        unscored_path = unscored_generations_path(output_path)
        responses = load_unscored_generations(
            unscored_path,
            n_prompts=len(prompts),
            n_samples=decoding_settings.get("n_samples", 1),
        )
        if responses is not None:
            log(f"Using the saved generations in {unscored_path}, not generating again")
        else:
            with log_timer("generation"):
                if api is not None:
                    responses = generate_api_responses(
                        model=base,
                        api=api,
                        prompts=prompts,
                        decoding_settings=decoding_settings,
                        max_workers=api.max_workers,
                    )
                else:
                    # deferred alongside the vllm import above, for the same reason
                    from src.eval.generate_vllm import generate_vllm_responses

                    responses = generate_vllm_responses(
                        llm=llm,
                        prompts=prompts,
                        decoding_settings=decoding_settings,
                    )
            # saved before scoring, so nothing that goes wrong while scoring loses them
            save_unscored_generations(responses, unscored_path)

        # an api request that failed holds no answer: ask again for it (and for any an
        # earlier run left), and score nothing while any request still fails, since a
        # failed request would count as a wrong answer
        if api is not None and count_failed_samples(responses):
            with log_timer("retrying failed requests"):
                responses = retry_failed_api_samples(
                    responses,
                    prompts,
                    model=base,
                    api=api,
                    decoding_settings=decoding_settings,
                )
            save_unscored_generations(responses, unscored_path)
            still_failed = count_failed_samples(responses)
            if still_failed:
                log(
                    f"{still_failed} request(s) still failed, so {name} is not scored - its "
                    f"generations are kept in {unscored_path}, and running the evaluation "
                    "again asks for the failed ones only"
                )
                failed_datasets.append(name)
                continue

        # score the responses the way this dataset's task is verified (execution for the
        # tested split, prose pick / import classification for the new splits); score_dataset
        # takes care of which evaluators want every sample vs just the first
        try:
            analysis, results = score_dataset(
                dataset=dataset,
                responses=responses,
                evaluator=evaluator,
                python_executable=python_executable,
                environment_root=environment_root,
                environment_cache_dir=environment_cache_dir,
            )
        except Exception:
            # keep the generations and carry on with the other datasets - a later run scores
            # them (see load_unscored_generations)
            log(traceback.format_exc())
            log(f"Scoring {name} failed - its generations are kept in {unscored_path}")
            failed_datasets.append(name)
            continue
        log(f"Scored {analysis['n_total']} responses ({analysis['task']})")

        # which build of the dataset was scored (library-usage rows stamp their own
        # dataset_version), so results are never mistaken for another build's - None for a
        # dataset whose rows carry no version. the dataset is a huggingface Dataset, or a
        # plain {column: values} dict in tests
        columns = getattr(dataset, "column_names", None) or list(dataset)
        dataset_version = (
            sorted(set(dataset["dataset_version"]) - {None})
            if "dataset_version" in columns
            else None
        )

        output = {
            "metadata": {
                "base": base,
                # which checkpoint produced this, or None for an untrained model
                "weights": weights_path,
                "provider": "api" if api is not None else "vllm",
                "dataset": name,
                "dataset_file": dataset_file,
                "dataset_version": dataset_version,
                "evaluator": evaluator,
                # the lure version that parsed and scored every response
                "lure_version": lure.__version__,
                "python_executable": python_executable,
                "environment_root": environment_root,
                "environment_cache_dir": environment_cache_dir,
                "model_dtype": None if api is not None else "auto",
                "system_prompt": system_prompt,
                "task_suffix": task_suffix,
                # lib/use need the requirements file to matter; the others apply regardless
                "prompt_style": (
                    prompt_style
                    if libraries_requirements_file
                    or prompt_style in FIXED_SYSTEM_PROMPTS
                    or prompt_style in TASK_SUFFIXES
                    else None
                ),
                "decoding_settings": decoding_settings,
                "debug": debug,
            },
            "analysis": analysis,
        }

        # two files per dataset, both committed as the experimental record: the analysis
        # above, and beside it one line per response - its full text and its scores (see
        # src/eval/scoring). the code is not stored separately: lure.parse_response
        # re-extracts it from the text if needed
        save_json(output, output_path)
        generations_path = output_path.with_suffix(".generations.jsonl")
        save_jsonl(results, generations_path)
        # the scored generations above replace the unscored copy
        unscored_path.unlink(missing_ok=True)
        log(f"Saved analysis to {output_path}")
        log(f"Saved every response and its scores to {generations_path}")

    if failed_datasets:
        raise RuntimeError(
            f"Not finished for {failed_datasets} - scoring failed, or api requests still "
            "failed. Their generations are kept beside their results (*.unscored.jsonl), "
            "and running the evaluation again retries any failed requests and scores them."
        )
