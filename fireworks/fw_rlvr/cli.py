"""Shared generation-only command line for base and RFT Fireworks inference."""

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.data import load_eval_dataset
from src.prompts import build_system_prompt, library_prompt_suffix
from src.utils.config import load_full_yaml, load_yaml_config

from .inference import (
    FIREWORKS_API_BASE,
    GenerationConfig,
    create_client,
    generate_dataset,
)


DEFAULT_DATASETS = ("library_usage", "libreusebench")


def _dataset_names(value: str) -> list[str]:
    """Parse and validate a comma-separated dataset selection.

    Returns unique dataset names in requested order.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one inference dataset is required.")
    if len(names) != len(set(names)):
        raise ValueError(f"Inference datasets contain duplicates: {names}")
    return names


def model_output_name(model: str, override: str | None = None) -> str:
    """Choose the safe directory name used beneath ``output/fireworks``.

    Returns an explicit name or the terminal model/deployment resource segment.
    """
    name = override or model.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Invalid Fireworks output model name: {name!r}")
    return name


def inference_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    decoding_profile: str,
    prompt_suffix: str,
) -> Path:
    """Build one raw-generation output path.

    Returns a path beneath ``output_root/model_name``.
    """
    if Path(dataset_name).name != dataset_name:
        raise ValueError(f"Invalid dataset name for output: {dataset_name!r}")
    return (
        output_root
        / model_name
        / f"{dataset_name}-{decoding_profile}{prompt_suffix}.jsonl"
    )


def _prepare_inference_rows(
    dataset: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only task IDs and main-repository chat prompts for generation.

    Returns rows accepted by the Fireworks inference loop.
    """
    rows: list[dict[str, Any]] = []
    for source_index, record in enumerate(dataset):
        task_id = record.get("task_id")
        messages = record.get("prompt")
        if not task_id:
            raise ValueError(f"Dataset row {source_index} has no task_id.")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Dataset row {source_index} has no prepared chat prompt.",
            )
        rows.append(
            {
                "task_id": str(task_id),
                "messages": messages,
            },
        )
    return rows


def build_parser(run_kind: str) -> argparse.ArgumentParser:
    """Build the common generation-only parser.

    Returns the configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Generate raw Fireworks {run_kind.upper()} responses.",
    )
    parser.add_argument("--model", required=True, help="Fireworks deployment ID.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Output directory name; defaults to the deployment ID's final segment.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated names from config/eval_datasets.yaml.",
    )
    parser.add_argument(
        "--eval-datasets-config",
        default="config/eval_datasets.yaml",
        help="YAML mapping dataset names to source paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/fireworks"),
        help="Root for raw generation files.",
    )
    parser.add_argument(
        "--decoding-config",
        default="config/training.yaml",
        help="YAML file containing eval_decoding_profiles.",
    )
    parser.add_argument(
        "--decoding-profile",
        default="qwen3",
        help="Named eval_decoding_profiles entry.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["lib", "use", "std", "base"],
        default="base",
    )
    parser.add_argument("--libraries-requirements-file", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max"],
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset row limit for a smoke run.",
    )
    parser.add_argument("--api-base", default=FIREWORKS_API_BASE)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete task/sample keys in existing outputs.",
    )
    return parser


def resolve_generation_config(args: argparse.Namespace) -> GenerationConfig:
    """Merge a repository decoding profile with explicit CLI overrides.

    Returns the complete Fireworks generation configuration.
    """
    profile = load_yaml_config(
        config_file=args.decoding_config,
        key=args.decoding_profile,
        section="eval_decoding_profiles",
    )
    return GenerationConfig(
        model=args.model,
        temperature=(
            args.temperature
            if args.temperature is not None
            else float(profile["temperature"])
        ),
        top_p=args.top_p if args.top_p is not None else float(profile["top_p"]),
        top_k=args.top_k if args.top_k is not None else int(profile.get("top_k", 0)),
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else int(profile["max_new_tokens"])
        ),
        seed=args.seed,
        samples=(
            args.samples
            if args.samples is not None
            else int(profile.get("n_samples", 1))
        ),
        reasoning_effort=args.reasoning_effort,
    )


def run_cli(run_kind: str) -> None:
    """Generate named evaluation datasets without running any scorer."""
    args = build_parser(run_kind).parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    config = resolve_generation_config(args)
    dataset_names = _dataset_names(args.datasets)
    available_datasets = load_full_yaml(args.eval_datasets_config)
    system_prompt = build_system_prompt(
        args.libraries_requirements_file,
        style=args.prompt_style,
    )
    prompt_suffix = library_prompt_suffix(
        args.libraries_requirements_file,
        style=args.prompt_style,
    )
    output_model_name = model_output_name(args.model, args.model_name)

    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, Path] = {}
    for dataset_name in dataset_names:
        try:
            dataset_config = available_datasets[dataset_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown dataset {dataset_name!r}; available: "
                f"{sorted(available_datasets)}",
            ) from error
        if not isinstance(dataset_config, dict) or "path" not in dataset_config:
            raise ValueError(f"Dataset {dataset_name!r} must define a path.")
        dataset_path = Path(dataset_config["path"])
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} does not exist: {dataset_path}",
            )
        dataset = load_eval_dataset(
            dataset_file=dataset_path,
            system_prompt=system_prompt,
            n=args.limit,
        )
        rows_by_dataset[dataset_name] = _prepare_inference_rows(dataset)
        output_path = inference_output_path(
            output_root=args.output_root,
            model_name=output_model_name,
            dataset_name=dataset_name,
            decoding_profile=args.decoding_profile,
            prompt_suffix=prompt_suffix,
        )
        if output_path.exists() and not args.resume:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --resume to continue it.",
            )
        outputs[dataset_name] = output_path

    # create the client only after every dataset and output has passed validation.
    client = create_client(
        api_base=args.api_base,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    for dataset_name in dataset_names:
        written = generate_dataset(
            rows=rows_by_dataset[dataset_name],
            output_path=outputs[dataset_name],
            config=config,
            run_kind=run_kind,
            dataset_name=dataset_name,
            client=client,
            resume=args.resume,
        )
        print(f"Wrote {written} generation record(s) to {outputs[dataset_name]}.")
