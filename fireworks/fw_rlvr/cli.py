"""Shared command-line parsing for base and RFT Fireworks inference."""

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config

from .inference import (
    FIREWORKS_API_BASE,
    GenerationConfig,
    create_client,
    generate_dataset,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load object records from a JSONL file.

    Returns parsed records in file order.
    """
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        records.append(value)
    return records


def build_parser(run_kind: str) -> argparse.ArgumentParser:
    """Build the common inference parser with a run-specific description.

    Returns the configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Run matched Fireworks {run_kind.upper()} inference.",
    )
    parser.add_argument("--model", required=True, help="Fireworks model/deployment ID.")
    parser.add_argument("--input", required=True, type=Path, help="Prepared JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Generation JSONL.")
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max"],
        default=None,
    )
    parser.add_argument("--api-base", default=FIREWORKS_API_BASE)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete task/sample keys in an existing output.",
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
        samples=args.samples,
        reasoning_effort=args.reasoning_effort,
    )


def run_cli(run_kind: str) -> None:
    """Run base or RFT inference through the exact same implementation."""
    args = build_parser(run_kind).parse_args()
    config = resolve_generation_config(args)
    rows = _load_jsonl(args.input)
    client = create_client(
        api_base=args.api_base,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    written = generate_dataset(
        rows=rows,
        output_path=args.output,
        config=config,
        run_kind=run_kind,
        client=client,
        resume=args.resume,
    )
    print(f"Wrote {written} generation record(s) to {args.output}.")
