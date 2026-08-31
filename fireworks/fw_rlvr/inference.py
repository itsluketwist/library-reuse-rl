"""Shared Fireworks chat-completion inference with resumable JSONL output."""

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from openai import OpenAI


FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"


class _ChatCompletions(Protocol):
    """The small part of an OpenAI-compatible client used by this module."""

    def create(self, **kwargs: Any) -> Any:
        """Create one chat completion response."""


class _Chat(Protocol):
    """Protocol for the chat namespace of an OpenAI-compatible client."""

    completions: _ChatCompletions


class ChatClient(Protocol):
    """Protocol enabling the inference loop to use a mocked client in tests."""

    chat: _Chat


@dataclass(frozen=True)
class GenerationConfig:
    """Resolved decoding configuration shared by base and RFT inference."""

    model: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int | None = None
    samples: int = 1
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Model identifier must not be empty.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if self.samples < 1:
            raise ValueError("samples must be positive.")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative.")


def create_client(
    api_base: str = FIREWORKS_API_BASE,
    timeout_seconds: float = 120.0,
    max_retries: int = 2,
) -> ChatClient:
    """Build an OpenAI client configured for the Fireworks endpoint.

    Returns the configured client after validating `FIREWORKS_API_KEY`.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks inference.")
    return cast(
        ChatClient,
        OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout_seconds,
            max_retries=max_retries,
        ),
    )


def _model_dump(value: Any) -> dict[str, Any]:
    """Serialise an SDK model while retaining provider-specific extra fields.

    Returns a JSON-compatible dictionary.
    """
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot serialise SDK value of type {type(value).__name__}.")


def _request_parameters(config: GenerationConfig) -> dict[str, Any]:
    """Translate the shared configuration to Fireworks chat parameters.

    Returns keyword arguments excluding the per-row messages.
    """
    parameters: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "n": config.samples,
        "extra_body": {"top_k": config.top_k},
    }
    if config.seed is not None:
        parameters["seed"] = config.seed
    if config.reasoning_effort is not None:
        parameters["reasoning_effort"] = config.reasoning_effort
    return parameters


def _validate_input_row(
    row: dict[str, Any],
    source_index: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate one prepared dataset row.

    Returns its task identifier and chat messages.
    """
    messages = row.get("messages")
    ground_truth = row.get("ground_truth")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Input row {source_index} has no non-empty messages list.")
    if not isinstance(ground_truth, dict) or not ground_truth.get("task_id"):
        raise ValueError(f"Input row {source_index} has no ground_truth.task_id.")
    return str(ground_truth["task_id"]), messages


def _existing_keys(output_path: Path) -> set[tuple[str, int]]:
    """Read completed task/sample keys from an existing output file.

    Returns the keys, rejecting malformed or duplicate records.
    """
    keys: set[tuple[str, int]] = set()
    if not output_path.exists():
        return keys
    for line_number, line in enumerate(output_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = (str(record["task_id"]), int(record["sample_index"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed generation record at {output_path}:{line_number}."
            ) from error
        if key in keys:
            raise ValueError(f"Duplicate generation key in {output_path}: {key}")
        keys.add(key)
    return keys


def _append_jsonl(record: dict[str, Any], output_path: Path) -> None:
    """Append and flush one generation so interrupted runs remain resumable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def _generation_record(
    *,
    response: Any,
    choice: Any,
    row: dict[str, Any],
    task_id: str,
    source_index: int,
    run_kind: str,
    config: GenerationConfig,
) -> dict[str, Any]:
    """Normalise one Fireworks choice without discarding its raw SDK data.

    Returns the stable generation record written to JSONL.
    """
    response_data = _model_dump(response)
    choice_data = _model_dump(choice)
    message_data = _model_dump(choice.message)
    usage_data = _model_dump(getattr(response, "usage", None))
    sample_index = int(getattr(choice, "index", 0))
    return {
        "task_id": task_id,
        "source_index": source_index,
        "sample_index": sample_index,
        "run_kind": run_kind,
        "requested_model": config.model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "messages": row["messages"],
        "ground_truth": row.get("ground_truth"),
        "content": message_data.get("content"),
        "reasoning_content": message_data.get("reasoning_content"),
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage_data,
        "generation_config": asdict(config),
        "raw_choice": choice_data,
        "raw_response": response_data,
    }


def generate_dataset(
    rows: Iterable[dict[str, Any]],
    output_path: str | Path,
    config: GenerationConfig,
    run_kind: str,
    client: ChatClient,
    resume: bool = False,
) -> int:
    """Generate every prepared row and stream complete choices to JSONL.

    Returns the number of newly written generation records.
    """
    if run_kind not in {"base", "rft"}:
        raise ValueError("run_kind must be either 'base' or 'rft'.")
    resolved_output = Path(output_path)
    if resolved_output.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {resolved_output}. Use --resume to continue it."
        )
    completed = _existing_keys(resolved_output) if resume else set()
    written = 0
    request_parameters = _request_parameters(config)
    for source_index, row in enumerate(rows):
        task_id, messages = _validate_input_row(row, source_index)
        expected_keys = {(task_id, index) for index in range(config.samples)}
        if expected_keys <= completed:
            continue
        if expected_keys & completed:
            raise ValueError(
                f"Task {task_id} has only some samples in the resume file; "
                "remove or repair that task's records before resuming."
            )

        response = client.chat.completions.create(
            messages=messages,
            **request_parameters,
        )
        choices = list(response.choices)
        if len(choices) != config.samples:
            raise RuntimeError(
                f"Fireworks returned {len(choices)} choices for {task_id}; "
                f"expected {config.samples}."
            )
        for choice in choices:
            record = _generation_record(
                response=response,
                choice=choice,
                row=row,
                task_id=task_id,
                source_index=source_index,
                run_kind=run_kind,
                config=config,
            )
            key = (task_id, record["sample_index"])
            if key in completed:
                raise ValueError(f"Fireworks returned a duplicate choice key: {key}")
            _append_jsonl(record, resolved_output)
            completed.add(key)
            written += 1
    return written
