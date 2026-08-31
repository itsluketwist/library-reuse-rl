"""Tests for matched and resumable Fireworks inference records."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fw_rlvr.inference import GenerationConfig, generate_dataset


class _Model:
    """Minimal serialisable SDK-model stand-in."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        """Return stored values in the shape used by the OpenAI SDK.

        Returns a shallow model dictionary.
        """
        assert mode == "json"

        def serialise(value: Any) -> Any:
            if isinstance(value, _Model):
                return value.model_dump(mode=mode)
            if isinstance(value, list):
                return [serialise(item) for item in value]
            return value

        return {key: serialise(value) for key, value in self.__dict__.items()}


class _Completions:
    """Capture requests and return deterministic completion choices."""

    def __init__(self, samples: int) -> None:
        self.samples = samples
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Model:
        """Build a mock Fireworks response.

        Returns a response with the requested number of choices.
        """
        self.requests.append(kwargs)
        choices = [
            _Model(
                index=index,
                finish_reason="stop",
                message=_Model(
                    content=f"answer-{index}",
                    reasoning_content=f"reasoning-{index}",
                ),
            )
            for index in range(self.samples)
        ]
        return _Model(
            id="response-1",
            model=kwargs["model"],
            created=123,
            choices=choices,
            usage=_Model(prompt_tokens=4, completion_tokens=2),
        )


def _client(samples: int) -> tuple[Any, _Completions]:
    """Build a structurally compatible fake chat client.

    Returns the client and request-capturing completions endpoint.
    """
    completions = _Completions(samples)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _rows() -> list[dict[str, Any]]:
    """Build one prepared prompt row.

    Returns a one-row Fireworks dataset.
    """
    return [
        {
            "messages": [{"role": "user", "content": "solve"}],
            "ground_truth": {"task_id": "task-1", "useful_libs": ["numpy"]},
        }
    ]


def _config(samples: int = 2) -> GenerationConfig:
    """Build a deterministic generation configuration.

    Returns the shared base/RFT decode configuration.
    """
    return GenerationConfig(
        model="accounts/test/models/model",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=256,
        seed=7,
        samples=samples,
    )


def test_generation_preserves_complete_and_separate_response_fields(
    tmp_path: Path,
) -> None:
    """The stable JSONL retains final content, reasoning, raw data, and config."""
    output = tmp_path / "generations.jsonl"
    client, completions = _client(samples=2)

    written = generate_dataset(
        _rows(),
        output,
        _config(),
        run_kind="base",
        client=client,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert written == 2
    assert records[0]["content"] == "answer-0"
    assert records[0]["reasoning_content"] == "reasoning-0"
    assert records[0]["generation_config"] == records[1]["generation_config"]
    assert records[0]["raw_response"]["id"] == "response-1"
    assert records[0]["usage"]["prompt_tokens"] == 4
    assert completions.requests[0]["extra_body"] == {"top_k": 20}
    assert completions.requests[0]["seed"] == 7


def test_resume_skips_complete_tasks(tmp_path: Path) -> None:
    """A completed task is not requested or duplicated during resume."""
    output = tmp_path / "generations.jsonl"
    first_client, _ = _client(samples=1)
    generate_dataset(
        _rows(),
        output,
        _config(samples=1),
        run_kind="base",
        client=first_client,
    )
    resumed_client, completions = _client(samples=1)

    written = generate_dataset(
        _rows(),
        output,
        _config(samples=1),
        run_kind="base",
        client=resumed_client,
        resume=True,
    )

    assert written == 0
    assert completions.requests == []
    assert len(output.read_text().splitlines()) == 1


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    """Inference never silently overwrites prior experimental generations."""
    output = tmp_path / "generations.jsonl"
    output.write_text("{}\n")
    client, _ = _client(samples=1)

    with pytest.raises(FileExistsError, match="--resume"):
        generate_dataset(
            _rows(),
            output,
            _config(samples=1),
            run_kind="rft",
            client=client,
        )


def test_partial_sample_set_is_rejected_on_resume(tmp_path: Path) -> None:
    """A partial task cannot be silently regenerated with mismatched randomness."""
    output = tmp_path / "generations.jsonl"
    output.write_text(json.dumps({"task_id": "task-1", "sample_index": 0}) + "\n")
    client, _ = _client(samples=2)

    with pytest.raises(ValueError, match="only some samples"):
        generate_dataset(
            _rows(),
            output,
            _config(samples=2),
            run_kind="rft",
            client=client,
            resume=True,
        )


def test_base_and_rft_runs_have_matched_requests_and_schema(tmp_path: Path) -> None:
    """Changing only run kind/model class cannot change inference mechanics."""
    config = _config(samples=1)
    base_client, base_completions = _client(samples=1)
    rft_client, rft_completions = _client(samples=1)
    base_output = tmp_path / "base.jsonl"
    rft_output = tmp_path / "rft.jsonl"

    generate_dataset(
        _rows(),
        base_output,
        config,
        run_kind="base",
        client=base_client,
    )
    generate_dataset(
        _rows(),
        rft_output,
        config,
        run_kind="rft",
        client=rft_client,
    )
    base_record = json.loads(base_output.read_text())
    rft_record = json.loads(rft_output.read_text())

    assert base_completions.requests == rft_completions.requests
    assert set(base_record) == set(rft_record)
    assert base_record["generation_config"] == rft_record["generation_config"]
    assert (base_record["run_kind"], rft_record["run_kind"]) == ("base", "rft")
