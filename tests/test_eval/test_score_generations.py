"""Tests for scoring externally generated responses (e.g. from fireworks/) against a dataset."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.eval.score_generations import score_generations


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON records for a scorer fixture."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a JSON array of records for a scorer fixture."""
    path.write_text(json.dumps(records))


def test_score_generations_uses_content_and_keeps_generation_provenance(
    tmp_path: Path,
) -> None:
    """Downstream metrics ignore reasoning while retaining it for analysis."""
    dataset = tmp_path / "dataset.jsonl"
    generations = tmp_path / "generations.jsonl"
    output = tmp_path / "results.json"
    _write_jsonl(
        dataset,
        [
            {
                "task_id": "task-1",
                "prompt": "Write code.",
                "task_type": "library_usage",
                "category_id": "arrays",
                "local_modules": [],
                "useful_libs": ["numpy"],
                "phrasing_mode": "bare",
            }
        ],
    )
    _write_jsonl(
        generations,
        [
            {
                "task_id": "task-1",
                "sample_index": 0,
                "run_kind": "base",
                "requested_model": "accounts/fireworks/models/qwen3-8b",
                "content": "```python\nprint(sum(values))\n```",
                "reasoning_content": "import numpy as np; np.sum(values)",
                "finish_reason": "stop",
                "usage": {"completion_tokens": 4},
                "generation_config": {"temperature": 0.6},
            }
        ],
    )

    result = score_generations(
        dataset_file=dataset,
        generations_file=generations,
        output_file=output,
        evaluator="library_usage",
    )

    assert result["analysis"]["n_external_lib_used"] == 0
    assert not result["results"][0]["external_lib_used"]
    assert result["results"][0]["sample_index"] == 0
    assert "numpy" in result["results"][0]["generation"]["reasoning_content"]
    assert output.is_file()


def test_score_generations_accepts_json_generations_file(tmp_path: Path) -> None:
    """A .json (array) generations file scores the same as an equivalent .jsonl one."""
    dataset = tmp_path / "dataset.jsonl"
    generations = tmp_path / "generations.json"
    _write_jsonl(
        dataset,
        [
            {
                "task_id": "task-1",
                "prompt": "Write code.",
                "task_type": "library_usage",
                "category_id": "arrays",
                "local_modules": [],
                "useful_libs": ["numpy"],
                "phrasing_mode": "bare",
            }
        ],
    )
    _write_json(
        generations,
        [
            {
                "task_id": "task-1",
                "sample_index": 0,
                "content": "```python\nimport numpy as np\nprint(np.sum(values))\n```",
                "finish_reason": "stop",
            }
        ],
    )

    result = score_generations(
        dataset_file=dataset,
        generations_file=generations,
        output_file=tmp_path / "results.json",
        evaluator="library_usage",
    )

    assert result["analysis"]["n_external_lib_used"] == 1
    assert result["results"][0]["external_lib_used"]


def test_score_generations_rejects_incomplete_samples(tmp_path: Path) -> None:
    """Every scored sample must cover the identical source task set."""
    dataset = tmp_path / "dataset.jsonl"
    generations = tmp_path / "generations.jsonl"
    rows = [
        {
            "task_id": task_id,
            "prompt": "Write code.",
            "task_type": "library_usage",
            "category_id": "arrays",
            "local_modules": [],
            "useful_libs": ["numpy"],
            "phrasing_mode": "bare",
        }
        for task_id in ("task-1", "task-2")
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        generations,
        [
            {
                "task_id": "task-1",
                "sample_index": 0,
                "content": "```python\nprint(1)\n```",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing generations for"):
        score_generations(
            dataset_file=dataset,
            generations_file=generations,
            output_file=tmp_path / "results.json",
            evaluator="library_usage",
        )


def test_score_generations_preserves_multi_sample_task_grouping(
    tmp_path: Path,
) -> None:
    """Library-use scoring receives every sampled response for each source task."""
    dataset = tmp_path / "dataset.jsonl"
    generations = tmp_path / "generations.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "task_id": "task-1",
                "prompt": "Write code.",
                "task_type": "library_usage",
                "category_id": "arrays",
                "local_modules": [],
                "useful_libs": ["numpy"],
                "phrasing_mode": "bare",
            }
        ],
    )
    _write_jsonl(
        generations,
        [
            {
                "task_id": "task-1",
                "sample_index": 0,
                "content": "```python\nprint(sum(values))\n```",
            },
            {
                "task_id": "task-1",
                "sample_index": 1,
                "content": "```python\nimport numpy as np\nprint(np.sum(values))\n```",
            },
        ],
    )

    result = score_generations(
        dataset_file=dataset,
        generations_file=generations,
        output_file=tmp_path / "results.json",
        evaluator="library_usage",
    )

    assert result["analysis"]["n_total"] == 2
    assert result["analysis"]["library_use_per_task"]["using_lib_rate"] == 1.0
    assert [row["sample_index"] for row in result["results"]] == [0, 1]


def test_libreusebench_provenance_uses_its_sample_idx_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LibReuseBench's idx is a sample index, and each sample keeps its own token usage."""
    from src.eval import score_generations as scoring_module

    dataset = tmp_path / "dataset.jsonl"
    generations = tmp_path / "generations.jsonl"
    _write_jsonl(dataset, [{"task_id": "one", "prompt": "Write code."}])
    _write_jsonl(
        generations,
        [
            {
                "task_id": "one",
                "sample_index": index,
                "content": "print(1)",
                "usage": {"completion_tokens": 10 + index},
            }
            for index in range(5)
        ],
    )

    def score(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return {}, [{"task_id": "one", "idx": index} for index in range(5)]

    monkeypatch.setattr(scoring_module, "score_dataset", score)
    result = score_generations(
        dataset, generations, tmp_path / "result.json", "libreusebench"
    )
    assert [row["sample_index"] for row in result["results"]] == list(range(5))
    assert [
        row["generation"]["usage"]["completion_tokens"] for row in result["results"]
    ] == list(range(10, 15))
