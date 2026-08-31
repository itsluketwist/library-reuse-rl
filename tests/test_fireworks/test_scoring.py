"""Tests for routing Fireworks JSONL through existing project scorers."""

import json
from pathlib import Path
from typing import Any

from fw_rlvr.scoring import score_generation_file


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON records for a scorer fixture."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_score_generation_file_uses_content_and_preserves_reasoning(
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

    result = score_generation_file(
        dataset_file=dataset,
        generations_file=generations,
        output_file=output,
        evaluator="library_usage",
    )

    assert result["analysis"]["n_external_lib_used"] == 0
    assert not result["results"][0]["external_lib_used"]
    assert "numpy" in result["results"][0]["fireworks"]["reasoning_content"]
    assert output.is_file()


def test_score_generation_file_rejects_incomplete_samples(tmp_path: Path) -> None:
    """Every compared sample must cover the identical source task set."""
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

    try:
        score_generation_file(
            dataset_file=dataset,
            generations_file=generations,
            output_file=tmp_path / "results.json",
            evaluator="library_usage",
        )
    except ValueError as error:
        assert "does not match dataset" in str(error)
    else:
        raise AssertionError("Incomplete generation samples must be rejected.")


def test_score_generation_file_preserves_multi_sample_task_grouping(
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

    result = score_generation_file(
        dataset_file=dataset,
        generations_file=generations,
        output_file=tmp_path / "results.json",
        evaluator="library_usage",
    )

    assert result["analysis"]["n_total"] == 2
    assert result["analysis"]["library_use_per_task"]["using_lib_rate"] == 1.0
    assert [row["sample_index"] for row in result["results"]] == [0, 1]
