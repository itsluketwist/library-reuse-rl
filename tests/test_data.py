"""Tests for the training and evaluation dataset loaders."""

import json
from pathlib import Path

from src.data import load_eval_dataset, load_train_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write `records` to `path` as jsonl."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_skywork_record(prompt: str, inputs: list[str], outputs: list[str]) -> dict:
    """Build a single skywork-format record, as produced by data/download_skywork.py."""
    return {
        "prompt": prompt,
        "ground_truth": {"inputs": inputs, "outputs": outputs},
        "data_source": "test",
        "source": "skywork_or1",
    }


def _make_eval_record(prompt: str, inputs: list[str], outputs: list[str]) -> dict:
    """Build a single eval-format record, as produced by data/download_eval.py."""
    return {
        "prompt": prompt,
        "inputs": inputs,
        "expected_outputs": outputs,
        "source": "test",
    }


class TestLoadTrainDataset:
    """Tests for load_train_dataset."""

    def test_loads_train_and_val_splits(self, tmp_path: Path):
        """Both train.jsonl and val.jsonl are loaded into their own DatasetDict split."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [_make_skywork_record("train prompt", ["1\n"], ["1\n"])],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [_make_skywork_record("val prompt", ["2\n"], ["2\n"])],
        )

        dataset = load_train_dataset(tmp_path, system_prompt=None)

        assert set(dataset.keys()) == {"train", "val"}
        assert len(dataset["train"]) == 1
        assert len(dataset["val"]) == 1

    def test_flattens_ground_truth_into_top_level_inputs_and_expected_outputs(
        self, tmp_path: Path
    ):
        """ground_truth.inputs/outputs are renamed to the columns evaluate_record reads."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [_make_skywork_record("solve this", ["3\n4\n"], ["7\n"])],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [_make_skywork_record("val prompt", ["1\n"], ["1\n"])],
        )

        dataset = load_train_dataset(tmp_path, system_prompt=None)
        example = dataset["train"][0]

        assert example["inputs"] == ["3\n4\n"]
        assert example["expected_outputs"] == ["7\n"]

    def test_builds_chat_messages_prompt_without_system_prompt(self, tmp_path: Path):
        """With no system_prompt, the prompt is just a single user message."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [_make_skywork_record("solve this", ["1\n"], ["1\n"])],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [_make_skywork_record("val prompt", ["1\n"], ["1\n"])],
        )

        dataset = load_train_dataset(tmp_path, system_prompt=None)
        example = dataset["train"][0]

        assert example["prompt"] == [{"role": "user", "content": "solve this"}]

    def test_builds_chat_messages_prompt_with_system_prompt(self, tmp_path: Path):
        """With a system_prompt, it's prepended as its own message."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [_make_skywork_record("solve this", ["1\n"], ["1\n"])],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [_make_skywork_record("val prompt", ["1\n"], ["1\n"])],
        )

        dataset = load_train_dataset(tmp_path, system_prompt="be helpful")
        example = dataset["train"][0]

        assert example["prompt"] == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "solve this"},
        ]

    def test_truncates_train_and_val_splits(self, tmp_path: Path):
        """train_n/val_n cap each split to that many rows."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [
                _make_skywork_record(f"train prompt {i}", ["1\n"], ["1\n"])
                for i in range(5)
            ],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [
                _make_skywork_record(f"val prompt {i}", ["1\n"], ["1\n"])
                for i in range(5)
            ],
        )

        dataset = load_train_dataset(
            tmp_path,
            system_prompt=None,
            train_n=2,
            val_n=3,
        )

        assert len(dataset["train"]) == 2
        assert len(dataset["val"]) == 3

    def test_train_n_larger_than_split_keeps_full_split(self, tmp_path: Path):
        """A truncation larger than the split doesn't error, and keeps every row."""
        _write_jsonl(
            tmp_path / "train.jsonl",
            [_make_skywork_record("train prompt", ["1\n"], ["1\n"])],
        )
        _write_jsonl(
            tmp_path / "val.jsonl",
            [_make_skywork_record("val prompt", ["1\n"], ["1\n"])],
        )

        dataset = load_train_dataset(
            tmp_path,
            system_prompt=None,
            train_n=10,
            val_n=10,
        )

        assert len(dataset["train"]) == 1
        assert len(dataset["val"]) == 1


class TestLoadEvalDataset:
    """Tests for load_eval_dataset."""

    def test_loads_flat_schema_without_renaming(self, tmp_path: Path):
        """Eval records are already flat, so inputs/expected_outputs pass through untouched."""
        dataset_file = tmp_path / "eval.jsonl"
        _write_jsonl(
            dataset_file,
            [_make_eval_record("solve this", ["3\n4\n"], ["7\n"])],
        )

        dataset = load_eval_dataset(dataset_file, system_prompt=None)
        example = dataset[0]

        assert example["inputs"] == ["3\n4\n"]
        assert example["expected_outputs"] == ["7\n"]

    def test_builds_chat_messages_prompt_without_system_prompt(self, tmp_path: Path):
        """With no system_prompt, the prompt is just a single user message."""
        dataset_file = tmp_path / "eval.jsonl"
        _write_jsonl(
            dataset_file,
            [_make_eval_record("solve this", ["1\n"], ["1\n"])],
        )

        dataset = load_eval_dataset(dataset_file, system_prompt=None)
        example = dataset[0]

        assert example["prompt"] == [{"role": "user", "content": "solve this"}]

    def test_builds_chat_messages_prompt_with_system_prompt(self, tmp_path: Path):
        """With a system_prompt, it's prepended as its own message."""
        dataset_file = tmp_path / "eval.jsonl"
        _write_jsonl(
            dataset_file,
            [_make_eval_record("solve this", ["1\n"], ["1\n"])],
        )

        dataset = load_eval_dataset(dataset_file, system_prompt="be helpful")
        example = dataset[0]

        assert example["prompt"] == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "solve this"},
        ]
