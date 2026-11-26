"""Tests for where training runs and evaluations write their results."""

from pathlib import Path

import pytest

from src.utils import paths
from src.utils.paths import (
    baseline_dir,
    debug_output_dir,
    model_output_dir,
    moved_baseline_result,
    training_run_dir,
)


class TestModelOutputDir:
    """model_output_dir files eval results by whether a model is base or trained."""

    def test_a_base_model_goes_in_baselines(self) -> None:
        settings = {"base": "Qwen/Qwen3-8B", "path": None}

        assert model_output_dir("qwen3-8b", settings) == Path(
            "output/baselines/qwen3-8b"
        )

    def test_an_epoch_goes_in_its_training_runs_folder(self) -> None:
        """Beside the run's grpo/ metrics, in the folder of the base model it trained."""
        settings = {
            "base": "Qwen/Qwen3-8B",
            "path": "/checkpoints/qwen3-8b-a1b2/checkpoint-28",
            "properties": {
                "run_id": "qwen3-8b-a1b2",
                "base_model": "qwen3-8b",
                "epoch": 1,
            },
        }

        assert model_output_dir("qwen3-8b-a1b2-epoch1", settings) == Path(
            "output/qwen3-8b/qwen3-8b-a1b2/epoch1"
        )

    def test_an_epoch_of_a_written_up_run_goes_in_its_iterations_folder(self) -> None:
        """Runs grouped into a numbered iteration live under output/{iteration}/."""
        settings = {
            "base": "Qwen/Qwen3-8B",
            "path": "/checkpoints/qwen3-8b-a1b2/checkpoint-28",
            "properties": {
                "run_id": "qwen3-8b-a1b2",
                "base_model": "qwen3-8b",
                "epoch": 1,
                "iteration": "01",
            },
        }

        assert model_output_dir("qwen3-8b-a1b2-epoch1", settings) == Path(
            "output/01/qwen3-8b-a1b2/epoch1"
        )

    def test_an_older_entry_is_placed_from_its_key(self) -> None:
        """Entries registered before run_id/base_model were recorded still land right."""
        settings = {
            "base": "Qwen/Qwen3-8B",
            "path": "/checkpoints/qwen3-8b-ef47/checkpoint-56",
            "properties": {"epoch": 2},
        }

        assert model_output_dir("qwen3-8b-ef47-epoch2", settings) == Path(
            "output/qwen3-8b/qwen3-8b-ef47/epoch2"
        )

    def test_an_unrecognisable_trained_entry_raises(self) -> None:
        settings = {"base": "Qwen/Qwen3-8B", "path": "/checkpoints/x", "properties": {}}

        with pytest.raises(ValueError, match="training run"):
            model_output_dir("mystery", settings)


def test_training_metrics_and_epochs_share_the_run_folder() -> None:
    run = training_run_dir(base_model="qwen3-8b", run_id="qwen3-8b-a1b2")

    assert run == Path("output/qwen3-8b/qwen3-8b-a1b2")
    assert baseline_dir("qwen3-8b") == Path("output/baselines/qwen3-8b")


def test_debug_results_mirror_the_normal_layout_under_debug() -> None:
    """Debug evals never land in (and so can't be mistaken for) real result folders."""
    assert debug_output_dir(baseline_dir("olmo3-7b")) == Path(
        "output/debug/baselines/olmo3-7b"
    )
    assert debug_output_dir(training_run_dir("qwen3-debug", "run")) == Path(
        "output/debug/qwen3-debug/run"
    )


def test_a_moved_baseline_result_is_found_but_a_debug_one_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results moved into an iteration's folder still exist; debug results never count."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    name = "libreusebench_v2_control-default_16k-base.json"
    assert moved_baseline_result(model="qwen3-8b", file_name=name) is None

    debug = tmp_path / "debug" / "baselines" / "qwen3-8b" / name
    debug.parent.mkdir(parents=True)
    debug.write_text("{}")
    assert moved_baseline_result(model="qwen3-8b", file_name=name) is None

    moved = tmp_path / "02" / "baselines" / "qwen3-8b" / name
    moved.parent.mkdir(parents=True)
    moved.write_text("{}")
    assert moved_baseline_result(model="qwen3-8b", file_name=name) == moved
    assert moved_baseline_result(model="olmo3-7b", file_name=name) is None
