"""Tests for the guarded Eval Protocol RFT launcher."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fw_rlvr.launch import build_command, launch_rft


def test_build_command_uses_installed_cli_flags() -> None:
    """The wrapper follows the inspected Eval Protocol CLI shape."""
    command = build_command(
        executable="eval-protocol",
        base_model="accounts/fireworks/models/qwen3-8b",
        extra_arguments=["--epochs", "2"],
    )

    assert command[:3] == ["eval-protocol", "create", "rft"]
    assert "--yes" in command
    assert "--skip-validation" in command
    assert "--dataset-jsonl" not in command
    assert command[-2:] == ["--epochs", "2"]


def test_dry_run_never_requires_or_invokes_cli(tmp_path: Path) -> None:
    """Dry-run validation is safe on machines without Eval Protocol installed."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "task"}], '
        '"ground_truth": {"task_id": "one"}}\n'
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    configuration = launch_rft(
        executable="definitely-not-installed",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=dataset,
        evaluator_directory=evaluator,
        record_path=tmp_path / "record.json",
        confirmed=False,
        dry_run=True,
    )

    assert configuration["base_model"].endswith("qwen3-8b")
    assert not (tmp_path / "record.json").exists()


def test_real_launch_requires_deliberate_confirmation(tmp_path: Path) -> None:
    """A non-dry launch is refused before checking or invoking the executable."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "task"}], '
        '"ground_truth": {"task_id": "one"}}\n'
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    with pytest.raises(RuntimeError, match="pass --yes"):
        launch_rft(
            executable="eval-protocol",
            base_model="accounts/fireworks/models/qwen3-8b",
            dataset=dataset,
            evaluator_directory=evaluator,
            record_path=tmp_path / "record.json",
            confirmed=False,
            dry_run=False,
        )


def test_completed_launch_captures_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful CLI output is retained with dashboard and resource identifiers."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "task"}], '
        '"ground_truth": {"task_id": "one"}}\n'
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    record = tmp_path / "record.json"
    monkeypatch.setattr("fw_rlvr.launch.shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr(
        "fw_rlvr.launch.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "https://app.fireworks.ai/dashboard/fine-tuning/job-1\n"
                "accounts/demo/fineTuningJobs/job-1\n"
            ),
            stderr="",
        ),
    )

    result = launch_rft(
        executable="eval-protocol",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=dataset,
        evaluator_directory=evaluator,
        record_path=record,
        confirmed=True,
        dry_run=False,
    )

    assert result["return_code"] == 0
    assert result["resources"] == ["accounts/demo/fineTuningJobs/job-1"]
    assert result["urls"][0].startswith("https://app.fireworks.ai/")
    assert record.is_file()


def test_failed_launch_is_recorded_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote CLI failures remain auditable rather than losing their output."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "task"}], '
        '"ground_truth": {"task_id": "one"}}\n'
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    record = tmp_path / "record.json"
    monkeypatch.setattr("fw_rlvr.launch.shutil.which", lambda _: "/bin/fake")
    monkeypatch.setattr(
        "fw_rlvr.launch.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="invalid configuration\n",
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        launch_rft(
            executable="eval-protocol",
            base_model="accounts/fireworks/models/qwen3-8b",
            dataset=dataset,
            evaluator_directory=evaluator,
            record_path=record,
            confirmed=True,
            dry_run=False,
        )

    assert '"return_code": 2' in record.read_text()
