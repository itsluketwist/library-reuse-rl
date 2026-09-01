"""Safely launch and record an Eval Protocol RFT creation command."""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


_URL_PATTERN = re.compile(r"https?://[^\s\]\[()<>\"']+")
_RESOURCE_PATTERN = re.compile(
    r"accounts/[A-Za-z0-9_.-]+/"
    r"(?:fineTuningJobs|reinforcementFineTuningJobs|models|evaluatorsV2)/"
    r"[A-Za-z0-9_.-]+"
)


def build_command(
    *,
    executable: str,
    base_model: str,
    eval_protocol_entrypoint: str | None = None,
    extra_arguments: Sequence[str] = (),
) -> list[str]:
    """Build the installed Eval Protocol single-turn RFT command.

    Returns an argument list suitable for a non-shell subprocess.
    """
    command_prefix = [executable]
    if eval_protocol_entrypoint is not None:
        command_prefix.append(eval_protocol_entrypoint)
    return [
        *command_prefix,
        "create",
        "rft",
        "--yes",
        # the full dataset is selected through FIREWORKS_RFT_DATASET. skipping
        # cli validation prevents it from generating hundreds of local rollouts.
        "--skip-validation",
        "--base-model",
        base_model,
        *extra_arguments,
    ]


def _validate_dataset(dataset: Path) -> int:
    """Validate prepared RFT JSONL before bypassing CLI rollout validation.

    Returns the number of unique, structurally valid rows.
    """
    task_ids: set[str] = set()
    row_count = 0
    for line_number, line in enumerate(dataset.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {dataset}:{line_number}.") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {dataset}:{line_number}.")
        messages = row.get("messages")
        ground_truth = row.get("ground_truth")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing messages at {dataset}:{line_number}.")
        if not isinstance(ground_truth, dict) or not ground_truth.get("task_id"):
            raise ValueError(
                f"Missing ground_truth.task_id at {dataset}:{line_number}."
            )
        task_id = str(ground_truth["task_id"])
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id in {dataset}: {task_id}")
        task_ids.add(task_id)
        row_count += 1
    if not row_count:
        raise ValueError(f"RFT dataset is empty: {dataset}")
    return row_count


def _extract_identifiers(output: str) -> dict[str, list[str]]:
    """Extract dashboard URLs and Fireworks resource names from CLI output.

    Returns sorted, de-duplicated identifier lists.
    """
    return {
        "urls": sorted(set(_URL_PATTERN.findall(output))),
        "resources": sorted(set(_RESOURCE_PATTERN.findall(output))),
    }


def _write_record(record: dict[str, Any], path: Path) -> None:
    """Write one launch record atomically as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def launch_rft(
    *,
    executable: str,
    base_model: str,
    dataset: Path,
    evaluator_directory: Path,
    record_path: Path,
    confirmed: bool,
    dry_run: bool,
    eval_protocol_entrypoint: Path | None = None,
    extra_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, optionally run, and record an Eval Protocol RFT launch.

    Returns the printed configuration for a dry run or the completed launch record.
    """
    if not dataset.is_file():
        raise FileNotFoundError(f"RFT dataset does not exist: {dataset}")
    if not evaluator_directory.is_dir():
        raise FileNotFoundError(
            f"Evaluator directory does not exist: {evaluator_directory}"
        )
    row_count = _validate_dataset(dataset)
    discovered_executable = shutil.which(executable)
    resolved_executable = (
        # keep virtual-environment interpreter symlinks intact. resolving the
        # symlink would run the base interpreter without the venv packages.
        str(Path(discovered_executable).absolute())
        if discovered_executable is not None
        else executable
    )
    command = build_command(
        executable=resolved_executable,
        base_model=base_model,
        eval_protocol_entrypoint=(
            str(eval_protocol_entrypoint.resolve())
            if eval_protocol_entrypoint is not None
            else None
        ),
        extra_arguments=extra_arguments,
    )
    configuration: dict[str, Any] = {
        "base_model": base_model,
        "command": command,
        "dataset": str(dataset.resolve()),
        "dataset_rows": row_count,
        "evaluator_directory": str(evaluator_directory.resolve()),
        "record_path": str(record_path.resolve()),
    }
    print(json.dumps(configuration, indent=2, sort_keys=True))
    if dry_run:
        return configuration
    if not confirmed:
        raise RuntimeError(
            "RFT launch refused. Review the printed configuration and pass --yes."
        )
    if discovered_executable is None:
        raise FileNotFoundError(
            f"Could not find {executable!r}; install fireworks/requirements.txt."
        )

    started_at = datetime.now(UTC).isoformat()
    environment = dict(os.environ)
    environment["FIREWORKS_RFT_DATASET"] = str(dataset.resolve())
    completed = subprocess.run(
        command,
        cwd=evaluator_directory,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    combined_output = "\n".join(
        value for value in (completed.stdout, completed.stderr) if value
    )
    record = {
        **configuration,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        **_extract_identifiers(combined_output),
    }
    _write_record(record, record_path)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return record


def build_parser() -> argparse.ArgumentParser:
    """Build the guarded RFT launcher CLI.

    Returns the configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--evaluator-directory",
        type=Path,
        default=Path("fireworks/rft/evaluator"),
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("fireworks/rft/runs/latest-launch.json"),
    )
    parser.add_argument(
        "--executable",
        default=".venv-eval/bin/python",
    )
    parser.add_argument(
        "--eval-protocol-entrypoint",
        type=Path,
        default=Path("fireworks/fw_rlvr/eval_protocol_compat.py"),
        help=(
            "Local Eval Protocol entrypoint with the pinned release's Literal-parser "
            "compatibility fix."
        ),
    )
    parser.add_argument(
        "--extra-argument",
        action="append",
        default=[],
        help="Repeat to pass a current Eval Protocol option through unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without requiring the CLI or creating a job.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that creating the remote RFT job is intentional.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the guarded RFT launcher."""
    args = build_parser().parse_args()
    launch_rft(
        executable=args.executable,
        base_model=args.base_model,
        dataset=args.dataset,
        evaluator_directory=args.evaluator_directory,
        record_path=args.record,
        confirmed=args.yes,
        dry_run=args.dry_run,
        eval_protocol_entrypoint=args.eval_protocol_entrypoint,
        extra_arguments=args.extra_argument,
    )
