"""Tests for stage 05 of the benchmark build: the splits and the task ids.

The stage is a script (benchmark/pipeline/05_build.py), so it is loaded by path.
"""

import importlib.util
import sys
from pathlib import Path

import build
import pytest
import review

from src.utils.io import load_jsonl


_SCRIPT = Path(__file__).parents[2] / "benchmark" / "pipeline" / "05_build.py"
_spec = importlib.util.spec_from_file_location("build_stage", _SCRIPT)
assert _spec is not None and _spec.loader is not None
build_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_stage)


def _task(
    number: int,
    library: str = "pandas",
    value: int = 30,
    tier: str = "high",
    category: str | None = None,
) -> dict:
    """A kept reuse-track task, with the fields the build reads."""
    task = {
        "seed_id": f"BigCodeBench/{number}",
        "seed_libs": [library],
        "category": category or f"category-{number}",
        "domain": "other",
        "prompt": f"Write a function `solve_{number}(x)`.",
        "function_name": f"solve_{number}",
        "test_inputs": ["[1]"],
        "expected_outputs": ["1"],
        "stdlib_solution": "def f(x):\n    return x\n",
        "extlib_solution": f"import {library}\n\n\ndef f(x):\n    return x\n",
        "gold_extlib_packages": [f"{library}==1.0"],
        "primary_library": library,
        "reuse_value": value,
        "reuse_tier": tier,
        "stdlib_lloc": value + 10,
        "extlib_lloc": 10,
    }
    return task | {"task_hash": build.task_hash(task)}


def _control_task(
    number: int,
) -> dict:
    """A kept control-track task: it has no library solution, and so no reuse value."""
    task = _task(number, category=f"control-{number}")
    for field in (
        "extlib_solution",
        "gold_extlib_packages",
        "primary_library",
        "reuse_value",
        "reuse_tier",
        "stdlib_lloc",
        "extlib_lloc",
    ):
        del task[field]
    return task | {"seed_libs": []}


def _ids(
    tasks: list[dict],
) -> list[str]:
    """The seed ids of some tasks, in order."""
    return [task["seed_id"] for task in tasks]


class TestBuildSplits:
    """Every task goes to the split its BigCodeBench seed decides, and each gets a unique id."""

    @pytest.fixture
    def splits(self) -> dict[str, list[dict]]:
        """Splits from 30 high, 4 medium and 6 low reuse tasks over 10 libraries, and 5 control."""
        reuse = [
            _task(number, library=f"lib{number % 10}", value=20 + number)
            for number in range(30)
        ]
        reuse += [
            _task(100 + number, library=f"lib{number}", value=8, tier="medium")
            for number in range(4)
        ]
        reuse += [
            _task(200 + number, library=f"lib{number % 3}", value=2, tier="low")
            for number in range(6)
        ]
        control = [_control_task(300 + number) for number in range(5)]
        return build_stage.build_splits(reuse_tasks=reuse, control_tasks=control)

    def test_every_task_ships_in_its_track_split(self, splits) -> None:
        # nothing is capped, held out or left out by tier
        assert {name: len(rows) for name, rows in splits.items()} == {
            "test": 40,
            "control": 5,
        }

    def test_the_tier_is_a_field_not_a_split(self, splits) -> None:
        tiers = [row["reuse_tier"] for row in splits["test"]]
        assert {tier: tiers.count(tier) for tier in set(tiers)} == {
            "high": 30,
            "medium": 4,
            "low": 6,
        }

    def test_task_ids(self, splits) -> None:
        assert [row["task_id"] for row in splits["test"]][:2] == [
            "LRB2/T000",
            "LRB2/T001",
        ]
        assert splits["test"][-1]["task_id"] == "LRB2/T039"
        assert splits["control"][0]["task_id"] == "LRB2/C000"
        ids = [row["task_id"] for rows in splits.values() for row in rows]
        assert len(set(ids)) == len(ids)

    def test_row_fields(self, splits) -> None:
        row = splits["test"][0]
        assert list(row) == [
            "task_id",
            "category",
            "domain",
            "prompt",
            "function_name",
            "test_inputs",
            "expected_outputs",
            "ground_truth",
            "reuse_value",
            "reuse_tier",
            "stdlib_lloc",
            "extlib_lloc",
            "source",
        ]
        assert list(row["ground_truth"]) == [
            "gold_stdlib",
            "gold_extlib",
            "gold_extlib_packages",
        ]
        assert list(row["source"]) == ["seed_id", "seed_libs", "primary_library"]

    def test_a_control_row_has_no_library_fields(self, splits) -> None:
        row = splits["control"][0]
        assert list(row["ground_truth"]) == ["gold_stdlib"]
        assert "reuse_value" not in row
        assert list(row["source"]) == ["seed_id", "seed_libs"]

    def test_the_build_is_reproducible(self, splits) -> None:
        tasks = [_task(number, library=f"lib{number % 10}") for number in range(30)]
        first = build_stage.build_splits(reuse_tasks=tasks, control_tasks=[])
        second = build_stage.build_splits(reuse_tasks=tasks[::-1], control_tasks=[])
        assert first == second


def test_a_pilot_build_ships_every_task_under_its_own_ids() -> None:
    """A pilot build ships the same splits, under ids that can't be the benchmark's."""
    reuse = [_task(number, value=30) for number in range(6)]  # six tasks, all pandas
    reuse += [_task(10, value=8, tier="medium"), _task(11, value=2, tier="low")]
    control = [_control_task(20), _control_task(21)]

    splits = build_stage.build_splits(
        reuse_tasks=reuse,
        control_tasks=control,
        pilot=True,
    )

    assert {name: len(rows) for name, rows in splits.items()} == {
        "test": 8,
        "control": 2,
    }
    assert {row["reuse_tier"] for row in splits["test"]} == {"high", "medium", "low"}
    assert splits["test"][0]["task_id"] == "LRB2-PILOT/T000"
    assert splits["control"][1]["task_id"] == "LRB2-PILOT/C001"


class TestPools:
    """A provisional build takes what the agent kept; a final build needs every human verdict."""

    @pytest.fixture(autouse=True)
    def _artefacts(self, tmp_path, monkeypatch) -> list[dict]:
        """Two agent-kept reuse tasks, with every artefact in a temporary folder."""
        monkeypatch.setattr(build, "ARTEFACTS_DIR", tmp_path)
        tasks = [_task(1, library="scipy"), _task(2, library="bs4")]
        monkeypatch.setattr(
            build_stage,
            "agent_kept_tasks",
            lambda track: tasks if track == "reuse" else [],
        )
        return tasks

    def test_a_provisional_build_needs_no_human_review(self, _artefacts) -> None:
        assert build_stage.load_pool("reuse", provisional=True) == _artefacts

    def test_a_final_build_refuses_while_verdicts_are_missing(self, _artefacts) -> None:
        review.record_verdict(_artefacts[0], "keep", author="author_1")
        review.record_verdict(_artefacts[0], "keep", author="author_2")
        review.record_verdict(_artefacts[1], "keep", author="author_1")
        with pytest.raises(SystemExit, match="1 of 2 reuse tasks still need human"):
            build_stage.load_pool("reuse", provisional=False)

    def test_a_final_build_takes_only_approved_tasks(self, _artefacts) -> None:
        for author in review.REVIEWERS:
            review.record_verdict(_artefacts[0], "keep", author=author)
        review.record_verdict(_artefacts[1], "keep", author="author_1")
        review.record_verdict(_artefacts[1], "reject", author="author_2")
        pool = build_stage.load_pool("reuse", provisional=False)
        assert _ids(pool) == ["BigCodeBench/1"]

    def test_a_provisional_build_writes_its_files_with_the_canary(
        self,
        _artefacts,
        tmp_path,
        monkeypatch,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "CANARY.txt").write_text("canary")
        monkeypatch.setattr(build_stage, "DATA_DIR", data_dir)
        monkeypatch.setattr(build_stage, "PROVISIONAL_DIR", tmp_path / "provisional")
        monkeypatch.setattr(sys, "argv", ["05_build.py", "--provisional"])

        build_stage.main()

        provisional = tmp_path / "provisional"
        assert (provisional / "CANARY.txt").read_text() == "canary"
        assert len(load_jsonl(provisional / "test.jsonl")) == 2
        # an empty split writes no file, and nothing goes to the final data folder
        assert not (provisional / "control.jsonl").exists()
        assert not (data_dir / "test.jsonl").exists()
