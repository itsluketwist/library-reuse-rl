"""Tests for stage 05 of the benchmark build: the variety caps, the splits and the task ids.

The stage is a script (benchmark/pipeline/05_build.py), so it is loaded by path.
"""

import importlib.util
import sys
from collections import Counter
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


class TestVarietyCaps:
    """No category and no library may dominate, and the highest reuse values are kept."""

    def test_a_category_keeps_its_three_highest_value_tasks(self) -> None:
        tasks = [
            _task(number, value=value, category="html")
            for number, value in enumerate([20, 50, 30, 40, 10])
        ]
        tasks.append(_task(9, value=5, category="dates"))

        kept = build_stage.cap_categories(tasks)

        # values 50, 40 and 30 of "html", then the one "dates" task
        assert _ids(kept) == [
            "BigCodeBench/1",
            "BigCodeBench/3",
            "BigCodeBench/2",
            "BigCodeBench/9",
        ]

    def test_no_library_holds_more_than_a_fifth(self) -> None:
        # 8 pandas tasks, and 2 tasks each for 6 other libraries
        tasks = [_task(number, value=100 - number) for number in range(8)]
        tasks += [
            _task(10 + number, library=f"lib{number // 2}", value=50 - number)
            for number in range(12)
        ]

        kept = build_stage.cap_libraries(tasks)

        counts = Counter(task["primary_library"] for task in kept)
        assert len(kept) == 15
        assert counts["pandas"] == 3  # a fifth of 15
        # the pandas tasks that stay are its three highest-value ones
        pandas_kept = [task for task in kept if task["primary_library"] == "pandas"]
        assert _ids(pandas_kept) == [f"BigCodeBench/{number}" for number in range(3)]

    def test_a_pool_with_few_libraries_is_not_emptied(self) -> None:
        tasks = [_task(0), _task(1), _task(2, library="scipy")]
        kept = build_stage.cap_libraries(tasks)
        assert Counter(task["primary_library"] for task in kept) == {
            "pandas": 1,
            "scipy": 1,
        }

    def test_low_value_tasks_are_spread_across_libraries(self) -> None:
        tasks = [_task(number, library="pandas") for number in range(5)]
        tasks += [_task(10, library="numpy"), _task(11, library="scipy")]

        picked = build_stage.spread_across_libraries(tasks, count=4)

        assert Counter(task["primary_library"] for task in picked) == {
            "pandas": 2,
            "numpy": 1,
            "scipy": 1,
        }


class TestBuildSplits:
    """The tasks are split by tier, and each gets a unique id."""

    @pytest.fixture
    def splits(self) -> dict[str, list[dict]]:
        """Splits from 30 high, 4 medium and 6 low tasks over 10 libraries, and 5 control."""
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

    def test_split_sizes(self, splits) -> None:
        sizes = {name: len(rows) for name, rows in splits.items()}
        # 6 of the 30 high tasks are held out for dev (never more than a fifth)
        assert sizes == {"test": 24, "dev": 6, "low_value": 6, "control": 5}

    def test_tiers_go_to_their_splits(self, splits) -> None:
        assert {row["reuse_tier"] for row in splits["test"] + splits["dev"]} == {"high"}
        assert {row["reuse_tier"] for row in splits["low_value"]} == {"low"}
        shipped = {row["source"]["seed_id"] for rows in splits.values() for row in rows}
        # medium-tier tasks (seeds 100 to 103) are in no split
        assert not shipped & {f"BigCodeBench/{number}" for number in range(100, 104)}
        assert len(shipped) == 41

    def test_task_ids(self, splits) -> None:
        assert [row["task_id"] for row in splits["test"]][:2] == [
            "LRB2/T000",
            "LRB2/T001",
        ]
        assert splits["dev"][0]["task_id"] == "LRB2/D000"
        assert splits["low_value"][-1]["task_id"] == "LRB2/L005"
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
    """Nothing is capped, held out or left out by tier, and the ids can't be the benchmark's."""
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
