"""Tests for the human-review helpers in benchmark/pipeline/review.py (loaded via pythonpath)."""

from typing import Any

import build
import pytest
import review


def _task(
    seed_id: str,
    with_extlib: bool = True,
    prompt: str | None = None,
) -> dict:
    """A minimal kept task with the fields the review helpers read.

    Pass `with_extlib=False` for a control-track-shaped task (no library solution).

    Returns the task record.
    """
    task: dict[str, Any] = {
        "seed_id": seed_id,
        "category": "misc",
        "domain": "other",
        "function_name": "solve_it",
        "prompt": prompt or f"do the {seed_id} task",
        "test_inputs": ["[1]", "[2]"],
        "expected_outputs": ["'a'", "'b'"],
        "stdlib_solution": "def solve_it(x):\n    return 'a'\n",
    }
    if with_extlib:
        task |= {
            "primary_library": "pandas",
            "extlib_solution": "import pandas\n\n\ndef solve_it(x):\n    return 'a'\n",
            "gold_extlib_packages": ["pandas==3.0.5"],
            "stdlib_lloc": 40,
            "extlib_lloc": 10,
            "reuse_value": 30,
            "reuse_tier": "high",
        }
    return task | {"task_hash": build.task_hash(task)}


@pytest.fixture(autouse=True)
def _artefacts(tmp_path, monkeypatch) -> None:
    """Keep every verdict file in a temporary folder, away from the real artefacts."""
    monkeypatch.setattr(build, "ARTEFACTS_DIR", tmp_path)


def _approved_ids(
    tasks: list[dict],
    track: str = "reuse",
) -> set[str]:
    """The seed ids of the tasks that would ship."""
    return {task["seed_id"] for task in review.approved_tasks(track, tasks)}


class TestVerdicts:
    """Both reviewers independently verdict the same pool; reject wins on any disagreement."""

    def test_record_and_accept(self) -> None:
        a, b = _task("a"), _task("b")
        review.record_verdict(a, "keep", author="author_1")
        review.record_verdict(a, "keep", author="author_2")
        review.record_verdict(b, "reject", "too trivial", author="author_1")
        review.record_verdict(b, "keep", author="author_2")

        assert set(review.verdicts()) == {"a", "b"}
        assert _approved_ids([a, b]) == {"a"}

    def test_needs_both_reviewers_to_accept(self) -> None:
        a = _task("a")
        review.record_verdict(a, "keep", author="author_1")
        assert _approved_ids([a]) == set()  # author_2 hasn't weighed in yet

    def test_latest_verdict_wins_per_author(self) -> None:
        a = _task("a")
        review.record_verdict(a, "reject", author="author_1")
        review.record_verdict(a, "keep", author="author_1")  # revised on a second look
        review.record_verdict(a, "keep", author="author_2")
        assert _approved_ids([a]) == {"a"}

    def test_a_verdict_on_an_earlier_version_does_not_count(self) -> None:
        """A task regenerated after it was reviewed has to be reviewed again."""
        old = _task("a")
        review.record_verdict(old, "keep", author="author_1")
        review.record_verdict(old, "keep", author="author_2")
        new = _task("a", prompt="do the a task, with ties broken by name")

        assert _approved_ids([old]) == {"a"}
        assert _approved_ids([new]) == set()
        assert review.pending([new]) == [new]

    def test_pending_without_author_needs_both_reviewers(self) -> None:
        tasks = [_task("a"), _task("b"), _task("c")]
        review.record_verdict(tasks[0], "keep", author="author_1")
        review.record_verdict(tasks[0], "keep", author="author_2")
        review.record_verdict(
            tasks[1], "keep", author="author_1"
        )  # one reviewer so far
        assert [task["seed_id"] for task in review.pending(tasks)] == ["b", "c"]

    def test_pending_with_author_is_that_authors_own_progress(self) -> None:
        tasks = [_task("a"), _task("b"), _task("c")]
        review.record_verdict(tasks[0], "keep", author="author_1")
        mine = review.pending(tasks, author="author_1")
        assert [task["seed_id"] for task in mine] == ["b", "c"]
        # author_2 hasn't reviewed anything yet - everything is pending for them
        assert review.pending(tasks, author="author_2") == tasks

    def test_tracks_use_separate_verdict_files(self) -> None:
        """The reuse and control tracks are reviewed and shipped independently."""
        a = _task("a")
        review.record_verdict(a, "keep", track="reuse", author="author_1")
        review.record_verdict(a, "keep", track="reuse", author="author_2")
        review.record_verdict(a, "reject", track="control", author="author_1")

        assert review.verdicts("reuse")["a"]["author_1"]["verdict"] == "keep"
        assert review.verdicts("control")["a"]["author_1"]["verdict"] == "reject"
        assert _approved_ids([a], "reuse") == {"a"}
        assert _approved_ids([a], "control") == set()


class TestApprovedTasks:
    """Approved tasks ship, with an edited task text applied for an update."""

    def test_applies_prompt_override(self) -> None:
        tasks = [_task("a"), _task("b"), _task("c")]
        review.record_verdict(tasks[0], "keep", author="author_1")
        review.record_verdict(tasks[0], "keep", author="author_2")
        review.record_verdict(
            tasks[1], "update", prompt="a clearer prompt", author="author_1"
        )
        review.record_verdict(tasks[1], "keep", author="author_2")
        review.record_verdict(tasks[2], "reject", author="author_1")
        review.record_verdict(tasks[2], "keep", author="author_2")  # reject wins

        approved = {
            task["seed_id"]: task for task in review.approved_tasks("reuse", tasks)
        }
        assert set(approved) == {"a", "b"}
        assert approved["a"]["prompt"] == "do the a task"  # unchanged
        assert approved["b"]["prompt"] == "a clearer prompt"  # edited version ships


def _buttons(widget: object) -> dict[str, Any]:
    """Collect every clickable Button in a widget tree.

    Returns the buttons found, by their label.
    """
    found: dict[str, Any] = {}
    for child in getattr(widget, "children", []):
        found |= _buttons(child)
    if getattr(widget, "description", None) is not None and hasattr(widget, "click"):
        found[getattr(widget, "description")] = widget
    return found


def _of_type(gui: object, name: str) -> list:
    """The GUI's top-level widgets of one type (e.g. "Accordion"), in order.

    Returns the widgets.
    """
    return [
        child for child in getattr(gui, "children", []) if type(child).__name__ == name
    ]


def _editor(gui: object) -> Any:
    """The editable task text of a GUI.

    Returns the text area widget.
    """
    return _of_type(gui, "Textarea")[0]


def _html(gui: object) -> str:
    """All the text a GUI shows at its top level (the header, reuse line and status).

    Returns the joined html.
    """
    return " ".join(widget.value for widget in _of_type(gui, "HTML"))


class TestReviewGui:
    """The GUI drives keep/update/reject on the current task and saves the verdict."""

    def test_keep_records_and_advances(self) -> None:
        tasks = [_task("a"), _task("b")]
        gui = review.review_gui("author_1", tasks=tasks)

        buttons = _buttons(gui)
        assert {"keep", "update", "reject", "← back"} <= set(buttons)

        buttons["keep"].click()  # verdict for the first task (a), advance to b
        buttons["reject"].click()  # verdict for b

        assert review.verdicts()["b"]["author_1"]["verdict"] == "reject"
        review.record_verdict(tasks[0], "keep", author="author_2")
        assert _approved_ids(tasks) == {"a"}

    def test_verdict_is_attributed_to_the_reviewing_author(self) -> None:
        gui = review.review_gui("author_2", tasks=[_task("a")])
        _buttons(gui)["keep"].click()
        record = review.verdicts()["a"]["author_2"]
        assert record["author"] == "author_2"
        assert record["task_hash"] == _task("a")["task_hash"]

    @pytest.mark.parametrize(
        "bad_author", ["", "  ", "alice", "Author_1", "author_", "author_0", "author1"]
    )
    def test_non_anonymous_author_is_rejected(self, bad_author: str) -> None:
        # only "author_<N>" (N >= 1) is accepted - a real name would break submission anonymity
        with pytest.raises(ValueError):
            review.review_gui(bad_author, tasks=[_task("a")])

    def test_update_ships_edited_prompt(self) -> None:
        gui = review.review_gui("author_1", tasks=[_task("a"), _task("b")])

        # edit the task text in the editor, then click update
        _editor(gui).value = "a much clearer prompt"
        _buttons(gui)["update"].click()

        record = review.verdicts()["a"]["author_1"]
        assert record["verdict"] == "update"
        assert record["prompt"] == "a much clearer prompt"

    def test_the_task_is_shown_with_its_reuse_value(self) -> None:
        gui = review.review_gui("author_1", tasks=[_task("a")])
        assert "reuse value 30" in _html(gui)
        assert "40 logical lines" in _html(gui)
        titles = [panel.get_title(0) for panel in _of_type(gui, "Accordion")]
        assert titles == [
            "agent review",
            "tests (2)",
            "gold: standard library",
            "gold: library ['pandas==3.0.5']",
        ]

    def test_control_track_verdict_uses_control_file(self) -> None:
        gui = review.review_gui(
            "author_1", tasks=[_task("a", with_extlib=False)], track="control"
        )
        _buttons(gui)["keep"].click()

        assert review.verdicts("control")["a"]["author_1"]["verdict"] == "keep"
        assert review.verdicts("reuse") == {}

    def test_no_library_panel_for_a_control_task(self) -> None:
        """A control-track task has no library solution, so its panel is hidden."""
        gui = review.review_gui(
            "author_1", tasks=[_task("a", with_extlib=False)], track="control"
        )
        assert _of_type(gui, "Accordion")[-1].layout.display == "none"
        assert "control task" in _html(gui)

    def test_second_author_starts_at_zero_reviewed_and_first_task(self) -> None:
        """The bug this guards: author_2 must not see author_1's tasks as already reviewed."""
        tasks = [_task("a"), _task("b")]
        gui_1 = review.review_gui("author_1", tasks=tasks)
        _buttons(gui_1)["keep"].click()  # author_1 reviews "a"

        gui_2 = review.review_gui("author_2", tasks=tasks)
        assert "reviewed 0 / 2" in _html(gui_2)
        # resumes at the first task ("a"), not "b" - author_1 reviewing "a" doesn't count
        assert _editor(gui_2).value == tasks[0]["prompt"]

    def test_second_author_does_not_see_first_authors_answer(self) -> None:
        tasks = [_task("a")]
        gui_1 = review.review_gui("author_1", tasks=tasks)
        _editor(gui_1).value = "author 1's edited prompt"
        _buttons(gui_1)["update"].click()

        gui_2 = review.review_gui("author_2", tasks=tasks)
        assert _editor(gui_2).value == tasks[0]["prompt"]  # original, not author_1's
        # no verdict shown - author_2 hasn't reviewed it yet
        assert "your verdict" not in _html(gui_2)

    def test_an_author_resumes_after_their_last_verdict(self) -> None:
        tasks = [_task("a"), _task("b")]
        _buttons(review.review_gui("author_1", tasks=tasks))["keep"].click()

        resumed = review.review_gui("author_1", tasks=tasks)
        assert "reviewed 1 / 2" in _html(resumed)
        assert _editor(resumed).value == tasks[1]["prompt"]
