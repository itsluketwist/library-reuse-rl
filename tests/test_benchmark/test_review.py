"""Tests for the stage-06 review helpers in benchmark/pipeline/review.py (loaded via pythonpath)."""

from typing import Any

import pytest
import review


def _problem(seed_id: str, with_extlib: bool = True) -> dict:
    """A minimal validated-problem row with the fields the review helpers read.

    Pass `with_extlib=False` for a control-track-shaped row (no library gold).

    Returns the problem dict.
    """
    problem = {
        "seed_id": seed_id,
        "category": "misc",
        "seed_libs": ["pandas"] if with_extlib else [],
        "prompt": f"do the {seed_id} task",
        "inputs": ["1", "2"],
        "expected_outputs": ["a", "b"],
        "gold_stdlib": "print('a')",
    }
    if with_extlib:
        problem |= {
            "gold_extlib": "import pandas\nprint('a')",
            "gold_extlib_packages": ["pandas==3.0.5"],
        }
    return problem


class TestVerdicts:
    """Verdicts persist to the review file, latest entry per seed wins."""

    def test_record_and_accept(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep")
        review.record_verdict("b", "reject", "too trivial")

        assert set(review.verdicts()) == {"a", "b"}
        assert review.accepted_ids() == {"a"}

    def test_latest_verdict_wins(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "reject")
        review.record_verdict("a", "keep")  # revised on a second look
        assert review.accepted_ids() == {"a"}

    def test_pending_excludes_reviewed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep")
        problems = [_problem("a"), _problem("b"), _problem("c")]
        assert [p["seed_id"] for p in review.pending(problems)] == ["b", "c"]

    def test_update_counts_as_accepted(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep")
        review.record_verdict("b", "update", prompt="a clearer prompt")
        review.record_verdict("c", "reject")
        assert review.accepted_ids() == {"a", "b"}

    def test_tracks_use_separate_verdict_files(self, tmp_path, monkeypatch) -> None:
        """The reuse and control tracks are reviewed and shipped independently."""
        monkeypatch.setattr(
            review,
            "review_path",
            lambda track="reuse": tmp_path / f"reviews_{track}.jsonl",
        )
        review.record_verdict("a", "keep", track="reuse")
        review.record_verdict("a", "reject", track="control")

        assert review.verdicts("reuse")["a"]["verdict"] == "keep"
        assert review.verdicts("control")["a"]["verdict"] == "reject"
        assert review.accepted_ids("reuse") == {"a"}
        assert review.accepted_ids("control") == set()


class TestApprovedProblems:
    """Approved problems keep/update ship, with an edited prompt applied for updates."""

    def test_applies_prompt_override(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        monkeypatch.setattr(
            review,
            "validated_problems",
            lambda track="reuse": [_problem("a"), _problem("b"), _problem("c")],
        )
        review.record_verdict("a", "keep")
        review.record_verdict("b", "update", prompt="a clearer prompt")
        review.record_verdict("c", "reject")

        approved = {p["seed_id"]: p for p in review.approved_problems()}
        assert set(approved) == {"a", "b"}
        assert approved["a"]["prompt"] == "do the a task"  # unchanged
        assert approved["b"]["prompt"] == "a clearer prompt"  # edited version ships


def _buttons(widget: object) -> list:
    """Collect every clickable Button in a widget tree.

    Returns the buttons found, depth-first.
    """
    found: list = []
    for child in getattr(widget, "children", []):
        found += _buttons(child)
    if getattr(widget, "description", None) is not None and hasattr(widget, "click"):
        found.append(widget)
    return found


def _textareas(widget: object) -> list:
    """Collect every Textarea in a widget tree.

    Returns the text areas found, depth-first.
    """
    found: list = []
    for child in getattr(widget, "children", []):
        found += _textareas(child)
    if type(widget).__name__ == "Textarea":
        found.append(widget)
    return found


def _extlib_accordion(gui: object) -> Any:
    """The extlib gold-solution accordion - the sixth child of the GUI's top-level VBox.

    Returns the accordion widget.
    """
    return getattr(gui, "children", [])[5]


class TestReviewGui:
    """The GUI drives keep/update/reject on the current problem and persists the verdict."""

    def test_keep_records_and_advances(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui("author_1", problems=[_problem("a"), _problem("b")])

        by_label = {b.description: b for b in _buttons(gui)}
        assert {"keep", "update", "reject", "← back"} <= set(by_label)

        by_label["keep"].click()  # verdict for the first problem (a), advance to b
        by_label["reject"].click()  # verdict for b

        assert review.accepted_ids() == {"a"}
        assert review.verdicts()["b"]["verdict"] == "reject"

    def test_verdict_is_attributed_to_the_reviewing_author(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui("author_2", problems=[_problem("a")])
        {b.description: b for b in _buttons(gui)}["keep"].click()
        assert review.verdicts()["a"]["author"] == "author_2"

    @pytest.mark.parametrize(
        "bad_author", ["", "  ", "alice", "Author_1", "author_", "author_0", "author1"]
    )
    def test_non_anonymous_author_is_rejected(
        self, bad_author, tmp_path, monkeypatch
    ) -> None:
        # only "author_<N>" (N >= 1) is accepted - a real name would break submission anonymity
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        with pytest.raises(ValueError):
            review.review_gui(bad_author, problems=[_problem("a")])

    def test_update_ships_edited_prompt(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui("author_1", problems=[_problem("a"), _problem("b")])

        # edit the prompt in the editor, then click update
        _textareas(gui)[0].value = "a much clearer prompt"
        {b.description: b for b in _buttons(gui)}["update"].click()

        record = review.verdicts()["a"]
        assert record["verdict"] == "update"
        assert record["prompt"] == "a much clearer prompt"

    def test_control_track_verdict_uses_control_file(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review,
            "review_path",
            lambda track="reuse": tmp_path / f"reviews_{track}.jsonl",
        )
        gui = review.review_gui(
            "author_1", problems=[_problem("a", with_extlib=False)], track="control"
        )
        {b.description: b for b in _buttons(gui)}["keep"].click()

        assert review.verdicts("control")["a"]["verdict"] == "keep"
        assert review.verdicts("reuse") == {}

    def test_no_extlib_panel_for_control_problem(self, tmp_path, monkeypatch) -> None:
        """A control-track problem has no gold_extlib, so its accordion panel is hidden."""
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui(
            "author_1", problems=[_problem("a", with_extlib=False)], track="control"
        )
        assert _extlib_accordion(gui).layout.display == "none"

    def test_extlib_panel_shown_for_reuse_problem(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui("author_1", problems=[_problem("a")])
        assert _extlib_accordion(gui).layout.display != "none"
