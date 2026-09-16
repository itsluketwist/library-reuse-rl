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
    """Both reviewers independently verdict the same pool; reject wins on any disagreement."""

    def test_record_and_accept(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        review.record_verdict("a", "keep", author="author_2")
        review.record_verdict("b", "reject", "too trivial", author="author_1")
        review.record_verdict("b", "keep", author="author_2")

        assert set(review.verdicts()) == {"a", "b"}
        assert review.accepted_ids() == {"a"}

    def test_reject_wins_on_disagreement(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        review.record_verdict("a", "reject", author="author_2")
        assert review.accepted_ids() == set()

    def test_needs_both_reviewers_to_accept(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        assert review.accepted_ids() == set()  # author_2 hasn't weighed in yet

    def test_latest_verdict_wins_per_author(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "reject", author="author_1")
        review.record_verdict(
            "a", "keep", author="author_1"
        )  # revised on a second look
        review.record_verdict("a", "keep", author="author_2")
        assert review.accepted_ids() == {"a"}

    def test_pending_without_author_needs_both_reviewers(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        review.record_verdict("a", "keep", author="author_2")
        review.record_verdict(
            "b", "keep", author="author_1"
        )  # only one reviewer so far
        problems = [_problem("a"), _problem("b"), _problem("c")]
        assert [p["seed_id"] for p in review.pending(problems)] == ["b", "c"]

    def test_pending_with_author_is_that_authors_own_progress(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        problems = [_problem("a"), _problem("b"), _problem("c")]
        assert [p["seed_id"] for p in review.pending(problems, author="author_1")] == [
            "b",
            "c",
        ]
        # author_2 hasn't reviewed anything yet - everything is pending for them
        assert [p["seed_id"] for p in review.pending(problems, author="author_2")] == [
            "a",
            "b",
            "c",
        ]

    def test_legacy_rows_without_an_author_count_as_the_first_reviewer(
        self, tmp_path, monkeypatch
    ) -> None:
        """Rows written before per-author attribution existed predate the two-reviewer setup."""
        path = tmp_path / "reviews.jsonl"
        monkeypatch.setattr(review, "review_path", lambda track="reuse": path)
        path.write_text('{"seed_id": "a", "verdict": "keep", "note": ""}\n')
        assert review.author_verdicts(author="author_1") == {
            "a": {"seed_id": "a", "verdict": "keep", "note": ""}
        }
        assert review.author_verdicts(author="author_2") == {}

    def test_update_counts_as_accepted(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        review.record_verdict("a", "keep", author="author_1")
        review.record_verdict(
            "a", "update", prompt="a clearer prompt", author="author_2"
        )
        assert review.accepted_ids() == {"a"}

    def test_tracks_use_separate_verdict_files(self, tmp_path, monkeypatch) -> None:
        """The reuse and control tracks are reviewed and shipped independently."""
        monkeypatch.setattr(
            review,
            "review_path",
            lambda track="reuse": tmp_path / f"reviews_{track}.jsonl",
        )
        review.record_verdict("a", "keep", track="reuse", author="author_1")
        review.record_verdict("a", "keep", track="reuse", author="author_2")
        review.record_verdict("a", "reject", track="control", author="author_1")

        assert review.verdicts("reuse")["a"]["author_1"]["verdict"] == "keep"
        assert review.verdicts("control")["a"]["author_1"]["verdict"] == "reject"
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
        review.record_verdict("a", "keep", author="author_1")
        review.record_verdict("a", "keep", author="author_2")
        review.record_verdict(
            "b", "update", prompt="a clearer prompt", author="author_1"
        )
        review.record_verdict("b", "keep", author="author_2")
        review.record_verdict("c", "reject", author="author_1")
        review.record_verdict(
            "c", "keep", author="author_2"
        )  # disagreement - reject wins

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


def _header(gui: object) -> str:
    """The header HTML - the first child of the GUI's top-level VBox.

    Returns the header text.
    """
    return getattr(gui, "children", [])[0].value


def _status(gui: object) -> str:
    """The status HTML - the last child of the GUI's top-level VBox.

    Returns the status text.
    """
    return getattr(gui, "children", [])[-1].value


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

        assert review.author_verdicts(author="author_1")["b"]["verdict"] == "reject"
        review.record_verdict("a", "keep", author="author_2")
        assert review.accepted_ids() == {"a"}

    def test_verdict_is_attributed_to_the_reviewing_author(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        gui = review.review_gui("author_2", problems=[_problem("a")])
        {b.description: b for b in _buttons(gui)}["keep"].click()
        assert review.verdicts()["a"]["author_2"]["author"] == "author_2"

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

        record = review.verdicts()["a"]["author_1"]
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

        assert review.verdicts("control")["a"]["author_1"]["verdict"] == "keep"
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

    def test_second_author_starts_at_zero_reviewed_and_first_problem(
        self, tmp_path, monkeypatch
    ) -> None:
        """The bug this guards: author_2 must not see author_1's problems as already reviewed."""
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        problems = [_problem("a"), _problem("b")]
        gui_1 = review.review_gui("author_1", problems=problems)
        {b.description: b for b in _buttons(gui_1)}[
            "keep"
        ].click()  # author_1 reviews "a"

        gui_2 = review.review_gui("author_2", problems=problems)
        assert "reviewed 0 / 2" in _header(gui_2)
        # resumes at the first problem ("a"), not "b" - author_1 reviewing "a" doesn't count
        assert _textareas(gui_2)[0].value == problems[0]["prompt"]

    def test_second_author_does_not_see_first_authors_answer(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            review, "review_path", lambda track="reuse": tmp_path / "reviews.jsonl"
        )
        problems = [_problem("a")]
        gui_1 = review.review_gui("author_1", problems=problems)
        _textareas(gui_1)[0].value = "author 1's edited prompt"
        {b.description: b for b in _buttons(gui_1)}["update"].click()

        gui_2 = review.review_gui("author_2", problems=problems)
        assert (
            _textareas(gui_2)[0].value == problems[0]["prompt"]
        )  # original, not author_1's
        # no verdict shown - author_2 hasn't reviewed it yet
        assert "verdict" not in _status(gui_2)
