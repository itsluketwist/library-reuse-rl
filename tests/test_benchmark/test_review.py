"""Tests for the stage-06 review helpers in benchmark/pipeline/review.py (loaded via pythonpath)."""

import review


def _problem(seed_id: str) -> dict:
    """A minimal validated-problem row with the fields the review helpers read.

    Returns the problem dict.
    """
    return {
        "seed_id": seed_id,
        "category": "misc",
        "seed_libs": ["pandas"],
        "prompt": f"do the {seed_id} task",
        "inputs": ["1", "2"],
        "expected_outputs": ["a", "b"],
        "gold_stdlib": "print('a')",
        "gold_extlib": "import pandas\nprint('a')",
        "gold_extlib_packages": ["pandas==3.0.5"],
    }


class TestVerdicts:
    """Verdicts persist to the review file, latest entry per seed wins."""

    def test_record_and_accept(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        review.record_verdict("a", "keep")
        review.record_verdict("b", "reject", "too trivial")

        assert set(review.verdicts()) == {"a", "b"}
        assert review.accepted_ids() == {"a"}

    def test_latest_verdict_wins(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        review.record_verdict("a", "reject")
        review.record_verdict("a", "keep")  # revised on a second look
        assert review.accepted_ids() == {"a"}

    def test_pending_excludes_reviewed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        review.record_verdict("a", "keep")
        problems = [_problem("a"), _problem("b"), _problem("c")]
        assert [p["seed_id"] for p in review.pending(problems)] == ["b", "c"]

    def test_update_counts_as_accepted(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        review.record_verdict("a", "keep")
        review.record_verdict("b", "update", prompt="a clearer prompt")
        review.record_verdict("c", "reject")
        assert review.accepted_ids() == {"a", "b"}


class TestApprovedProblems:
    """Approved problems keep/update ship, with an edited prompt applied for updates."""

    def test_applies_prompt_override(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        monkeypatch.setattr(
            review,
            "validated_problems",
            lambda: [_problem("a"), _problem("b"), _problem("c")],
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


class TestReviewGui:
    """The GUI drives keep/update/reject on the current problem and persists the verdict."""

    def test_keep_records_and_advances(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        gui = review.review_gui(problems=[_problem("a"), _problem("b")])

        by_label = {b.description: b for b in _buttons(gui)}
        assert {"keep", "update", "reject", "← back"} <= set(by_label)

        by_label["keep"].click()  # verdict for the first problem (a), advance to b
        by_label["reject"].click()  # verdict for b

        assert review.accepted_ids() == {"a"}
        assert review.verdicts()["b"]["verdict"] == "reject"

    def test_update_ships_edited_prompt(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "reviews.jsonl")
        gui = review.review_gui(problems=[_problem("a"), _problem("b")])

        # edit the prompt in the editor, then click update
        _textareas(gui)[0].value = "a much clearer prompt"
        {b.description: b for b in _buttons(gui)}["update"].click()

        record = review.verdicts()["a"]
        assert record["verdict"] == "update"
        assert record["prompt"] == "a much clearer prompt"
