"""Tests for the staged function wordings: the deterministic check, the prompt and the generator."""

import json
from pathlib import Path
from typing import Any

import pytest

from data.build.library import augment
from data.build.library.augment import (
    build_function_stage_prompt,
    check_function_stages,
    generate_function_stages,
)


# a taxonomy.yaml category, as the generator is given it
_CATEGORY: dict[str, Any] = {
    "id": "datetime",
    "name": "Date/Time Handling",
    "seed_libraries": ["dateutil", "arrow", "pendulum", "pytz"],
    "diy_temptation": "parsing or arithmetic on dates by hand instead of a date library",
}

_SIGNATURE = "count_weekdays(dates)"
_REQUEST = (
    f"Write a function `{_SIGNATURE}` that takes a list of date strings and returns a dict "
    "mapping each weekday name to how many of the dates fall on it."
)


def _function_entry(
    fits_example: bool = True,
) -> dict[str, Any]:
    """A cache entry whose three requests all pass check_function_stages."""
    example_call = "count_weekdays(['2024-03-11', '14 March 2024'])"
    example_result = "{'Monday': 1, 'Thursday': 1}"
    return {
        "fits_example": fits_example,
        "reason": "" if fits_example else "the result depends on the clock",
        "function_name": "count_weekdays",
        "function_signature": _SIGNATURE,
        "solution": "from dateutil import parser\n\ndef count_weekdays(dates): ...",
        "name": (
            "Write a function called `count_weekdays` that counts how many dates fall on "
            "each day of the week."
        ),
        "signature": _REQUEST,
        "example": (
            f"{_REQUEST} For example, `{example_call}` returns `{example_result}`."
            if fits_example
            else ""
        ),
        "example_call": example_call if fits_example else "",
        "example_result": example_result if fits_example else "",
    }


def _problems(
    entry: dict[str, Any],
) -> list[str]:
    """Check `entry` against the test category's libraries.

    Returns the problems found.
    """
    return check_function_stages(entry, _CATEGORY["seed_libraries"])


class TestCheckFunctionStages:
    """Tests for check_function_stages, the deterministic check of the three requests."""

    def test_a_well_formed_entry_passes(self):
        """Three requests for the same function, each at its own level, have no problems."""
        assert _problems(_function_entry()) == []

    def test_an_entry_with_no_example_passes_without_one(self):
        """A task with no honest example keeps only its name and signature requests."""
        assert _problems(_function_entry(fits_example=False)) == []

    def test_parameter_list_in_the_name_stage(self):
        """The name stage gives only the name, so a parameter list is rejected."""
        entry = {**_function_entry(), "name": _REQUEST}
        assert any(p.startswith("name: writes") for p in _problems(entry))

    def test_example_in_the_signature_stage(self):
        """The signature stage has no example."""
        entry = _function_entry()
        entry["signature"] = entry["example"]
        assert any(p.startswith("signature: writes") for p in _problems(entry))

    def test_two_examples(self):
        """The example stage has exactly one example."""
        entry = _function_entry()
        entry["example"] += " And `count_weekdays([])` returns `{}`."
        assert any(p.startswith("example: writes") for p in _problems(entry))

    def test_a_different_signature(self):
        """Every stage after the first gives the signature exactly as function_signature."""
        entry = _function_entry()
        entry["signature"] = entry["signature"].replace("(dates)", "(values)")
        assert (
            "signature: doesn't give the signature exactly as in function_signature"
            in _problems(entry)
        )

    def test_example_not_word_for_word(self):
        """The example's result must appear in the request exactly as written."""
        entry = {**_function_entry(), "example_result": "{'Monday': 1}"}
        assert (
            "example: example_result doesn't appear in it exactly as written"
            in _problems(entry)
        )

    def test_example_result_must_be_a_literal(self):
        """The example's result is a python literal, not prose or an object."""
        # in the request word for word, but not a literal
        entry = {**_function_entry(), "example_result": "weekday name"}
        assert "example: example_result is not a python literal" in _problems(entry)

    def test_library_named_in_a_request(self):
        """A request that names one of the category's libraries is rejected."""
        entry = _function_entry()
        entry["signature"] += " It can use pytz."
        assert any(
            p.startswith("signature: mentions libraries") for p in _problems(entry)
        )

    def test_library_in_the_function_name(self):
        """A function name made from a library's name points at that library."""
        entry = json.loads(
            json.dumps(_function_entry()).replace("count_weekdays", "slugify_title")
        )
        problems = check_function_stages(entry, ["slugify"])
        assert any("function name 'slugify_title' mentions" in p for p in problems)

    def test_over_the_length_limit(self):
        """Each stage has a length limit - the name stage is one short sentence."""
        entry = _function_entry()
        entry["name"] += " It should cope with every common way of writing a date." * 5
        assert any(p.startswith("name: ") and "limit" in p for p in _problems(entry))


def test_prompt_shows_both_sides_of_the_choice():
    """The request names the task's libraries and diy temptation, and each stage's limit."""
    prompt = build_function_stage_prompt("Tally dates by weekday.", _CATEGORY)

    assert "Tally dates by weekday." in prompt
    assert "dateutil, arrow, pendulum, pytz" in prompt
    assert _CATEGORY["diy_temptation"] in prompt
    for stage, limit in augment.FUNCTION_STAGE_MAX_CHARS.items():
        assert f'"{stage}"' in prompt
        assert f"At most {limit} characters" in prompt
    assert "previous attempt was rejected" not in prompt


class TestGenerateFunctionStages:
    """Tests for generate_function_stages, with the LLM call replaced by canned replies."""

    def _fake_llm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        replies: list[dict[str, Any]],
    ) -> list[str]:
        """Make llm.complete_json return `replies` in turn.

        Returns a list that collects the prompt of every call made.
        """
        calls: list[str] = []

        def complete_json(**kwargs: Any) -> tuple[dict[str, Any], str]:
            calls.append(kwargs["prompt"])
            return replies[len(calls) - 1], "claude-test"

        monkeypatch.setattr(augment.llm, "complete_json", complete_json)
        return calls

    def test_rerolls_until_every_stage_passes(self, monkeypatch: pytest.MonkeyPatch):
        """One bad request rejects the whole attempt, and the next good one is kept."""
        good = _function_entry()
        bad = {**good, "name": _REQUEST}
        calls = self._fake_llm(monkeypatch, [bad, good])

        entry = generate_function_stages("task", _CATEGORY)

        assert len(calls) == 2
        # the second attempt is told what was wrong with the first
        assert "previous attempt was rejected" not in calls[0]
        assert "name: writes" in calls[1].split("previous attempt was rejected")[1]
        assert entry == {
            **good,
            "model": "claude-test",
            "prompt_version": augment.FUNCTION_STAGE_PROMPT_VERSION,
        }

    def test_task_with_no_example(self, monkeypatch: pytest.MonkeyPatch):
        """A task judged to have no honest example is cached with an empty example stage."""
        reply = {**_function_entry(), "fits_example": False, "reason": "uses the clock"}
        self._fake_llm(monkeypatch, [reply])

        entry = generate_function_stages("task", _CATEGORY)

        assert entry is not None
        assert entry["fits_example"] is False
        assert (
            entry["example"] == entry["example_call"] == entry["example_result"] == ""
        )
        assert entry["signature"] == _REQUEST

    def test_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch):
        """Every attempt failing returns None, so nothing is cached for the task."""
        bad = {**_function_entry(), "name": _REQUEST}
        calls = self._fake_llm(monkeypatch, [bad, bad])

        assert generate_function_stages("task", _CATEGORY, max_attempts=2) is None
        assert len(calls) == 2


def test_regenerate_writes_the_function_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Each task is generated from its own category, and cached under its seed task."""
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(augment, "FUNCTION_STAGE_CACHE_FILE", cache_file)
    seen: list[tuple[str, str]] = []

    def fake_generate(
        task: str,
        category: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen.append((task, category["id"]))
        return _function_entry()

    monkeypatch.setattr(augment, "generate_function_stages", fake_generate)

    augment.regenerate_function_stage_cache(
        {"code": [_CATEGORY]},
        {"datetime": {1: "Tally dates by weekday."}},
        model="m",
    )

    assert seen == [("Tally dates by weekday.", "datetime")]
    assert json.loads(cache_file.read_text()) == {"datetime.1": _function_entry()}
