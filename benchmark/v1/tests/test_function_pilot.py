"""Tests for the function-style pilot's generator: its checks, test conversion and retry loop.

The generator is a script (benchmark/pipeline/function_pilot/generate.py), so it is loaded by path.
No model is called: the request is stubbed, and the generated code runs with this test's python.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


_GENERATE_PATH = (
    Path(__file__).parents[2]
    / "benchmark"
    / "pipeline"
    / "function_pilot"
    / "generate.py"
)
_spec = importlib.util.spec_from_file_location("function_pilot", _GENERATE_PATH)
assert _spec is not None and _spec.loader is not None
function_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(function_pilot)


def _row(
    with_library_gold: bool = False,
) -> dict[str, Any]:
    """A small benchmark task: count the words on standard input, or report an empty input."""
    ground_truth: dict[str, Any] = {
        "gold_stdlib": "import sys\nprint(len(sys.stdin.read().split()))\n",
    }
    if with_library_gold:
        ground_truth["gold_extlib"] = "import sys\nimport regex\nprint(1)\n"
        ground_truth["gold_extlib_packages"] = ["regex==2026.1.1"]
    return {
        "task_id": "LRB/C999",
        "category": "text",
        "prompt": "Write a program that reads text and prints how many words it has.",
        "test_inputs": ["a b c\n", "x\n", "\n"],
        "expected_outputs": ["3", "1", "Invalid input"],
        "ground_truth": ground_truth,
        "source": {"seed_id": "BigCodeBench/1", "seed_libs": []},
    }


def _entry(
    **changes: Any,
) -> dict[str, Any]:
    """A well-formed rewrite of _row's task, with any fields replaced by `changes`."""
    entry = {
        "fits_function": True,
        "reason": "",
        "function_name": "count_words",
        "prompt": (
            "Write a function `count_words(text)` that returns how many words the text has, "
            "and raises a ValueError if it has none. For example, `count_words('a b')` "
            "returns `2`."
        ),
        "stdlib_solution": (
            "def count_words(text):\n"
            "    words = text.split()\n"
            "    if not words:\n"
            "        raise ValueError('no words')\n"
            "    return len(words)\n"
        ),
        "extlib_solution": "",
        "convert_input": "def convert_input(stdin_text):\n    return [stdin_text]\n",
        "transcribed_outputs": ["3", "1", function_pilot.RAISES],
        "example_call": "count_words('a b')",
        "example_result": "2",
    }
    return {**entry, **changes}


class TestCheckText:
    """The text, names and code are checked before anything runs."""

    def test_a_well_formed_rewrite_passes(self) -> None:
        assert function_pilot.check_text(_entry(), _row()) == []

    def test_the_function_must_be_named_in_the_text(self) -> None:
        entry = _entry(prompt="Write a function that counts words.")
        problems = function_pilot.check_text(entry, _row())
        assert any("doesn't name the function" in problem for problem in problems)

    def test_a_generic_name_is_rejected(self) -> None:
        problems = function_pilot.check_text(_entry(function_name="task_func"), _row())
        assert any("descriptive" in problem for problem in problems)

    def test_naming_a_library_is_a_leak(self) -> None:
        entry = _entry(prompt=_entry()["prompt"] + " Use pandas.")
        problems = function_pilot.check_text(entry, _row())
        assert any("names libraries: pandas" in problem for problem in problems)

    def test_a_library_the_original_prompt_names_is_allowed(self) -> None:
        row = _row()
        row["prompt"] += " The text may be a regex pattern."
        entry = _entry(prompt=_entry()["prompt"] + " The text may be a regex pattern.")
        assert function_pilot.check_text(entry, row) == []

    def test_every_case_needs_a_transcribed_output(self) -> None:
        problems = function_pilot.check_text(
            _entry(transcribed_outputs=["3", "1"]),
            _row(),
        )
        assert any("transcribed_outputs has 2 entries" in p for p in problems)

    def test_the_library_gold_must_use_a_library(self) -> None:
        entry = _entry(
            extlib_solution="def count_words(text):\n    return len(text.split())\n"
        )
        problems = function_pilot.check_text(entry, _row(with_library_gold=True))
        assert "extlib_solution doesn't use a third-party library" in problems


class TestCheckBehaviour:
    """The tests are converted by running the stdlib gold, and checked against the original."""

    @pytest.fixture(autouse=True)
    def _run_with_this_python(self, monkeypatch) -> None:
        monkeypatch.setattr(function_pilot, "VALIDATION_PYTHON", sys.executable)

    def test_tests_become_argument_and_return_value_pairs(self) -> None:
        problems, test_inputs, expected_outputs = function_pilot.check_behaviour(
            _entry(),
            _row(),
        )
        assert problems == []
        assert test_inputs == ["['a b c\\n']", "['x\\n']", "['\\n']"]
        assert expected_outputs == ["3", "1", function_pilot.RAISES]

    def test_a_result_that_differs_from_the_original_is_rejected(self) -> None:
        entry = _entry(
            stdlib_solution="def count_words(text):\n    return len(text.split()) + 1\n",
        )
        problems, _, _ = function_pilot.check_behaviour(entry, _row())
        assert any("test 1" in problem for problem in problems)
        assert any("test 3" in problem for problem in problems)

    def test_a_wrong_example_is_rejected(self) -> None:
        problems, _, _ = function_pilot.check_behaviour(
            _entry(example_result="3"),
            _row(),
        )
        assert any("example call" in problem for problem in problems)

    def test_return_values_must_be_literals(self) -> None:
        entry = _entry(
            stdlib_solution="def count_words(text):\n    return object()\n",
            transcribed_outputs=["3", "1", "0"],
        )
        problems, _, _ = function_pilot.check_behaviour(entry, _row())
        assert any("isn't a python literal" in problem for problem in problems)


def test_weak_tests_are_flagged() -> None:
    assert function_pilot.weak_tests(["None", "None", "1"])
    assert not function_pilot.weak_tests(["None", "2", "1"])


def test_a_rejected_attempt_is_retried_with_its_problems(monkeypatch) -> None:
    """The first reply leaks a library; its problem is fed back, and the second is accepted."""
    monkeypatch.setattr(function_pilot, "VALIDATION_PYTHON", sys.executable)
    replies = [
        _entry(prompt=_entry()["prompt"] + " Use pandas."),
        _entry(),
    ]
    requests = []

    def fake_complete_json(prompt: str, **kwargs: Any) -> tuple[dict, str]:
        requests.append(prompt)
        return replies[len(requests) - 1], "claude-opus-5-5"

    monkeypatch.setattr(function_pilot, "complete_json", fake_complete_json)
    seed = {
        "instruct_prompt": "Count the words.",
        "complete_prompt": "def task_func(text):\n",
        "canonical_solution": "    return len(text.split())\n",
        "test": "assert task_func('a b') == 2",
    }

    entry = function_pilot.generate_function_task(_row(), seed)

    assert entry is not None
    assert len(requests) == 2
    assert "names libraries: pandas" in requests[1]
    assert entry["expected_outputs"] == ["3", "1", function_pilot.RAISES]
    assert entry["prompt_version"] == function_pilot.PROMPT_VERSION


def test_rows_keep_the_benchmark_fields_and_skip_unfitting_tasks() -> None:
    entry = {
        **_entry(),
        "test_inputs": ["['a b']"],
        "expected_outputs": ["2"],
        "weak_tests": False,
    }
    unfitting = {**_entry(fits_function=False)}
    other_row = {**_row(), "task_id": "LRB/C998"}

    rows = function_pilot.build_rows(
        [_row(), other_row],
        {"LRB/C999": entry, "LRB/C998": unfitting},
    )

    assert len(rows) == 1
    assert rows[0]["function_name"] == "count_words"
    assert rows[0]["source"]["variant"] == "function"
    assert rows[0]["stdio_prompt"] == _row()["prompt"]
    assert rows[0]["ground_truth"] == {"gold_stdlib": entry["stdlib_solution"]}
