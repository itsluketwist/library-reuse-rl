"""Tests for stage 02 of the benchmark build: the checks on a generated task, and the retry loop.

The stage is a script (benchmark/pipeline/02_generate.py), so it is loaded by path. No model is
called and nothing is installed: the request is stubbed, and generated code runs with this test's
own python.
"""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import build
import pytest
from libreusebench.environment import InstallResult


_SCRIPT = Path(__file__).parents[2] / "benchmark" / "pipeline" / "02_generate.py"
_spec = importlib.util.spec_from_file_location("generate", _SCRIPT)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)


# a library solution and a standard-library solution of the same small task: read a date
_EXTLIB_SOLUTION = (
    "from dateutil import parser\n\n\n"
    "def parse_date_parts(text):\n"
    "    try:\n"
    "        moment = parser.parse(text)\n"
    "    except (ValueError, OverflowError) as error:\n"
    "        raise ValueError('not a date') from error\n"
    "    return (moment.year, moment.month, moment.day)\n"
)
_STDLIB_SOLUTION = (
    "from datetime import datetime\n\n\n"
    "def parse_date_parts(text):\n"
    "    for pattern in ('%Y-%m-%d', '%d %B %Y', '%B %d, %Y'):\n"
    "        try:\n"
    "            moment = datetime.strptime(text, pattern)\n"
    "        except ValueError:\n"
    "            continue\n"
    "        return (moment.year, moment.month, moment.day)\n"
    "    raise ValueError('not a date')\n"
)
_TEST_INPUTS = [
    "['2024-03-05']",
    "['1999-12-31']",
    "['6 March 2024']",
    "['17 July 2001']",
    "['March 7, 2024']",
    "['January 1, 2000']",
    "['2010-10-10']",
    "['28 February 2023']",
    "['August 9, 1985']",
    "['not a date']",
]
_EXPECTED_OUTPUTS = [
    "(2024, 3, 5)",
    "(1999, 12, 31)",
    "(2024, 3, 6)",
    "(2001, 7, 17)",
    "(2024, 3, 7)",
    "(2000, 1, 1)",
    "(2010, 10, 10)",
    "(2023, 2, 28)",
    "(1985, 8, 9)",
    generate.RAISES,
]


def _entry(
    **changes: Any,
) -> dict[str, Any]:
    """A well-formed reuse-track reply, with any fields replaced by `changes`."""
    entry = {
        "verdict": "task",
        "reason": "",
        "function_name": "parse_date_parts",
        "prompt": (
            "Write a function `parse_date_parts(text)` that reads a date written in a "
            "common format, such as '2024-03-05', '5 March 2024' or 'March 5, 2024', and "
            "returns a tuple (year, month, day) of ints. It raises a ValueError if the "
            "text is not a date. For example, `parse_date_parts('2024-03-05')` returns "
            "`(2024, 3, 5)`."
        ),
        "domain": "dates and times",
        "category": "date-parsing",
        "primary_library": "dateutil",
        "stdlib_solution": _STDLIB_SOLUTION,
        "extlib_solution": _EXTLIB_SOLUTION,
        "extlib_packages": ["python-dateutil"],
        "test_inputs": list(_TEST_INPUTS),
        "example_call": "parse_date_parts('2024-03-05')",
        "example_result": "(2024, 3, 5)",
    }
    return {**entry, **changes}


def _control_entry(
    **changes: Any,
) -> dict[str, Any]:
    """A well-formed control-track reply: the same task, without the library fields."""
    entry = _entry(**changes)
    for field in ("primary_library", "extlib_solution", "extlib_packages"):
        entry.pop(field, None)
    return entry


_SEED = {
    "seed_id": "BigCodeBench/1",
    "instruct_prompt": "Parse a date string.",
    "complete_prompt": "from dateutil import parser\ndef task_func(text):\n",
    "canonical_solution": "    return parser.parse(text)\n",
    "test": "assert task_func('2024-03-05').year == 2024",
    "ext_libs": ["dateutil"],
}


@contextmanager
def _no_install(
    python_executable: str,
    packages: list[str],
) -> Iterator[InstallResult]:
    """Stand in for temporary_packages: install nothing, and report a version for each package."""
    yield InstallResult(
        failed=set(),
        versions={package: "1.0" for package in packages},
    )


@pytest.fixture(autouse=True)
def _run_locally(monkeypatch) -> None:
    """Run generated code with this test's python, and never install anything."""
    monkeypatch.setattr(generate, "BUILD_PYTHON", sys.executable)
    monkeypatch.setattr(generate, "temporary_packages", _no_install)


class TestBuildRequest:
    """The request shows the seed, and an earlier reply with what was wrong with it."""

    def test_the_reuse_request_has_the_seed_and_the_example_texts(self) -> None:
        request = generate.build_request(_SEED, "reuse")
        assert "Parse a date string." in request
        assert "which uses dateutil" in request
        assert "Reword the seed" in request and "## The wording we want" in request

    def test_the_request_sets_no_target_for_the_solutions(self) -> None:
        """How much code a library saves is measured afterwards, never asked for."""
        for track in ("reuse", "control"):
            request = generate.build_request(_SEED, track).lower()
            for word in (
                "lines of code",
                "logical lines",
                "reuse value",
                "longer than",
            ):
                assert word not in request, word

    def test_the_control_request_asks_for_no_library(self) -> None:
        request = generate.build_request(_SEED | {"ext_libs": []}, "control")
        assert "which uses only the standard library" in request
        assert "CONTROL split" in request
        assert "a third-party library would clearly help with it" in request
        assert "extlib_solution" not in request

    def test_the_request_says_what_a_task_is_for(self) -> None:
        """The reason behind the rules is stated: models write too many custom
        implementations, and a task must be feasible both ways to measure that."""
        request = " ".join(generate.build_request(_SEED, "reuse").split())
        assert "known to write too many custom implementations" in request
        assert "We call such a task CONTESTED" in request
        assert "Both answers must be realistic" in request
        assert "it is too difficult without a library" in request
        assert "the library has no real job in it" in request

    def test_the_request_allows_a_small_change_to_make_a_task_contested(self) -> None:
        request = " ".join(generate.build_request(_SEED, "reuse").split())
        assert "You may make a small change to the seed's task" in request
        assert "if the seed is too difficult by hand, narrow it" in request
        assert "Change as little as you can" in request
        # a control task has no such choice to make
        control = " ".join(generate.build_request(_SEED, "control").split())
        assert "contested" not in control.lower()

    def test_feedback_shows_the_earlier_reply_and_its_problems(self) -> None:
        request = generate.build_request(
            _SEED,
            "reuse",
            previous=_entry(),
            problems=["the text names libraries: pandas"],
        )
        assert "- the text names libraries: pandas" in request
        assert '"function_name": "parse_date_parts"' in request


class TestCheckText:
    """The text, names, code and test inputs are checked before anything runs."""

    def test_a_well_formed_reply_passes(self) -> None:
        assert generate.check_text(_entry()) == []
        assert generate.check_text(_control_entry(), "control") == []

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            ({"prompt": "Write a function that reads a date."}, "doesn't name the"),
            ({"function_name": "task_func"}, "not a descriptive snake_case name"),
            ({"example_result": "(2024, 3, 6)"}, "example_result doesn't appear"),
            ({"example_call": "read('2024-03-05')"}, "example_call doesn't call"),
            ({"prompt": _entry()["prompt"] + " x" * 400}, "over the 900 limit"),
            # a library from the benchmark's blocklist, and the task's own library
            ({"prompt": _entry()["prompt"] + " Use pandas."}, "libraries: pandas"),
            ({"prompt": _entry()["prompt"] + " Use dateutil."}, "libraries: dateutil"),
            (
                {"stdlib_solution": "import numpy\n" + _STDLIB_SOLUTION},
                "stdlib_solution imports a third-party library",
            ),
            (
                {"extlib_solution": _STDLIB_SOLUTION},
                "extlib_solution doesn't use a third-party library",
            ),
            ({"test_inputs": _TEST_INPUTS[:9]}, "fewer than the 10 needed"),
            ({"test_inputs": _TEST_INPUTS[:9] + ["'x'"]}, "not python list literals"),
            ({"test_inputs": _TEST_INPUTS[:9] * 2}, "test_inputs repeats a case"),
        ],
    )
    def test_each_check_catches_its_failure(
        self,
        changes: dict[str, Any],
        problem: str,
    ) -> None:
        problems = generate.check_text(_entry(**changes))
        assert any(problem in found for found in problems), problems

    def test_a_format_named_like_its_library_is_not_a_leak(self) -> None:
        """A YAML task must be able to say "YAML", though its library is called yaml."""
        entry = _entry(
            primary_library="yaml",
            prompt=_entry()["prompt"] + " The text may also be a YAML date.",
        )
        assert generate.check_text(entry) == []
        entry["prompt"] += " Parse it with yaml."
        assert any("libraries: yaml" in found for found in generate.check_text(entry))


class TestCheckBehaviour:
    """The expected values are taken by running the reference solution, then checked."""

    def test_expected_values_come_from_the_library_solution(self) -> None:
        problems, checked = generate.check_behaviour(_entry())
        assert problems == []
        assert checked == {
            "expected_outputs": _EXPECTED_OUTPUTS,
            "gold_extlib_packages": ["python-dateutil==1.0"],
        }

    def test_the_control_track_uses_its_only_solution(self) -> None:
        problems, checked = generate.check_behaviour(_control_entry(), "control")
        assert problems == []
        assert checked == {"expected_outputs": _EXPECTED_OUTPUTS}

    @pytest.mark.parametrize(
        ("changes", "problem"),
        [
            # the standard-library solution gets a date wrong
            (
                {"stdlib_solution": _STDLIB_SOLUTION.replace("moment.day)", "1)")},
                "stdlib_solution disagrees with extlib_solution",
            ),
            ({"example_result": "(2024, 3, 6)"}, "the example call gives (2024, 3, 5)"),
            # an example whose number is rounded, where the function returns it in full
            (
                {
                    "extlib_solution": (
                        "def parse_date_parts(text):\n    return len(text) / 3\n"
                    ),
                    "stdlib_solution": (
                        "def parse_date_parts(text):\n    return len(text) / 3\n"
                    ),
                    "example_result": "3.333",
                },
                "show the result exactly as returned",
            ),
            # a result that can't be written as a literal: the datetime itself
            (
                {
                    "extlib_solution": _EXTLIB_SOLUTION.replace(
                        "return (moment.year, moment.month, moment.day)",
                        "return moment",
                    )
                },
                "returned a datetime that can't be written",
            ),
            # a result that changes from one run to the next
            (
                {
                    "extlib_solution": (
                        "import random\n\n\n"
                        "def parse_date_parts(text):\n"
                        "    return random.random()\n"
                    )
                },
                "gave a different result on a second run",
            ),
            # tests that can't tell answers apart
            (
                {
                    "extlib_solution": (
                        "def parse_date_parts(text):\n    return len(text) > 12\n"
                    )
                },
                "tests expect the same result",
            ),
            (
                {
                    "test_inputs": _TEST_INPUTS[:6]
                    + ["['a']", "['b']", "['c']", "['d']"]
                },
                "raises on 4 tests, more than the 3 allowed",
            ),
        ],
    )
    def test_each_check_catches_its_failure(
        self,
        changes: dict[str, Any],
        problem: str,
    ) -> None:
        problems, checked = generate.check_behaviour(_entry(**changes))
        assert any(problem in found for found in problems), problems
        assert checked == {}

    def test_a_library_value_is_turned_into_plain_python(self) -> None:
        """A numpy-like result (anything with tolist) is accepted as the list it holds."""
        solution = (
            "class Array:\n"
            "    def __init__(self, values):\n"
            "        self.values = values\n\n"
            "    def tolist(self):\n"
            "        return self.values\n\n\n"
            "def parse_date_parts(text):\n"
            "    return Array([len(text), text.count('a')])\n"
        )
        entry = _control_entry(
            stdlib_solution=solution,
            example_result="[10, 0]",
        )
        problems, checked = generate.check_behaviour(entry, "control")
        assert problems == []
        assert checked["expected_outputs"][0] == "[10, 0]"

    def test_a_set_may_print_in_any_order(self) -> None:
        assert generate._same_result("{'a', 'b'}", "{'b', 'a'}")
        assert not generate._same_result("['a', 'b']", "['b', 'a']")
        assert not generate._same_result("1", generate.RAISES)


def _stub_model(
    monkeypatch,
    replies: list[dict],
) -> list[str]:
    """Answer each model call with the next of `replies`.

    Returns the list the requests are recorded in.
    """
    requests: list[str] = []

    def fake_complete_json(prompt: str, **kwargs: Any) -> tuple[dict, str]:
        requests.append(prompt)
        return replies[len(requests) - 1], "claude-opus-5-5"

    monkeypatch.setattr(generate, "complete_json", fake_complete_json)
    return requests


class TestGenerateTask:
    """A reply that fails a check is fed back, and every seed ends with one record."""

    def test_a_rejected_reply_is_fed_back_and_the_next_is_kept(
        self,
        monkeypatch,
    ) -> None:
        leaking = _entry(prompt=_entry()["prompt"] + " Use pandas.")
        requests = _stub_model(monkeypatch, [leaking, _entry()])

        record = generate.generate_task(_SEED)

        assert len(requests) == 2
        assert "the text names libraries: pandas" in requests[1]
        assert "Use pandas." in requests[1]  # the rejected reply itself is shown too
        assert record["kept"] and record["drop_reason"] is None
        assert record["attempts"] == 2
        assert record["expected_outputs"] == _EXPECTED_OUTPUTS
        assert record["task_hash"] == build.task_hash(record)
        assert record["prompt_version"] == generate.PROMPT_VERSION
        assert record["seed_id"] == "BigCodeBench/1"

    def test_a_low_reuse_value_is_recorded_not_sent_back(self, monkeypatch) -> None:
        """The two date solutions are about as long as each other, and that is kept as it is."""
        requests = _stub_model(monkeypatch, [_entry()])

        record = generate.generate_task(_SEED)

        assert len(requests) == 1
        assert record["kept"]
        assert record["reuse_tier"] == "low"
        assert record["reuse_value"] == record["stdlib_lloc"] - record["extlib_lloc"]

    def test_a_skip_is_recorded_with_its_reason(self, monkeypatch) -> None:
        skip = _entry(verdict="skip", reason="needs the network")
        requests = _stub_model(monkeypatch, [skip])

        record = generate.generate_task(_SEED)

        assert len(requests) == 1
        assert not record["kept"]
        assert record["drop_reason"] == "skipped"
        assert record["reason"] == "needs the network"

    def test_a_seed_that_never_passes_is_recorded_as_failed(self, monkeypatch) -> None:
        leaking = _entry(prompt=_entry()["prompt"] + " Use pandas.")
        requests = _stub_model(monkeypatch, [leaking] * generate.MAX_ATTEMPTS)

        record = generate.generate_task(_SEED)

        assert len(requests) == generate.MAX_ATTEMPTS
        assert not record["kept"]
        assert record["drop_reason"] == "checks failed"
        assert record["problems"] == ["the text names libraries: pandas"]

    def test_an_unreadable_reply_is_recorded(self, monkeypatch) -> None:
        _stub_model(monkeypatch, [{}] * generate.MAX_ATTEMPTS)
        record = generate.generate_task(_SEED)
        assert record["drop_reason"] == "no usable reply"

    def test_the_control_track_has_no_reuse_value(self, monkeypatch) -> None:
        _stub_model(monkeypatch, [_control_entry()])
        record = generate.generate_task(_SEED | {"ext_libs": []}, track="control")
        assert record["kept"]
        assert "reuse_tier" not in record and "extlib_solution" not in record


class TestReuseStats:
    """The reuse value is the lines a library saves, and sets the task's tier."""

    @staticmethod
    def _code(
        lines: int,
    ) -> str:
        """A program of `lines` statements."""
        return "\n".join(f"x{number} = {number}" for number in range(lines))

    @pytest.mark.parametrize(
        ("stdlib_lines", "extlib_lines", "value", "tier"),
        [
            (60, 30, 30, "high"),
            (
                45,
                30,
                15,
                "high",
            ),  # exactly the bounds: 15 lines saved, 1.5 times as long
            (
                120,
                100,
                20,
                "medium",
            ),  # many lines saved, but the library barely shortens it
            (20, 12, 8, "medium"),
            (10, 8, 2, "low"),
            (8, 10, -2, "low"),  # the library solution is the longer one
        ],
    )
    def test_tiers(
        self,
        stdlib_lines: int,
        extlib_lines: int,
        value: int,
        tier: str,
    ) -> None:
        stats = build.reuse_stats(
            stdlib_solution=self._code(stdlib_lines),
            extlib_solution=self._code(extlib_lines),
        )
        assert stats["stdlib_lloc"] == stdlib_lines
        assert stats["extlib_lloc"] == extlib_lines
        assert stats["reuse_value"] == value
        assert stats["reuse_tier"] == tier


class TestFixRequest:
    """A reviewer's note is fed back only for the very task it was written about."""

    def test_a_fix_for_the_current_task_is_fed_back(self) -> None:
        done = {"kept": True, "task_hash": "abc"}
        review = {"verdict": "fix", "task_hash": "abc", "note": "pin the tie order"}
        previous, problems = generate.fix_request(done, review)
        assert previous is done
        assert problems == [
            "an independent reviewer asked for a fix: pin the tie order"
        ]

    @pytest.mark.parametrize(
        ("done", "review"),
        [
            (None, None),
            ({"kept": True, "task_hash": "abc"}, None),
            # the review kept the task, or was of an earlier version of it
            (
                {"kept": True, "task_hash": "abc"},
                {"verdict": "keep", "task_hash": "abc", "note": ""},
            ),
            (
                {"kept": True, "task_hash": "new"},
                {"verdict": "fix", "task_hash": "old", "note": "x"},
            ),
        ],
    )
    def test_nothing_else_is(
        self,
        done: dict | None,
        review: dict | None,
    ) -> None:
        assert generate.fix_request(done, review) == (None, [])
