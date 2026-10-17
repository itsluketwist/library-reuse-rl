"""Tests for the staged script wordings: the template check, the generator, and the build rows."""

from pathlib import Path
from typing import Any

import pytest

from data.build.contexts import Context
from data.build.library import augment
from data.build.library.augment import (
    SCRIPT_STAGES,
    check_script_stage,
    generate_script_stages,
    pilot_seed_tasks,
)
from data.build.library.build import build_library_usage_rows, build_script_pilot_rows


def _statement(
    extra_output_rules: int = 0,
) -> str:
    """A short statement in LibReuseBench's template, padded with `extra_output_rules` lines."""
    return (
        "Write a program that reads timestamps from standard input and prints each one "
        "converted to UTC.\n\n"
        "Input format:\n"
        "- The first line contains an integer N.\n"
        "- Each of the next N lines holds an ISO 8601 timestamp with a UTC offset.\n\n"
        "Output format:\n"
        "- Print N lines, each the matching timestamp in UTC, in ISO 8601 form.\n"
        + "- Keep the order of the input.\n"
        * extra_output_rules
        + "\nExample:\nInput:\n1\n2024-03-10T08:00:00+02:00\n\n"
        "Output:\n2024-03-10T06:00:00+00:00\n"
    )


def _stage_entry(
    fits_stdio: bool = True,
) -> dict[str, Any]:
    """A cache entry whose three statements all pass check_script_stage."""
    statements = {
        "outline": _statement(),
        "precise": _statement(extra_output_rules=8),
        "benchmark": _statement(extra_output_rules=20),
    }
    return {
        "fits_stdio": fits_stdio,
        "reason": "" if fits_stdio else "needs a binary file",
        "exercise": "convert timestamps to UTC",
        **(statements if fits_stdio else dict.fromkeys(SCRIPT_STAGES, "")),
        "model": "claude-test",
    }


class TestCheckScriptStage:
    """Tests for check_script_stage, the deterministic template check."""

    def test_every_stage_of_a_well_formed_entry_passes(self):
        """A statement in the benchmark's template, of the right length, has no problems."""
        entry = _stage_entry()
        for stage in SCRIPT_STAGES:
            assert check_script_stage(entry[stage], stage, ["pytz"]) == []

    def test_missing_example(self):
        """A statement with no "Example:" section is rejected."""
        text = _statement().split("\nExample:")[0]
        assert any("Example" in p for p in check_script_stage(text, "outline", []))

    def test_two_examples(self):
        """A statement must have exactly one example, like the benchmark's."""
        text = _statement() + "\nExample:\nInput:\n0\n\nOutput:\n\n"
        assert any("2" in p for p in check_script_stage(text, "outline", []))

    def test_missing_input_format(self):
        """The "Input format:" section is required."""
        text = _statement().replace("Input format:", "What you get:")
        problems = check_script_stage(text, "outline", [])
        assert 'no "Input format:" section' in problems

    def test_standard_prefixed_headers_are_allowed(self):
        """The benchmark's "Standard input format:" variant counts as the input section."""
        text = _statement().replace("Input format:", "Standard input format:")
        assert check_script_stage(text, "outline", []) == []

    def test_sections_out_of_order(self):
        """The output format must come after the input format, and before the example."""
        text = _statement()
        input_block = text[text.index("Input format:") : text.index("Output format:")]
        output_block = text[text.index("Output format:") : text.index("\nExample:")]
        swapped = text.replace(
            input_block + output_block, output_block + "\n" + input_block
        )
        problems = check_script_stage(swapped, "outline", [])
        assert any("order" in p for p in problems)

    def test_code_is_rejected(self):
        """The benchmark's statements are prose, so python code is a problem."""
        text = _statement().replace("Example:", "import datetime\n\nExample:")
        assert "contains python code" in check_script_stage(text, "outline", [])

    def test_escaped_characters_are_rejected(self):
        """A character written as a literal json escape ("\\u00b1") is a problem."""
        text = _statement().replace(
            "ISO 8601 form", "ISO 8601 form, +HH:MM or \\u00b1HH:MM"
        )
        problems = check_script_stage(text, "outline", [])
        assert any("escaped character" in p for p in problems)

    def test_category_library_name_is_a_leak(self):
        """Naming one of the category's libraries is rejected."""
        text = _statement().replace("each one", "each one using pytz")
        problems = check_script_stage(text, "outline", ["pytz"])
        assert any("pytz" in p for p in problems)

    def test_talk_of_libraries_is_a_leak(self):
        """Mentioning libraries at all is rejected, even without naming one."""
        text = _statement().replace("each one", "each one (any library may be used)")
        problems = check_script_stage(text, "outline", [])
        assert any("library" in p for p in problems)

    def test_format_names_are_not_leaks(self):
        """A data format sharing a library's name ("YAML") is a domain word, not a leak."""
        text = _statement().replace(
            "ISO 8601 timestamp", "YAML-quoted ISO 8601 timestamp"
        )
        assert check_script_stage(text, "outline", ["yaml", "pyyaml"]) == []

    def test_too_short_for_the_benchmark_stage(self):
        """Each stage has length bounds - the benchmark stage needs benchmark-like detail."""
        problems = check_script_stage(_statement(), "benchmark", [])
        assert any("characters" in p for p in problems)


class TestGenerateScriptStages:
    """Tests for generate_script_stages, with the LLM call replaced by canned replies."""

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
        """One bad statement rejects the whole attempt, and the next good one is kept."""
        good = {k: v for k, v in _stage_entry().items() if k != "model"}
        bad = {**good, "precise": "Print the answer."}
        calls = self._fake_llm(monkeypatch, [bad, good])

        entry = generate_script_stages("task", ["pytz"])

        assert len(calls) == 2
        # the second attempt is told what was wrong with the first
        assert "previous attempt was rejected" not in calls[0]
        assert "precise: " in calls[1].split("previous attempt was rejected")[1]
        assert entry is not None
        assert entry["model"] == "claude-test"
        # the statements are kept as written, minus surrounding whitespace
        assert [entry[stage] for stage in SCRIPT_STAGES] == [
            good[stage].strip() for stage in SCRIPT_STAGES
        ]

    def test_task_that_does_not_fit_stdio(self, monkeypatch: pytest.MonkeyPatch):
        """A task judged not to fit stdin/stdout is cached with empty statements."""
        reply = {
            **{k: v for k, v in _stage_entry().items() if k != "model"},
            "fits_stdio": False,
            "reason": "needs a binary file",
        }
        self._fake_llm(monkeypatch, [reply])

        entry = generate_script_stages("task", [])

        assert entry is not None
        assert entry["fits_stdio"] is False
        assert all(entry[stage] == "" for stage in SCRIPT_STAGES)

    def test_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch):
        """Every attempt failing returns None, so nothing is cached for the task."""
        bad = {
            **{k: v for k, v in _stage_entry().items() if k != "model"},
            "outline": "Print it.",
        }
        calls = self._fake_llm(monkeypatch, [bad, bad])

        assert generate_script_stages("task", [], max_attempts=2) is None
        assert len(calls) == 2


# a one-category taxonomy and its building blocks, for the build tests below
_TAXONOMY = {
    "code": [
        {
            "id": "timezones",
            "seed_libraries": ["pytz"],
            "repo_hint": "toolbox",
            "benchmark_aligned": True,
        }
    ]
}
_TASKS = {"timezones": {1: "Convert a timestamp to UTC.", 2: "Play a sound file."}}
_TEMPLATES = {
    "repo": ["{context}\n\nTask: {task}"],
    "repo_project": ["{task} Use it in my project."],
    "simple": ["How do I {task_lower} in Python?"],
    "script": ["Write a script that reads standard input to {task_stdio}."],
}
_CONTEXTS = {
    "toolbox": Context(
        context_id="toolbox",
        stack="python",
        language="python",
        summary="# toolbox",
        local_modules=["toolbox"],
    )
}


class TestBuildRows:
    """Tests for the build's v4 rows and the script pilot's rows."""

    def test_function_rows_replace_the_script_rows(self):
        """A task gets 3 function rows and no script rows; one with no example gets 2."""
        rows = build_library_usage_rows(
            _TAXONOMY,
            _TASKS,
            _TEMPLATES,
            _CONTEXTS,
            prompt_caches={"simple": {}, "repo": {}},
            function_cache={
                "timezones.1": {"name": "N1", "signature": "S1", "example": "E1"},
                # a task with no honest example has an empty example stage
                "timezones.2": {"name": "N2", "signature": "S2", "example": ""},
            },
        )

        def function_rows(seed_task: str) -> list[tuple[str, str]]:
            return [
                (row["function_stage"], row["prompt"])
                for row in rows
                if row["seed_task"] == seed_task and row["phrasing_mode"] == "function"
            ]

        assert not any(row["phrasing_mode"] == "script" for row in rows)
        assert function_rows("timezones.1") == [
            ("name", "N1"),
            ("signature", "S1"),
            ("example", "E1"),
        ]
        assert function_rows("timezones.2") == [("name", "N2"), ("signature", "S2")]
        # function rows are claude wordings with no repo context
        function = [row for row in rows if row["phrasing_mode"] == "function"]
        assert all(
            r["source"] == "llm"
            and r["context_id"] == "none"
            and r["local_modules"] == []
            for r in function
        )
        # repo and simple rows carry no stage; repo has the picked template row and the
        # project-framed one
        other = [row for row in rows if row["phrasing_mode"] != "function"]
        assert len(other) == 6
        assert all(
            r["script_stage"] is None and r["function_stage"] is None for r in other
        )
        assert sorted(r["source"] for r in other if r["phrasing_mode"] == "repo") == [
            "project",
            "project",
            "template",
            "template",
        ]
        # the project row is the request alone - no repo context, so no local modules
        project = [r for r in other if r["source"] == "project"]
        assert all(
            r["context_id"] == "none" and r["local_modules"] == [] for r in project
        )
        assert (
            project[0]["prompt"] == "Convert a timestamp to UTC. Use it in my project."
        )

    def test_pilot_rows_hold_old_and_new_wordings(self):
        """The pilot scores the template, the v2 wording and the three stages side by side."""
        rows = build_script_pilot_rows(
            _TAXONOMY,
            _TASKS,
            _TEMPLATES,
            v2_script_cache={"timezones.1": ["The old v2 wording."]},
            stage_cache={"timezones.1": _stage_entry()},
            pilot_tasks=["timezones.1"],
        )

        assert [row["script_stage"] for row in rows] == [
            "template",
            "v2",
            "outline",
            "precise",
            "benchmark",
        ]
        assert rows[1]["prompt"] == "The old v2 wording."
        assert [row["task_id"] for row in rows] == [f"LRRL/S{i:04d}" for i in range(5)]
        assert all(row["phrasing_mode"] == "script" for row in rows)


def test_pilot_sample_is_reproducible_and_balanced():
    """The pilot takes one task per category, half aligned, the same every time."""
    # 4 aligned categories ("a0".."a3") and 4 unaligned ones ("u0".."u3"), 2 tasks each
    categories = [(f"{kind}{i}", kind == "a") for kind in ("a", "u") for i in range(4)]
    taxonomy: dict[str, Any] = {
        "code": [
            {"id": category, "benchmark_aligned": aligned}
            for category, aligned in categories
        ]
    }
    tasks: dict[str, dict[int, str]] = {
        category: {1: "x", 2: "y"} for category, _ in categories
    }

    sample = pilot_seed_tasks(taxonomy, tasks, n=4)

    assert sample == pilot_seed_tasks(taxonomy, tasks, n=4)
    categories = [seed_task.split(".")[0] for seed_task in sample]
    assert len(set(categories)) == 4
    assert sum(category.startswith("a") for category in categories) == 2


def test_resume_skips_only_entries_from_the_current_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """--skip-cached regenerates an entry made with an older prompt, and keeps a current one."""
    monkeypatch.setattr(augment, "SCRIPT_STAGE_CACHE_FILE", tmp_path / "cache.json")
    current = {**_stage_entry(), "prompt_version": augment.SCRIPT_STAGE_PROMPT_VERSION}
    stale = {**_stage_entry(), "prompt_version": "an-older-prompt"}
    augment.write_script_stage_cache({"timezones.1": current, "timezones.2": stale})

    generated: list[str] = []

    def fake_generate(task: str, **kwargs: Any) -> dict[str, Any]:
        generated.append(task)
        return current

    monkeypatch.setattr(augment, "generate_script_stages", fake_generate)
    taxonomy = {
        "code": [
            {
                "id": "timezones",
                "name": "Timezones",
                "diy_temptation": "offset maths by hand",
                "seed_libraries": ["pytz"],
            }
        ]
    }

    augment.regenerate_script_stage_cache(taxonomy, _TASKS, model="m", skip_cached=True)

    assert generated == [_TASKS["timezones"][2]]


def test_parallel_workers_generate_every_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """With several workers every task is still generated once and cached, and a task whose
    generation raises is left uncached rather than stopping the run."""
    monkeypatch.setattr(augment, "SCRIPT_STAGE_CACHE_FILE", tmp_path / "cache.json")

    def fake_generate(task: str, **kwargs: Any) -> dict[str, Any]:
        if task == _TASKS["timezones"][2]:
            raise TimeoutError("the api timed out")
        return {**_stage_entry(), "prompt_version": augment.SCRIPT_STAGE_PROMPT_VERSION}

    monkeypatch.setattr(augment, "generate_script_stages", fake_generate)
    taxonomy = {"code": [{"id": "timezones", "seed_libraries": ["pytz"]}]}

    augment.regenerate_script_stage_cache(taxonomy, _TASKS, model="m", workers=2)

    assert sorted(augment.load_script_stage_cache()) == ["timezones.1"]
