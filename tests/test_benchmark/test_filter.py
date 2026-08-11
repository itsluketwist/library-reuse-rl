"""Tests for the stage-02 judge in benchmark/pipeline/02_filter.py.

The filter is a numbered script (not importable by name), so it is loaded by path. Only the pure
prompt/verdict logic is exercised here - no model is called.
"""

import importlib.util
from pathlib import Path


_FILTER_PATH = Path(__file__).parents[2] / "benchmark" / "pipeline" / "02_filter.py"
_spec = importlib.util.spec_from_file_location("filter_stage", _FILTER_PATH)
assert _spec is not None and _spec.loader is not None
filter_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_stage)


class TestJudgePrompt:
    """The judge prompt states the rewrite and lists the criteria."""

    def test_mentions_rewrite_and_stdout(self) -> None:
        prompt = filter_stage.build_judge_prompt("desc", "sol", ["pandas"])
        # the judge must know the problem/solution get rewritten into a stdin/stdout task
        assert "REWRITTEN" in prompt or "rewritten" in prompt
        assert "standard input" in prompt and "standard output" in prompt

    def test_lists_criteria(self) -> None:
        prompt = filter_stage.build_judge_prompt("desc", "sol", ["pandas"])
        assert "dual-solvable" in prompt and "deterministic" in prompt


class TestNormaliseVerdict:
    """The verdict normaliser lowercases valid verdicts and fails safe to drop."""

    def test_valid_verdict_normalised(self) -> None:
        got = filter_stage.normalise_verdict(
            {"verdict": "KEEP", "category": "stats", "reason": "ok"}
        )
        assert got == {"verdict": "keep", "category": "stats", "reason": "ok"}

    def test_maybe_kept(self) -> None:
        assert (
            filter_stage.normalise_verdict({"verdict": "maybe"})["verdict"] == "maybe"
        )

    def test_invalid_defaults_to_drop(self) -> None:
        assert filter_stage.normalise_verdict({"verdict": "yes"})["verdict"] == "drop"

    def test_empty_defaults_to_drop(self) -> None:
        assert filter_stage.normalise_verdict({})["verdict"] == "drop"
