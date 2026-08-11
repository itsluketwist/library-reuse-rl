"""Tests for the stage-03 rewrite prompt in benchmark/pipeline/03_rewrite.py.

The rewrite is a numbered script (not importable by name), so it is loaded by path. Only the pure
prompt-building logic is exercised here - no model is called.
"""

import importlib.util
from pathlib import Path


_REWRITE_PATH = Path(__file__).parents[2] / "benchmark" / "pipeline" / "03_rewrite.py"
_spec = importlib.util.spec_from_file_location("rewrite_stage", _REWRITE_PATH)
assert _spec is not None and _spec.loader is not None
rewrite_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rewrite_stage)


class TestRewritePrompt:
    """The rewrite prompt asks for a natural-language, library-agnostic stdin/stdout task."""

    def test_includes_task_and_solution(self) -> None:
        prompt = rewrite_stage.build_rewrite_prompt(
            "Sum ints", "def f(): ...", "assert True"
        )
        assert "Sum ints" in prompt and "def f()" in prompt

    def test_demands_agnostic_natural_language_stdio(self) -> None:
        prompt = rewrite_stage.build_rewrite_prompt("d", "s", "t")
        assert "standard input" in prompt and "standard output" in prompt
        assert "natural" in prompt.lower()
        assert (
            "no specific library" in prompt.lower()
            or "names no specific library" in prompt.lower()
        )

    def test_schema_is_prompt_only(self) -> None:
        assert rewrite_stage._REWRITE_SCHEMA["properties"] == {
            "prompt": {"type": "string"}
        }
