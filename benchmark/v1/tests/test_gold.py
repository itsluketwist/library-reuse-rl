"""Tests for the stage-04 stdlib gold + minting in benchmark/pipeline/04_make_tests.py.

The stage is a numbered script (not importable by name), so it is loaded by path. The prompt
builder is pure; the minting test executes real subprocesses (like the reference solution does).
"""

import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[2] / "benchmark" / "pipeline" / "04_make_tests.py"
_spec = importlib.util.spec_from_file_location("gold_stage", _PATH)
assert _spec is not None and _spec.loader is not None
gold_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gold_stage)


class TestGoldPrompt:
    """The gold prompt asks for a standard-library-only stdin/stdout solution plus test inputs."""

    def test_restricts_to_stdlib_and_stdio(self) -> None:
        prompt = gold_stage.build_gold_prompt("Sum the input numbers.")
        assert "standard library" in prompt
        assert "standard input" in prompt and "standard output" in prompt
        assert "Sum the input numbers." in prompt


class TestMintDeterminism:
    """Minting keeps only inputs the gold runs cleanly and deterministically."""

    def test_keeps_deterministic(self) -> None:
        inputs, expected = gold_stage._mint("print(int(input()) + 1)", ["1\n", "2\n"])
        assert inputs == ["1\n", "2\n"]
        assert expected == ["2", "3"]

    def test_drops_nondeterministic(self) -> None:
        # a gold that depends on randomness gives different output across runs -> rejected
        inputs, expected = gold_stage._mint(
            "import random; print(random.random())", ["\n", "\n"]
        )
        assert inputs == []
        assert expected == []
