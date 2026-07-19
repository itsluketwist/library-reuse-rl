"""Tests for response parsing."""

from src.response.parser import (
    code_metrics,
    parse_response,
    select_code,
    strip_reasoning,
)


class TestStripReasoning:
    """Tests for strip_reasoning."""

    def test_with_think_tag(self):
        """A <think>...</think> preamble is split off into reasoning, tags stripped."""
        reasoning, answer = strip_reasoning("<think>working it out</think>the answer")

        assert reasoning == "working it out"
        assert answer == "the answer"

    def test_with_alternate_tag_name(self):
        """Other reasoning tag names (e.g. <reasoning>) are also recognised."""
        _, answer = strip_reasoning("<reasoning>steps</reasoning>final")

        assert answer == "final"

    def test_no_tag(self):
        """A response with no reasoning tag is treated entirely as the answer."""
        reasoning, answer = strip_reasoning("just the answer, no preamble")

        assert reasoning == ""
        assert answer == "just the answer, no preamble"


class TestSelectCode:
    """Tests for select_code."""

    def test_picks_longest_fenced_block(self):
        """Given multiple fenced code blocks, the longest one is selected."""
        answer = """
short one:
```python
x = 1
```
longer one:
```python
def add(a, b):
    return a + b

print(add(1, 2))
```
"""
        block = select_code(answer)

        assert "def add" in block.text

    def test_no_fenced_blocks_gives_empty_fragment(self):
        """An answer with no fenced code blocks yields an empty code fragment."""
        block = select_code("x = 1\nprint(x)")

        assert block.text == ""


class TestLibraryUsage:
    """Tests for the internal (std) vs external (ext) library split."""

    def test_std_and_ext_libs(self):
        """A fragment importing both stdlib and third-party modules is split correctly."""
        answer = """
```python
import json
import numpy as np

data = json.dumps({"a": 1})
arr = np.array([1, 2, 3])
```
"""
        block = select_code(answer)

        assert block.std_libs == ["json"]
        assert block.ext_libs == ["numpy"]
        assert "numpy" in block.lib_usage


class TestCodeMetrics:
    """Tests for code_metrics."""

    def test_known_snippet(self):
        """A two-statement function has 2 logical lines."""
        logical_lines, maintainability_index = code_metrics(
            "def add(a, b):\n    return a + b\n"
        )

        assert logical_lines == 2
        assert maintainability_index is not None

    def test_unparseable_code(self):
        """Code that fails to parse returns None metrics instead of raising."""
        logical_lines, maintainability_index = code_metrics(
            "def add(a, b:\n    return a + b"
        )

        assert logical_lines is None
        assert maintainability_index is None


class TestParseResponse:
    """End-to-end tests for parse_response."""

    def test_full_response(self):
        """A response with reasoning and a fenced code block is fully parsed."""
        response = """<think>
I'll use numpy for the array math.
</think>
Here's the solution:

```python
import numpy as np

def total(values):
    return np.sum(values)
```
"""
        parsed = parse_response(response)

        assert "numpy" in parsed.reasoning
        assert "Here's the solution" in parsed.answer
        assert parsed.valid is True
        assert parsed.ext_libs == ["numpy"]
        assert parsed.logical_lines == 3  # import + def + return
        assert parsed.maintainability_index is not None

    def test_overflow_response(self):
        """A response that never closes its reasoning tag has no answer or code."""
        parsed = parse_response("<think>still working it out...")

        assert parsed.reasoning == "still working it out..."
        # nothing was extracted, so everything downstream of the answer is None
        assert parsed.answer is None
        assert parsed.code is None
        assert parsed.valid is None
        assert parsed.logical_lines is None
        assert parsed.maintainability_index is None
