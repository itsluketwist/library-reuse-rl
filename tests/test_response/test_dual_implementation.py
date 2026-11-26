"""Tests for detecting answers that solve a task both with and without a library."""

from src.response.dual_implementation import find_dual_implementation


def fenced(
    *blocks: str,
) -> str:
    """
    Wrap code blocks in a model-style answer, one python fence each.

    Returns the answer text.
    """
    return "\n\nAlternatively:\n\n".join(f"```python\n{block}\n```" for block in blocks)


IMPORT_SWITCH = """
try:
    import numpy as np
except ImportError:
    np = None

def mean(values):
    if np is not None:
        return float(np.mean(values))
    total = 0
    for value in values:
        total += value
    return total / len(values)
"""

IMPORT_REQUIRED = """
try:
    import numpy as np
except ImportError:
    np = None

def mean(values):
    if np is None:
        raise RuntimeError("numpy is required")
    return float(np.mean(values))
"""

CALL_FALLBACK = """
from dateutil import parser

def manual_parse(text):
    year, month, day = text.split("-")
    return int(year), int(month), int(day)

def parse_date(text):
    try:
        return parser.parse(text)
    except Exception:
        return manual_parse(text)
"""

CALL_DEFAULT = """
import numpy as np

def solve(matrix, vector):
    try:
        return np.linalg.solve(matrix, vector).tolist()
    except np.linalg.LinAlgError:
        return [0.0] * len(vector)
"""

CALL_FALL_THROUGH = """
def to_date(text):
    try:
        from dateutil import parser
        return parser.parse(text).date()
    except Exception:
        pass
    year, month, day = text.split("-")
    for part in (year, month, day):
        assert part.isdigit()
    return int(year), int(month), int(day)
"""

LIBRARY_ONLY = """
import pandas as pd

def column_means(path):
    frame = pd.read_csv(path)
    return frame.mean().to_dict()
"""

HAND_WRITTEN = """
def mean(values):
    total = 0
    for value in values:
        total += value
    return total / len(values)
"""


class TestFindDualImplementation:
    """Tests for find_dual_implementation."""

    def test_import_fallback_with_switch(self):
        """An optional import with an `if np is not None` switch to hand-written code is found."""
        assert find_dual_implementation(fenced(IMPORT_SWITCH)) == ["import_fallback"]

    def test_required_import_is_not_flagged(self):
        """An optional import whose only other path raises has one way only."""
        assert find_dual_implementation(fenced(IMPORT_REQUIRED)) == []

    def test_call_fallback(self):
        """A library call with a hand-written fallback in the except clause is found."""
        assert find_dual_implementation(fenced(CALL_FALLBACK)) == ["call_fallback"]

    def test_call_fall_through(self):
        """A library call whose failure falls through to hand-written code below it is found."""
        assert find_dual_implementation(fenced(CALL_FALL_THROUGH)) == ["call_fallback"]

    def test_call_default_is_not_flagged(self):
        """An except clause that only returns a default value is not a second implementation."""
        assert find_dual_implementation(fenced(CALL_DEFAULT)) == []

    def test_extra_block(self):
        """A library answer followed by a separate hand-written version is found."""
        response = fenced(LIBRARY_ONLY, HAND_WRITTEN)
        assert find_dual_implementation(response) == ["extra_block"]

    def test_library_only(self):
        """A plain library answer has one way only."""
        assert find_dual_implementation(fenced(LIBRARY_ONLY)) == []

    def test_hand_written_only(self):
        """A plain hand-written answer has one way only."""
        assert find_dual_implementation(fenced(HAND_WRITTEN)) == []

    def test_reasoning_is_ignored(self):
        """Code in the reasoning doesn't count, like in the benchmark scorer."""
        response = (
            f"<think>\n{fenced(HAND_WRITTEN)}\n</think>\n\n{fenced(LIBRARY_ONLY)}"
        )
        assert find_dual_implementation(response) == []
