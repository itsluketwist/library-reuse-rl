"""Tests for the radon code metrics (response parsing itself is tested in tests/test_lure)."""

from src.response.metrics import code_metrics


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
