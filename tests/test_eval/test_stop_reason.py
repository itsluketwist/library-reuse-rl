"""Tests for the shared stop_reason vocabulary (src/eval/stop_reason.py)."""

import pytest

from src.eval.stop_reason import map_stop_reason


class TestMapStopReason:
    """One mapping shared by vllm, API, and externally generated responses."""

    @pytest.mark.parametrize(
        ("finish_reason", "expected"),
        [
            ("stop", "eos"),  # natural end-of-turn
            ("length", "length"),  # hit max_tokens - counted as overflow
            (None, "unknown"),  # external generations may not record one
            ("content_filter", "content_filter"),  # passed through, stays visible
        ],
    )
    def test_maps_every_case(self, finish_reason, expected) -> None:
        assert map_stop_reason(finish_reason) == expected

    def test_unknown_reasons_do_not_inflate_the_overflow_count(self) -> None:
        """Only a real "length" must map to "length" - scoring counts those as overflow
        (see n_overflow in src/eval/scoring.py), so a passthrough must never collide."""
        assert map_stop_reason("tool_calls") != "length"
        assert map_stop_reason(None) != "length"
