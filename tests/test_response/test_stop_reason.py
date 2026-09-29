"""Tests for the shared stop_reason vocabulary (src/response/stop_reason.py)."""

import pytest

from src.response.stop_reason import map_stop_reason, stop_reason_from_tokens


class TestMapStopReason:
    """One mapping shared by vllm and API responses."""

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


class TestStopReasonFromTokens:
    """grpo training gets no finish_reason from trl, so this derives one from tokens."""

    def test_ends_on_eos(self) -> None:
        assert stop_reason_from_tokens([1, 2, 3], eos_token_id=3) == "eos"

    def test_does_not_end_on_eos(self) -> None:
        assert stop_reason_from_tokens([1, 2, 3], eos_token_id=9) == "length"

    def test_missing_completion_ids_is_unknown(self) -> None:
        assert stop_reason_from_tokens(None, eos_token_id=3) == "unknown"
        assert stop_reason_from_tokens([], eos_token_id=3) == "unknown"

    def test_missing_eos_token_id_is_unknown(self) -> None:
        assert stop_reason_from_tokens([1, 2, 3], eos_token_id=None) == "unknown"
