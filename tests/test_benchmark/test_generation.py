"""Tests for the shared model client: a stream that drops part way through is sent again."""

import contextlib
from collections.abc import Iterator
from typing import Any

import generation
import httpx
import pytest


class _FakeClient:
    """Stands in for the API client: its stream drops `drops` times, then gives a message."""

    def __init__(
        self,
        drops: int,
    ) -> None:
        self.drops = drops
        self.calls = 0
        # the real client's streams live at client.messages.stream
        self.messages = self

    @contextlib.contextmanager
    def stream(
        self,
        **kwargs: Any,
    ) -> Iterator["_FakeClient"]:
        self.calls += 1
        yield self

    def get_final_message(self) -> str:
        if self.calls <= self.drops:
            raise httpx.RemoteProtocolError("peer closed connection")
        return "message"


def test_a_dropped_stream_is_sent_again(monkeypatch) -> None:
    client = _FakeClient(drops=generation.STREAM_ATTEMPTS - 1)
    monkeypatch.setattr(generation, "_client", client)
    assert generation._stream_message({}) == "message"
    assert client.calls == generation.STREAM_ATTEMPTS


def test_a_stream_that_keeps_dropping_raises(monkeypatch) -> None:
    client = _FakeClient(drops=generation.STREAM_ATTEMPTS)
    monkeypatch.setattr(generation, "_client", client)
    with pytest.raises(httpx.RemoteProtocolError):
        generation._stream_message({})
    assert client.calls == generation.STREAM_ATTEMPTS
