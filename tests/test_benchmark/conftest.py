"""Shared set-up for the benchmark tests: no test may reach the paid model API."""

import generation
import pytest


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch) -> None:
    """Remove the pipeline's API client, so a model call a test forgot to stub fails at once."""
    monkeypatch.setattr(generation, "_client", None)
