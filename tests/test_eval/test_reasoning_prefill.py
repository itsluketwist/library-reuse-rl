"""Tests for restoring a chat template's prefilled opening reasoning tag."""

import pytest

from src.eval.reasoning_prefill import prefilled_reasoning_tag


@pytest.mark.parametrize(
    ("prompt_end", "tag"),
    [
        # qwen3.5 and domyn-small end the prompt inside the reasoning
        ("<|im_start|>assistant\n<think>\n", "<think>\n"),
        ("<extra_id_1>Assistant\n<think>\n", "<think>\n"),
        # qwen3 lets the model write <think> itself
        ("<|im_start|>assistant\n", ""),
        # thinking switched off: the reasoning is already closed
        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", ""),
    ],
)
def test_prefilled_reasoning_tag(prompt_end: str, tag: str):
    """Only a prompt ending in an open reasoning tag has one to restore."""
    assert prefilled_reasoning_tag(prompt_end) == tag
