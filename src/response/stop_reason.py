"""Map how a generation ended to this project's shared stop_reason vocabulary."""

import lure


def map_stop_reason(finish_reason: str | None) -> str:
    """
    Map "stop" (natural end-of-turn) to "eos" and pass "length" (hit max_tokens) through.
    Shared by vllm and the OpenAI-compatible APIs, which both report these same two
    names. Anything else passes through verbatim so it stays
    visible rather than inflating the overflow count, and a missing reason is "unknown".

    Returns the mapped stop reason.
    """
    if finish_reason is None:
        return "unknown"
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)


def stop_reason_from_tokens(
    completion_ids: list[int] | None,
    eos_token_id: int | None,
) -> str:
    """
    Derive a stop reason from a completion's tokens rather than a backend finish reason.

    Used by grpo training, which trl gives no finish reason at all - only the completion
    tokens and the tokenizer's eos_token_id. lure works out "stop" or "length" from the
    tokens, which is mapped into map_stop_reason's vocabulary ("unknown" when either input
    is missing).

    Returns the derived stop reason.
    """
    return map_stop_reason(lure.stop_reason_from_tokens(completion_ids, eos_token_id))
