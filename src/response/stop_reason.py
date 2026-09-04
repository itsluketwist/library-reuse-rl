"""Map how a generation ended to this project's shared stop_reason vocabulary."""


def map_stop_reason(finish_reason: str | None) -> str:
    """
    Map "stop" (natural end-of-turn) to "eos" and pass "length" (hit max_tokens) through.
    Shared by vllm, the OpenAI-compatible APIs, and externally generated responses, which
    all report these same two names. Anything else passes through verbatim so it stays
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
    tokens and the tokenizer's eos_token_id. Shares map_stop_reason's vocabulary: "unknown"
    when either input is missing, "eos" when the completion ends on eos_token_id, "length"
    otherwise.

    Returns the derived stop reason.
    """
    if eos_token_id is None or not completion_ids:
        return "unknown"
    return "eos" if completion_ids[-1] == eos_token_id else "length"
