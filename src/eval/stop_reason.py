"""Map a generation backend's finish reason to this project's stop_reason vocabulary.

Eval only: grpo training answers the same "was this cut off?" question from tokens
instead, since trl gives reward functions no finish reason (see LibraryReward._is_truncated).
"""


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
