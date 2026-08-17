"""Map a generation backend's finish reason to this project's stop_reason vocabulary.

Shared by vllm generation (generate.py) and API generation (generate_api.py), since vllm
and OpenAI-compatible chat completion APIs both use "stop"/"length" for the cases this
project distinguishes.
"""


def map_stop_reason(finish_reason: str) -> str:
    """
    Map "stop" (natural end-of-turn) to "eos", pass "length" (hit max_tokens) through
    unchanged, and pass anything else through verbatim so it stays visible rather than
    inflating the overflow count.

    Returns the mapped stop reason.
    """
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)
