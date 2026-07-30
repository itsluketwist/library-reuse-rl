"""Generate responses from a model for evaluation, using vLLM for batched offline inference."""

from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _stop_reason(finish_reason: str) -> str:
    """
    Map vllm's finish_reason ("stop"/"length"/"abort") to this project's stop_reason
    vocabulary. "stop" (natural end-of-turn) becomes "eos"; "length" (hit max_tokens)
    passes through unchanged. Anything else (e.g. "abort" - shouldn't occur in normal
    single-request offline generation) passes through verbatim rather than being
    folded into "length", so it stays visible in saved results instead of silently
    inflating the overflow count.

    Returns the mapped stop reason.
    """
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)


def generate_responses(
    llm: LLM,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    lora_request: LoRARequest | None = None,
) -> list[tuple[str, str]]:
    """
    Generate one response per prompt from `llm`.

    `prompts` is a list of chat-messages lists, as produced by
    src/data.py::load_eval_dataset. `decoding_settings` is a decoding profile dict (see
    eval_decoding_profiles in config/training.yaml) with
    "temperature"/"top_p"/"top_k"/"max_new_tokens" - a temperature of 0 (the "greedy"
    profile) is vllm's own greedy-decoding convention, so no separate handling is
    needed for it.

    `lora_request`, when given, applies a trained lora adapter on top of the base model
    for this generation (see src/eval/evaluate.py) - None runs the base model alone.

    vllm handles chat templating, batching, and scheduling internally (unlike plain
    transformers generate(), which needed manual left-padding and batching here) and
    shows its own progress bar. Returns outputs in the same order as `prompts`.

    Returns a list of (response, stop_reason) tuples, where stop_reason is "eos" if
    the model stopped naturally or "length" if it hit the token cap.
    """
    sampling_params = SamplingParams(
        temperature=decoding_settings["temperature"],
        top_p=decoding_settings["top_p"],
        top_k=decoding_settings.get("top_k", 0),
        max_tokens=decoding_settings["max_new_tokens"],
    )

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    return [
        (output.outputs[0].text, _stop_reason(output.outputs[0].finish_reason))
        for output in outputs
    ]
