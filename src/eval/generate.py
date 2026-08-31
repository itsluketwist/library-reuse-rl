"""Generate responses from a model for evaluation, using vLLM for batched offline inference."""

from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _stop_reason(finish_reason: str) -> str:
    """
    Map vllm's finish_reason to this project's stop_reason vocabulary: "stop" (natural
    end-of-turn) becomes "eos", "length" (hit max_tokens) passes through. Anything else
    passes through verbatim so it stays visible rather than inflating the overflow count.

    Returns the mapped stop reason.
    """
    return {"stop": "eos", "length": "length"}.get(finish_reason, finish_reason)


def generate_responses(
    llm: LLM,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    lora_request: LoRARequest | None = None,
) -> list[list[tuple[str, str]]]:
    """
    Generate one or more sampled responses per prompt from `llm`.

    `prompts` is a list of chat-messages lists (see load_eval_dataset).
    `decoding_settings` is a decoding profile dict (see eval_decoding_profiles in
    config/training.yaml); temperature 0 is vllm's greedy-decoding convention.
    `decoding_settings["n_samples"]` (default 1) sets how many completions vllm samples
    per prompt - e.g. for LibReuseBench's pass@k or library-usage's per-response reuse
    statistics (see src/eval/scoring.py's MULTI_SAMPLE_EVALUATORS).
    `lora_request`, when given, applies a trained adapter on top of the base model.

    vllm handles chat templating, batching, and scheduling internally, and returns
    outputs in the same order as `prompts`.

    Returns a list aligned with `prompts`, each element a list of that prompt's
    (response, stop_reason) samples (length `n_samples`).
    """
    sampling_params = SamplingParams(
        temperature=decoding_settings["temperature"],
        top_p=decoding_settings["top_p"],
        top_k=decoding_settings.get("top_k", 0),
        max_tokens=decoding_settings["max_new_tokens"],
        n=decoding_settings.get("n_samples", 1),
    )

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    return [
        [(sample.text, _stop_reason(sample.finish_reason)) for sample in output.outputs]
        for output in outputs
    ]
