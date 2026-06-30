"""Generate responses from a model for evaluation, one prompt at a time."""

from typing import Any

from transformers import AutoModelForCausalLM, PreTrainedTokenizer


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """
    Generate one response per prompt from `model`.

    `prompts` is a list of chat-messages lists, as produced by
    src/data.py::load_eval_dataset. `decoding_settings` is a decoding profile dict
    (see config/decoding.yaml) with "temperature"/"top_p"/"top_k" - a temperature of
    0 (the "greedy" profile) is treated as greedy decoding, so sampling kwargs are
    only passed to `generate` when actually sampling, to avoid transformers warning
    about settings that have no effect.

    Generates one prompt at a time rather than batching, to keep this simple - no
    padding/attention-mask handling needed across prompts of different lengths.

    Returns one generated response string per prompt.
    """
    do_sample = decoding_settings["temperature"] > 0

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = decoding_settings["temperature"]
        generation_kwargs["top_p"] = decoding_settings["top_p"]
        if decoding_settings.get("top_k") is not None:
            generation_kwargs["top_k"] = decoding_settings["top_k"]

    responses = []
    for prompt in prompts:
        # return_dict=True also gives us the attention_mask - needed since Qwen has no
        # dedicated pad token (pad_token_id == eos_token_id), so transformers can't
        # safely infer it from input_ids alone and would otherwise warn/misbehave
        model_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        output_ids = model.generate(**model_inputs, **generation_kwargs)

        new_tokens = output_ids[0, model_inputs["input_ids"].shape[-1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return responses
