"""Generate responses from a model for evaluation, one prompt at a time."""

from copy import deepcopy
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
    0 (the "greedy" profile) is treated as greedy decoding.

    Generates one prompt at a time rather than batching, to keep this simple - no
    padding/attention-mask handling needed across prompts of different lengths.

    Returns one generated response string per prompt.
    """
    do_sample = decoding_settings["temperature"] > 0

    # start from the model's own generation config so model-specific token ids are
    # preserved, then override only the fields we care about
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = do_sample
    generation_config.max_new_tokens = max_new_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if do_sample:
        generation_config.temperature = decoding_settings["temperature"]
        generation_config.top_p = decoding_settings["top_p"]
        generation_config.top_k = decoding_settings.get("top_k")
    else:
        # reset to neutral values so transformers doesn't warn about sampling
        # params being set while do_sample=False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = None

    responses = []
    for prompt in prompts:
        # return_dict=True also gives the attention_mask, needed since qwen has no
        # dedicated pad token
        model_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        output_ids = model.generate(**model_inputs, generation_config=generation_config)

        new_tokens = output_ids[0, model_inputs["input_ids"].shape[-1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return responses
