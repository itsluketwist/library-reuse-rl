"""Generate responses from a model for evaluation, batching prompts together."""

from copy import deepcopy
from typing import Any

from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


# number of prompts generated together in one model.generate() call. batching keeps
# the gpu busy (batch size 1 leaves most of an h200 idle) - tune this down if a
# bigger model (e.g. qwen3-8b) runs out of gpu memory at max_new_tokens=8192
EVAL_BATCH_SIZE = 8


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[dict]],
    decoding_settings: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Generate one response per prompt from `model`, EVAL_BATCH_SIZE prompts at a time.

    `prompts` is a list of chat-messages lists, as produced by
    src/data.py::load_eval_dataset. `decoding_settings` is a decoding profile dict (see
    eval_decoding_profiles in config/train/demo.yaml) with
    "temperature"/"top_p"/"top_k"/"max_new_tokens" -
    a temperature of 0 (the "greedy" profile) is treated as greedy decoding.

    Prompts within a batch have different lengths, so they're left-padded (padding on
    the left keeps every sequence's most recent token last, which is required for
    causal-lm generation to continue correctly). Shows a tqdm progress bar over
    prompts, since generation is the slow part of evaluation and can otherwise run
    for hours with no feedback.

    Returns a list of (response, stop_reason) tuples, where stop_reason is "eos" if
    the model stopped naturally or "length" if it hit the token cap.
    """
    do_sample = decoding_settings["temperature"] > 0

    # start from the model's own generation config so model-specific token ids are
    # preserved, then override only the fields we care about
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = do_sample
    generation_config.max_new_tokens = decoding_settings["max_new_tokens"]
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

    # eos token id(s) used below to work out each row's real stop reason - can be a
    # single id or a list of ids depending on the model (e.g. qwen has more than one
    # valid end-of-turn token)
    eos_token_ids = generation_config.eos_token_id or []
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids)

    # batched generation needs left-padding (so every sequence ends at the same
    # position) and a real pad token to pad with - qwen models have no dedicated pad
    # token by default, so fall back to the eos token, matching pad_token_id above
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = []
    with tqdm(
        total=len(prompts),
        desc="generating",
        unit="sample",
    ) as progress:
        for start in range(0, len(prompts), EVAL_BATCH_SIZE):
            batch = prompts[start : start + EVAL_BATCH_SIZE]

            # return_dict=True also gives the attention_mask, needed since padded
            # prompts within a batch have different real lengths
            model_inputs = tokenizer.apply_chat_template(
                batch,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            ).to(model.device)

            output_ids = model.generate(
                **model_inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )

            input_length = model_inputs["input_ids"].shape[-1]
            for row_ids in output_ids:
                new_tokens = row_ids[input_length:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # can't tell "eos" from "length" by comparing new_tokens' length to
                # max_new_tokens any more, since generate() right-pads every row in
                # the batch out to the longest row - instead check whether this row
                # actually produced an eos token anywhere in its generated tokens
                stop_reason = (
                    "eos"
                    if any(token_id.item() in eos_token_ids for token_id in new_tokens)
                    else "length"
                )
                responses.append((response, stop_reason))

            progress.update(len(batch))

    return responses
