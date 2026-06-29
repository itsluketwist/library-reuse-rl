"""Model/tokenizer loading, shared by training and evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig, get_kbit_device_map, get_quantization_config


def get_tokenizer(
    model_args: ModelConfig,
    chat_template: str | None = None,
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    return tokenizer


def get_model(
    model_args: ModelConfig,
    gradient_checkpointing: bool = False,
) -> AutoModelForCausalLM:
    """Get the model, resolving dtype and quantization/device settings from `model_args`."""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
