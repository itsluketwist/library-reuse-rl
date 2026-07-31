"""Model/tokenizer loading, shared by training and evaluation."""

# unsloth patches trl's GRPOConfig/GRPOTrainer in place when imported, so it must be
# imported before trl anywhere in the process, or subclasses (e.g. src/grpo/config.py)
# bind to the slower, unpatched originals. see unslothai/unsloth#5183. the "isort:
# split" below stops isort reordering this among the other imports.
import unsloth  # noqa: F401


# isort: split

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import ModelConfig
from unsloth import FastLanguageModel


def get_model_and_tokenizer(
    model_args: ModelConfig,
    max_seq_length: int,
    gpu_memory_utilization: float,
    use_vllm: bool,
    gradient_checkpointing: bool = False,
    chat_template: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Get the model and tokenizer via unsloth's FastLanguageModel, resolving dtype from
    `model_args`.

    `max_seq_length` sizes the vllm engine's kv cache (must cover the longest prompt
    plus the rollout profile's max_new_tokens); `gpu_memory_utilization` is the fraction
    of gpu memory that engine may reserve. `use_vllm` is trl's GRPOConfig.use_vllm,
    reused here as unsloth's `fast_inference` flag.

    fast_inference doesn't support full fine-tuning, so this project trains lora adapters
    on a frozen base: get_peft_model attaches them (rank/alpha/target modules from the
    recipe's lora_* keys) before the model reaches the GRPOTrainer, and from_pretrained
    is passed max_lora_rank so vllm reserves adapter space for rollouts.

    Returns the loaded model and its tokenizer, with `chat_template` applied if given.
    """
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        max_seq_length=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        fast_inference=use_vllm,
        max_lora_rank=model_args.lora_r,
        load_in_4bit=False,
    )

    # attach lora adapters to the frozen base; gradient checkpointing uses unsloth's
    # memory-optimised "unsloth" mode rather than the generic enable() call
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
    )

    if gradient_checkpointing:
        # use_cache is incompatible with gradient checkpointing; re-enabled in
        # train_model.py before saving the model
        model.config.use_cache = False

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    return model, tokenizer
