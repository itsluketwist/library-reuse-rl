"""Model/tokenizer loading, shared by training and evaluation."""

# unsloth patches trl's GRPOConfig/GRPOTrainer in place (reassigning them to its own,
# faster versions) the moment it's imported - so it must be imported before trl is
# imported anywhere in the process, or code that reads trl.GRPOConfig/GRPOTrainer (eg.
# src/grpo/config.py's GRPOConfig subclass) binds to the slower, unpatched originals
# instead. see https://github.com/unslothai/unsloth/issues/5183 for what goes wrong
# when the import order is the other way round. the "isort: split" below stops isort
# reordering this above/below the other imports.
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

    `max_seq_length` sizes unsloth's vllm engine's kv cache - it must cover the longest
    training prompt plus the rollout decoding profile's max_new_tokens (see
    config/training.yaml). `gpu_memory_utilization` is the fraction of gpu memory
    that engine is allowed to reserve.

    `use_vllm` is trl's own GRPOConfig.use_vllm setting (see config/training.yaml) -
    reused here as unsloth's `fast_inference` flag, rather than adding a second on/off
    switch for the same thing.

    unsloth's vllm fast_inference does not support full-parameter fine-tuning, so this
    project trains lora adapters on a frozen base model instead: the base is loaded with
    `full_finetuning` off, and FastLanguageModel.get_peft_model(...) attaches the
    adapters before the model is handed to the GRPOTrainer. the lora settings (rank,
    alpha, target modules, ...) come from the recipe as trl ModelConfig fields, see the
    lora_* keys in config/training.yaml. `max_lora_rank` must also be passed to
    from_pretrained so the colocated vllm engine reserves space for adapters of that
    rank during rollouts.

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
    )

    # attach the lora adapters to the frozen base model. gradient checkpointing is set up
    # here too, via unsloth's memory-optimised "unsloth" mode, rather than through the
    # generic model.gradient_checkpointing_enable() used for a plain full-finetuning load
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
        # use_cache and gradient checkpointing don't work together (caching keeps
        # activations around that checkpointing is trying to avoid recomputing) -
        # re-enabled after training in train_model.py, right before saving the model
        model.config.use_cache = False

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    return model, tokenizer
