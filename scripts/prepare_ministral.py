"""Make a local, text-only copy of Ministral 3 8B Reasoning that loads like any other llm.

The hub repo needs three changes before our pipeline can use it:

- it is a vision-language model, and vllm 0.28's code for its vision encoder (pixtral) no
  longer imports with transformers 5.17. we never pass images, so the copy keeps only the
  language model: its weights (renamed from "language_model.model.*" to "model.*") and its
  text config, which both libraries load as a plain mistral-style causal lm. the outputs for
  text prompts are unchanged, and training no longer carries an unused vision encoder.
- it ships mistral's own format beside the hugging face one. with `tekken.json` present,
  transformers and vllm both switch to mistral's tokenizer, which ignores the repo's chat
  template - and with it the default system prompt that tells the model to reason in
  [THINK]...[/THINK]. the copy leaves the mistral-format files out.
- trl and vllm decode with skip_special_tokens, which would delete [THINK] and [/THINK] and
  leave the reward unable to tell reasoning from answer. the copy stops treating them as
  special tokens. they are still single tokens, so tokenisation is unchanged - only decoding
  now keeps them.

vllm reads the model's extra attention scaling for long contexts (`llama_4_scaling_beta`) only
from mistral's format, not from the hugging face config. it starts at position 16384, past
training's context limit, so it changes nothing for training.

Loading the copy's tokenizer logs transformers' "incorrect regex pattern" warning. It is a
false alarm, raised because the original config was written by a transformers 5 pre-release:
the pattern is already the corrected one, and the check below confirms tokenisation matches
mistral's own tokenizer.

Run once, on a login node (it downloads ~17GB to the hugging face cache, and writes ~16GB):

    python scripts/prepare_ministral.py

config/models.yaml points the `ministral3-8b` entry at the copy.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ID = "mistralai/Ministral-3-8B-Reasoning-2512"
# pinned, so every run uses the same weights and tokenizer
REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
MISTRAL_FORMAT_FILES = ["tekken.json", "params.json", "consolidated.safetensors"]
# rewritten for the text-only model, rather than copied
REWRITTEN_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "processor_config.json",
]
LANGUAGE_MODEL_PREFIX = "language_model."
REASONING_TAGS = ("[THINK]", "[/THINK]")


def default_output_dir() -> Path:
    """
    The copy's default home: project storage, next to the training checkpoints.

    Returns the directory path.
    """
    project_dir = os.environ.get("PROJECTDIR")
    if project_dir is None:
        raise SystemExit(
            "PROJECTDIR is not set - are you on isambard? pass --output-dir"
        )
    return (
        Path(project_dir)
        / os.environ["USER"]
        / "models"
        / "Ministral-3-8B-Reasoning-2512-hf"
    )


def write_text_only_model(
    source: Path,
    folder: Path,
) -> None:
    """
    Write the language model alone: its config, its weights under their causal-lm names, and
    every other file (tokenizer, chat template, generation config) as it is.
    """
    folder.mkdir(parents=True)
    for path in source.iterdir():
        if path.name not in REWRITTEN_FILES and path.suffix != ".safetensors":
            # follows the cache's symlinks, so the tokenizer edits never touch the cache
            shutil.copy(path, folder / path.name)

    config = json.loads((source / "config.json").read_text())
    text_config = config["text_config"]
    text_config["architectures"] = ["Ministral3ForCausalLM"]
    text_config["dtype"] = config["dtype"]
    text_config["tie_word_embeddings"] = False
    (folder / "config.json").write_text(json.dumps(text_config, indent=2))

    # each shard keeps its name, holding just its language-model tensors
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = {}
    total_size = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(source / shard, framework="pt") as weights:
            for name in weights.keys():
                if name.startswith(LANGUAGE_MODEL_PREFIX):
                    tensors[name.removeprefix(LANGUAGE_MODEL_PREFIX)] = (
                        weights.get_tensor(name)
                    )
        save_file(tensors, folder / shard, metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = shard
            total_size += tensor.nbytes
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def unmark_reasoning_tags(
    folder: Path,
) -> None:
    """
    Mark [THINK] and [/THINK] as ordinary (non-special) tokens in every tokenizer file.

    tokenizer.json holds the flag decoding reads. transformers also re-marks as special every
    token listed in tokenizer_config.json and special_tokens_map.json when it loads the
    tokenizer, so the tags are taken out of those lists too.
    """
    path = folder / "tokenizer.json"
    tokenizer = json.loads(path.read_text())
    for token in tokenizer["added_tokens"]:
        if token["content"] in REASONING_TAGS:
            token["special"] = False
    path.write_text(json.dumps(tokenizer, ensure_ascii=False))

    path = folder / "tokenizer_config.json"
    config = json.loads(path.read_text())
    for token in config["added_tokens_decoder"].values():
        if token["content"] in REASONING_TAGS:
            token["special"] = False
    config["additional_special_tokens"] = [
        token
        for token in config["additional_special_tokens"]
        if token not in REASONING_TAGS
    ]
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2))

    path = folder / "special_tokens_map.json"
    special = json.loads(path.read_text())
    special["additional_special_tokens"] = [
        token
        for token in special["additional_special_tokens"]
        if token["content"] not in REASONING_TAGS
    ]
    path.write_text(json.dumps(special, ensure_ascii=False, indent=2))


def check_copy(
    folder: Path,
) -> None:
    """
    Check the weights are exactly those a text-only model expects, and the tokenizer loads as
    a hugging face tokenizer, tokenises text as mistral's own tokenizer does, renders the
    reasoning system prompt, tokenises prompts exactly as the original does, and keeps the
    tags when decoding.
    """
    # an empty (meta-device) model lists the weight names it will load
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder))
    assert type(model).__name__ == "Ministral3ForCausalLM", type(model).__name__
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(model.state_dict())

    copy = AutoTokenizer.from_pretrained(folder)
    original = AutoTokenizer.from_pretrained(
        REPO_ID, revision=REVISION, mistral_format=False
    )
    mistral = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    assert type(copy).__name__ != "MistralCommonBackend", type(copy).__name__
    assert type(mistral).__name__ == "MistralCommonBackend", type(mistral).__name__

    code = 'import numpy as np\nx = np.arange(1024)\r\n\tprint(f"{x.mean()!r}")  # Ünïcödé 東京\n'
    assert copy.encode(code, add_special_tokens=False) == mistral.encode(
        code,
        add_special_tokens=False,
    )

    messages = [{"role": "user", "content": "Write a function that adds two numbers."}]
    text = copy.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert text.startswith("<s>[SYSTEM_PROMPT]# HOW YOU SHOULD THINK"), text[:80]
    assert text.endswith("[/INST]"), text[-80:]
    copy_ids = copy.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    original_ids = original.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert list(copy_ids["input_ids"]) == list(original_ids["input_ids"])

    think, end_think = copy.convert_tokens_to_ids(list(REASONING_TAGS))
    assert copy.encode("[THINK]", add_special_tokens=False) == [think]
    completion = (
        [think]
        + copy.encode("plan", add_special_tokens=False)
        + [end_think]
        + copy.encode("answer", add_special_tokens=False)
        + [copy.eos_token_id]
    )
    decoded = copy.decode(completion, skip_special_tokens=True)
    assert decoded == "[THINK]plan[/THINK]answer", repr(decoded)
    print(
        f"ok - {folder} holds the {len(index['weight_map'])} language-model weights, "
        f"and decodes {decoded!r} (eos {copy.eos_token!r} dropped)"
    )


def main() -> None:
    """Download the hugging face-format files, write the text-only copy, and check it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir()

    source = snapshot_download(
        REPO_ID,
        revision=REVISION,
        ignore_patterns=MISTRAL_FORMAT_FILES,
    )
    write_text_only_model(Path(source), output_dir)
    unmark_reasoning_tags(output_dir)
    check_copy(output_dir)


if __name__ == "__main__":
    main()
