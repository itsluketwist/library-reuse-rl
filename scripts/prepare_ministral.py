"""Make a local copy of Ministral 3 8B Reasoning that loads like any other hugging face model.

The hub repo ships two formats side by side, and the mistral-format files change how the
model is loaded:

- with `tekken.json` present, transformers and vllm both switch to mistral's own tokenizer,
  which ignores the repo's chat template - and with it the default system prompt that tells
  the model to reason in [THINK]...[/THINK]. our checkpoints are saved without it, so the
  base model and its trained epochs would be prompted differently.
- with `params.json` and `consolidated.safetensors` present, vllm prefers the mistral-format
  config and weights, whose parameter names differ from the hugging face ones that training
  syncs into its rollout engine.

So the copy leaves those three files out. It also stops the tokenizer treating [THINK] and
[/THINK] as special tokens: trl and vllm decode with skip_special_tokens, which would delete
them and leave the reward unable to tell reasoning from answer. They are still single tokens,
so tokenisation is unchanged - only decoding now keeps them.

Loading the copy logs transformers' "incorrect regex pattern" warning. It is a false alarm,
raised because config.json was written by a transformers 5 pre-release: the pattern is already
the corrected one, and the check below confirms tokenisation matches mistral's own tokenizer.

Run once, on a login node (it downloads ~17GB):

    python scripts/prepare_ministral.py

config/models.yaml points the `ministral3-8b` entry at the copy.
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


REPO_ID = "mistralai/Ministral-3-8B-Reasoning-2512"
# pinned, so every run uses the same weights and tokenizer
REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
MISTRAL_FORMAT_FILES = ["tekken.json", "params.json", "consolidated.safetensors"]
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
    Check the copy loads as a hugging face tokenizer, tokenises text as mistral's own tokenizer
    does, renders the reasoning system prompt, tokenises prompts exactly as the original does,
    and keeps the tags when decoding.
    """
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
    print(f"ok - {folder} decodes {decoded!r} (eos {copy.eos_token!r} dropped)")


def main() -> None:
    """Download the hugging face-format files, unmark the reasoning tags, and check the copy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir()

    snapshot_download(
        REPO_ID,
        revision=REVISION,
        local_dir=output_dir,
        ignore_patterns=MISTRAL_FORMAT_FILES,
    )
    unmark_reasoning_tags(output_dir)
    check_copy(output_dir)


if __name__ == "__main__":
    main()
