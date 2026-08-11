"""Build the product-choice dataset from the amplify seed (an option, not the main focus).

Each open-ended amplify question, set in every applicable vendored repo's context, becomes a
row whose reward reference is the category's allowlist. This is a deterministic build with no
LLM: real, trainable data straight from the amplify material. We keep it as a relocated option
alongside the library-usage split we actually focus on (see data/build/library/build.py).

The prompts and repos come from amplifying.ai's "claude-code-picks" study (credit to them):
https://amplifying.ai/research/claude-code-picks
https://github.com/amplifying-ai/claude-code-picks

Usage: python data/build/amplify/build.py
"""

import json

import yaml
from llm_cgr import save_jsonl

from data.build.contexts import Context, build_contexts
from data.build.paths import (
    ALLOWLISTS_FILE,
    PRODUCT_CHOICE_DIR,
    PROMPTS_FILE,
    REPOS_DIR,
)
from data.build.split import assign_task_ids, stratified_split


SEED = 42
VAL_FRAC = 0.1


def _flatten_allowlist(
    category: dict,
) -> list[str]:
    """All match terms for a category: canonical tool names plus their aliases.

    The dataset row carries a single flat list (rather than a nested aliases dict) so the
    verifier can match every spelling without the row needing a nested structure.

    Returns the flat list of match terms.
    """
    terms = list(category["allowlist"])
    for alias_list in (category.get("aliases") or {}).values():
        terms.extend(alias_list)
    return terms


def build_product_choice_rows(
    prompts: dict,
    allowlists: dict,
    contexts: dict[str, Context],
) -> list[dict]:
    """Build product-choice rows: each amplify question, in each applicable repo context.

    Returns the product-choice rows.
    """
    categories = allowlists["categories"]
    diy_default = allowlists["diy_terms_default"]

    rows = []
    for category in prompts["categories"]:
        cid = category["id"]
        config = categories.get(cid)
        if config is None:
            continue

        match_terms = _flatten_allowlist(config)
        diy_terms = diy_default + config.get("diy_terms", [])

        for repo in category["applicableTo"]:
            context = contexts.get(repo)
            if context is None:
                continue  # a repo the study used but we don't vendor (e.g. cal.com)

            for prompt in category["prompts"]:
                rows.append(
                    {
                        "prompt": f"{context.summary}\n\n{prompt['text']}",
                        "task_type": "product_choice",
                        "category_id": cid,
                        "language": context.language,
                        "context_id": context.context_id,
                        "allowlist": match_terms,
                        "diy_terms": diy_terms,
                        "allowlist_version": allowlists["version"],
                        "source": "amplify",
                    }
                )
    return rows


def dedupe(
    rows: list[dict],
) -> list[dict]:
    """Drop rows with a duplicate prompt, keeping first occurrence.

    Returns the deduplicated rows.
    """
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["prompt"] not in seen:
            seen.add(row["prompt"])
            unique.append(row)
    return unique


def main() -> None:
    """Build the product-choice dataset from the vendored seed and write its splits."""
    prompts = json.loads(PROMPTS_FILE.read_text())
    allowlists = yaml.safe_load(ALLOWLISTS_FILE.read_text())
    contexts = build_contexts(REPOS_DIR)
    print(f"built {len(contexts)} repo contexts: {sorted(contexts)}")

    rows = dedupe(build_product_choice_rows(prompts, allowlists, contexts))
    train, val, test = stratified_split(
        rows,
        key=lambda row: row["category_id"],
        val_frac=VAL_FRAC,
        seed=SEED,
        floor_holdout=True,
    )
    # label rows within the splits (test -> train -> val); the "P" tags the product-choice
    # dataset, keeping ids distinct from the library-usage split's "U" ids
    train, val, test = assign_task_ids(train, val, test, prefix="LRRL/P")

    PRODUCT_CHOICE_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        save_jsonl(split_rows, str(PRODUCT_CHOICE_DIR / f"{split_name}.jsonl"))
    print(
        f"product_choice: {len(rows)} rows -> train {len(train)}, "
        f"val {len(val)}, test {len(test)} in {PRODUCT_CHOICE_DIR}/"
    )


if __name__ == "__main__":
    main()
