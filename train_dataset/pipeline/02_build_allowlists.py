"""Stage 02: build config/library_allowlists.yaml from the taxonomy and amplify extractions.

For the product categories, the allowlist is seeded from the amplify tool-frequency data
(train_dataset/vendor/results/combined-extracted.json) - every tool models actually
recommended, minus the "Custom/DIY" marker. For the code-level categories, which the amplify
study does not cover, the allowlist is the taxonomy's seed_libraries. A small curated aliases
map and a shared diy_terms list are attached so the deterministic verifier
(src/response/pick.py) can match abbreviations ("postgres" -> PostgreSQL) and spot DIY answers.

This is the deterministic seed. An offline LLM judge can later enrich/normalise the result
(see train_dataset/PLAN.md) - it is intentionally kept out of the training loop.

Usage: python train_dataset/pipeline/02_build_allowlists.py
"""

import json

import yaml

from src.utils.config import load_full_yaml
from train_dataset.common import ALLOWLISTS_FILE as OUTPUT_FILE
from train_dataset.common import EXTRACTED_FILE, PROMPTS_FILE, TAXONOMY_FILE


# allowlist version, recorded on every dataset row so a reward stays reproducible
VERSION = "2026-08"

# the amplify marker for a roll-your-own recommendation - excluded from every allowlist
DIY_MARKER = "Custom/DIY"

# prose phrases that signal a roll-your-own answer, shared across categories
SHARED_DIY_TERMS = [
    "custom implementation",
    "roll your own",
    "hand-rolled",
    "from scratch",
    "build your own",
    "write your own",
    "in-memory dict",
    "in-memory store",
    "json file",
    "flat file",
    "plain text file",
    "diy",
]

# curated abbreviations/synonyms, keyed by the lowercased canonical tool name. only tools
# whose common shorthand wouldn't match the canonical name literally need an entry here.
ALIASES: dict[str, list[str]] = {
    "postgresql": ["postgres", "pg"],
    "mongodb": ["mongo"],
    "github actions": ["gh actions"],
    "amazon s3": ["s3", "aws s3"],
    "tailwind css": ["tailwind"],
    "beautifulsoup4": ["beautifulsoup", "bs4"],
    "pyjwt": ["jwt"],
    "pymupdf": ["fitz"],
    "pyyaml": ["yaml"],
    "python-dateutil": ["dateutil"],
    "thefuzz": ["fuzzywuzzy"],
}


def _category_names() -> dict[str, str]:
    """Map each product category id to its display name, from the amplify prompts file.

    Returns the {category_id: display_name} mapping.
    """
    prompts = json.loads(PROMPTS_FILE.read_text())
    return {c["id"]: c["name"] for c in prompts["categories"]}


def _amplify_tools(
    extracted: dict,
    display_name: str,
) -> list[str]:
    """The external tools amplify saw for a category, most-recommended first, DIY dropped.

    Returns the tool names.
    """
    category = extracted.get("byCategory", {}).get(display_name, {})
    tools = sorted(category.get("tools", []), key=lambda t: -t["count"])
    return [t["name"] for t in tools if t["name"] != DIY_MARKER]


def _aliases_for(
    allowlist: list[str],
) -> dict[str, list[str]]:
    """The subset of the curated aliases map that applies to these allowlist tools.

    Returns a {canonical: aliases} mapping (only for tools that have aliases).
    """
    return {
        tool: ALIASES[tool.lower()] for tool in allowlist if tool.lower() in ALIASES
    }


def build_allowlists() -> dict:
    """Assemble the per-category allowlist config from the taxonomy and amplify data.

    Returns the full config dict ready to serialise.
    """
    taxonomy = load_full_yaml(str(TAXONOMY_FILE))
    extracted = json.loads(EXTRACTED_FILE.read_text())
    names = _category_names()

    categories: dict[str, dict] = {}

    # product categories: allowlist seeded from the amplify tool frequencies
    for entry in taxonomy.get("product", []):
        cid = entry["id"]
        allowlist = _amplify_tools(extracted, names.get(cid, entry["name"]))
        # fold in any taxonomy seed_libraries not already present (usually none)
        for lib in entry.get("seed_libraries", []):
            if lib not in allowlist:
                allowlist.append(lib)
        categories[cid] = {
            "name": entry["name"],
            "family": "product",
            "allowlist": allowlist,
            "aliases": _aliases_for(allowlist),
            "diy_terms": [],
        }

    # code-level categories: allowlist is the taxonomy's seed_libraries (import names)
    for entry in taxonomy.get("code", []):
        allowlist = list(entry.get("seed_libraries", []))
        categories[entry["id"]] = {
            "name": entry["name"],
            "family": "code",
            "allowlist": allowlist,
            "aliases": _aliases_for(allowlist),
            "diy_terms": [],
        }

    return {
        "version": VERSION,
        "diy_terms_default": SHARED_DIY_TERMS,
        "categories": categories,
    }


def main() -> None:
    """Build the allowlist config and write it to config/library_allowlists.yaml."""
    config = build_allowlists()

    header = (
        "# per-category library allowlists for the library-reuse datasets.\n"
        "# built by train_dataset/pipeline/02_build_allowlists.py from taxonomy.yaml and\n"
        "# the amplify\n"
        "# tool-frequency extractions - re-run that script to regenerate. 'allowlist' lists\n"
        "# the appropriate external tools/libraries for a category; 'diy_terms' (plus the\n"
        "# shared diy_terms_default) flag roll-your-own answers.\n\n"
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )

    n_product = sum(c["family"] == "product" for c in config["categories"].values())
    n_code = sum(c["family"] == "code" for c in config["categories"].values())
    print(f"wrote {OUTPUT_FILE} - {n_product} product + {n_code} code categories")


if __name__ == "__main__":
    main()
