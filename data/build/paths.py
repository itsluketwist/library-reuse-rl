"""Shared paths for the dataset build pipelines (see data/README.md).

The build scripts live under data/build/ (some one level further down, in amplify/ and
library/), so every shared path is resolved from this module rather than each script's own
location. Built datasets land under data/train/ and data/eval/, kept separate from the
construction code here.
"""

from pathlib import Path


# data/build/ (this module) and its parent data/
BUILD_DIR = Path(__file__).parent
DATA_DIR = BUILD_DIR.parent

# vendored amplify seed - fetched by build/amplify/vendor_seed.py, gitignored
VENDOR_DIR = BUILD_DIR / "vendor"
PROMPTS_FILE = VENDOR_DIR / "prompts" / "benchmark-prompts.json"
EXTRACTED_FILE = VENDOR_DIR / "results" / "combined-extracted.json"
REPOS_DIR = VENDOR_DIR / "custom-repos"

# committed repos we author as extra prompt contexts for the library pipeline, so every
# taxonomy category has a plausible repo it could be added to (alongside vendored python-api)
LIBRARY_REPOS_DIR = BUILD_DIR / "library" / "repos"

# category taxonomies (committed) - the source of truth for each dataset family, now split
# by pipeline: product categories for amplify, code-level library categories for library.
AMPLIFY_TAXONOMY_FILE = BUILD_DIR / "amplify" / "taxonomy.yaml"
LIBRARY_TAXONOMY_FILE = BUILD_DIR / "library" / "taxonomy.yaml"

# built dataset output dirs (train splits gitignored, eval committed)
TRAIN_DIR = DATA_DIR / "train"
EVAL_DIR = DATA_DIR / "eval"
SKYWORK_DIR = TRAIN_DIR / "skywork"
PRODUCT_CHOICE_DIR = TRAIN_DIR / "product_choice"
LIBRARY_USAGE_DIR = TRAIN_DIR / "library_usage"

# audit.py writes one report per pipeline, alongside that pipeline's other docs
LIBRARY_REPORT_FILE = BUILD_DIR / "library" / "docs" / "REPORT.md"
AMPLIFY_REPORT_FILE = BUILD_DIR / "amplify" / "docs" / "REPORT.md"

# per-category allowlists live in the repo config dir (committed): written by
# build/amplify/allowlists.py, read by the reward at train time. relative to the repo root,
# since the pipeline is run from there (python data/build/...).
ALLOWLISTS_FILE = Path("config") / "library_allowlists.yaml"
