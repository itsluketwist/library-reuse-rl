"""Shared paths for the training-dataset build pipeline (see train_dataset/PLAN.md).

The pipeline stages live one level down in train_dataset/pipeline/, so every path is
resolved from this module (the package root) rather than each script's own location.
"""

from pathlib import Path


# package root (train_dataset/)
ROOT = Path(__file__).parent

# vendored amplify seed - fetched by pipeline/01_vendor_seed.py, gitignored
VENDOR_DIR = ROOT / "vendor"
PROMPTS_FILE = VENDOR_DIR / "prompts" / "benchmark-prompts.json"
EXTRACTED_FILE = VENDOR_DIR / "results" / "combined-extracted.json"
REPOS_DIR = VENDOR_DIR / "custom-repos"

# category taxonomy (committed) - the source of truth for both dataset families
TAXONOMY_FILE = ROOT / "taxonomy.yaml"

# built dataset splits (gitignored) and the audit report (committed)
DATA_DIR = ROOT / "data"
REPORT_FILE = ROOT / "REPORT.md"

# per-category allowlists live in the repo config dir (committed): written by
# pipeline/02_build_allowlists.py, read by the reward at train time. relative to the
# repo root, since the pipeline is run from there (PYTHONPATH=. python train_dataset/...)
ALLOWLISTS_FILE = Path("config") / "library_allowlists.yaml"
