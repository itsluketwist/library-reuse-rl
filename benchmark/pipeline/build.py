"""Shared, non-LLM helpers for the BigCodeBenchReuse build pipeline.

Holds the stage paths, the BigCodeBench library analysis, and the hard-exclude buckets that the
numbered build scripts use. Reusable library logic (import classification, the import->pip map,
the prompt library scan) lives in the released `bcbreuse` package and is imported from there, so
there is one source of truth shared by the build and the harness.
"""

import os
import sys
from pathlib import Path

from bcbreuse.libraries import analyse_source

from src.utils.io import load_jsonl


# interpreter used to execute model-generated code during the build (stage 04 runs the stdlib gold
# solutions). defaults to the current interpreter; point BCB_BUILD_PYTHON at an isolated venv
# (benchmark/pipeline/setup_build_venv.sh) so untrusted generated code runs in a
# throwaway environment.
BUILD_PYTHON = os.environ.get("BCB_BUILD_PYTHON") or sys.executable


# --- stage paths ----------------------------------------------------------------------

_BENCHMARK = Path(__file__).parent.parent  # the benchmark/ directory
VENDOR_DIR = _BENCHMARK / "vendor"  # fetched BigCodeBench seed (gitignored)
ARTEFACTS_DIR = _BENCHMARK / "artefacts"  # per-stage outputs (gitignored)
# built splits go straight into the package, where they are bundled and committed as the benchmark
DATA_DIR = _BENCHMARK / "bcbreuse" / "data"

# the BigCodeBench version the whole build is pinned to
BCB_VERSION = "v0.1.4"


def artefact_path(
    name: str,
) -> Path:
    """Path to a stage artefact (e.g. "04_tests.jsonl") under the artefacts directory.

    Returns the full path.
    """
    return ARTEFACTS_DIR / name


def completed_ids(
    path: Path,
    key: str = "seed_id",
) -> set[str]:
    """Ids already present in an output artefact, so a resumed stage can skip them.

    Returns the set of completed ids (empty if the artefact does not exist yet).
    """
    if not path.exists():
        return set()
    return {row[key] for row in load_jsonl(path)}


def latest_records(
    path: Path,
    key: str = "seed_id",
) -> dict[str, dict]:
    """The most recent record per id in a JSONL artefact (later attempts override earlier ones).

    Reruns append a fresh record for a retried id, so the last one written is its current state.

    Returns the {id: latest record} map (empty if the artefact does not exist yet).
    """
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    for row in load_jsonl(path):
        records[row[key]] = row
    return records


def succeeded_ids(
    path: Path,
    key: str = "seed_id",
) -> set[str]:
    """Ids whose latest record succeeded (kept=True), so a rerun retries only the ones that didn't.

    Returns the set of succeeded ids (empty if the artefact does not exist yet).
    """
    return {k for k, row in latest_records(path, key).items() if row.get("kept")}


# --- library analysis of BCB solutions ------------------------------------------------


# in BigCodeBench the imports and def live in `complete_prompt`, and `canonical_solution`
# is only the function body - so we join them to analyse the whole program at once
def full_solution(
    complete_prompt: str,
    canonical_solution: str,
) -> str:
    """Reconstruct a BCB problem's full reference program (imports + signature + body).

    Returns the joined source.
    """
    return complete_prompt + canonical_solution


def analyse_libraries(
    complete_prompt: str,
    canonical_solution: str,
) -> tuple[list[str], list[str]]:
    """Classify the libraries a BCB reference solution imports (via the shared analyser).

    Returns a (std_libs, ext_libs) tuple.
    """
    return analyse_source(full_solution(complete_prompt, canonical_solution))


# --- hard-exclude buckets (stage 02) --------------------------------------------------

# libraries that make real network calls - a sandboxed, deterministic eval cannot run them
_NETWORK_LIBS = {
    "requests",
    "urllib3",
    "httpx",
    "aiohttp",
    "flask",
    "django",
    "smtplib",
    "ftplib",
    "socket",
    "selenium",
    "mechanize",
    "yfinance",
    "wikipedia",
}

# libraries whose only output is a rendered figure / GUI - nothing to serialise to stdout
_PLOT_GUI_LIBS = {
    "matplotlib",
    "seaborn",
    "turtle",
    "tkinter",
    "wordcloud",
    "folium",
}


def hard_exclude_reason(
    ext_libs: list[str],
) -> str | None:
    """Reason a problem cannot possibly run in our sandbox, or None if it might.

    Deliberately narrow: only genuinely-impossible cases (live network, plot/GUI-only output) are
    excluded here; everything else is left for the LLM judge and the executable gates to decide.

    Returns a short reason string, or None to keep the problem as a candidate.
    """
    libs = set(ext_libs)
    if libs & _NETWORK_LIBS:
        return "network"
    if libs & _PLOT_GUI_LIBS:
        return "plot-gui"
    return None
