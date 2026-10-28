"""Shared, non-LLM helpers for the LibReuseBench build pipeline.

Holds the stage paths, the resume helpers for the append-only stage artefacts, and the reuse-value
measure that the numbered build scripts share. Reusable library logic (import classification, the
import->pip map, the prompt library scan, running and scoring function tasks) lives in the released
`libreusebench` package and is imported from there, so the build and the harness agree.
"""

import hashlib
import os
from pathlib import Path

from libreusebench.evaluate import reuse_tier
from libreusebench.libraries import analyse_source

from src.response.metrics import code_metrics
from src.utils.io import load_jsonl


# --- stage paths ----------------------------------------------------------------------

_BENCHMARK = Path(__file__).parent.parent  # the benchmark/ directory
ARTEFACTS_DIR = _BENCHMARK / "artefacts"  # per-stage outputs (tracked in git)
# provisional splits, built from the tasks the agent review kept, before any human review
PROVISIONAL_DIR = ARTEFACTS_DIR / "provisional"
# every task of a pilot run, written out for evaluating before the full run (05_build.py --pilot)
PILOT_DIR = ARTEFACTS_DIR / "pilot"
# final splits go straight into the package, where they are bundled and committed as the benchmark
DATA_DIR = _BENCHMARK / "libreusebench" / "data"
REPORT_PATH = _BENCHMARK / "docs" / "REPORT.md"

# the BigCodeBench version the whole build is pinned to
BCB_VERSION = "v0.1.4"

# the interpreter that runs model-generated code during the build: an isolated, throwaway venv
# (create it with benchmark/pipeline/setup_build_venv.sh), never the project's own. library
# solutions' packages are installed there while they are checked, and removed again afterwards
BUILD_PYTHON = os.environ.get("LRB_BUILD_PYTHON") or str(
    _BENCHMARK.parent / ".venv-build" / "bin" / "python"
)

# the stage-01 BigCodeBench seeds each track generates tasks from: "reuse" seeds use a
# third-party library, "control" seeds use only the standard library
SEED_FILES = {
    "reuse": "01_bcb_extlibs.jsonl",
    "control": "01_bcb_stdlibs.jsonl",
}
TRACKS = tuple(SEED_FILES)


def artefact_path(
    name: str,
) -> Path:
    """Path to a stage artefact (e.g. "02_generated.jsonl") under the artefacts directory.

    Returns the full path.
    """
    return ARTEFACTS_DIR / name


def track_artefact_path(
    name: str,
    track: str,
) -> Path:
    """Path to a stage artefact, suffixed for a non-default track (e.g. "control").

    The "reuse" track keeps the plain stage filename; any other track gets its name inserted
    before the extension (e.g. "02_generated.jsonl" -> "02_generated_control.jsonl"), so the
    two tracks' artefacts never collide.

    Returns the full path.
    """
    if track == "reuse":
        return artefact_path(name)
    stem, suffix = name.rsplit(".", 1)
    return artefact_path(f"{stem}_{track}.{suffix}")


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
    """Ids whose latest record succeeded (kept=True).

    Returns the set of succeeded ids (empty if the artefact does not exist yet).
    """
    return {k for k, row in latest_records(path, key).items() if row.get("kept")}


def task_hash(
    task: dict,
) -> str:
    """Fingerprint a generated task: its wording, its tests and its gold solutions.

    Reviews store the fingerprint of the task they looked at, so a review of an earlier version
    of a task (before it was regenerated) is never applied to the current one.

    Returns a short hex digest.
    """
    parts = [
        task["prompt"],
        task["stdlib_solution"],
        task.get("extlib_solution", ""),
        *task["test_inputs"],
        *task["expected_outputs"],
    ]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


# --- the tasks, as they move through the stages ----------------------------------------


def generated_tasks(
    track: str = "reuse",
) -> list[dict]:
    """The tasks stage 02 kept for a track: each seed's latest record, where it passed.

    Returns the task records.
    """
    path = track_artefact_path("02_generated.jsonl", track)
    return [task for task in latest_records(path).values() if task.get("kept")]


def agent_reviews(
    track: str = "reuse",
) -> dict[str, dict]:
    """Stage 03's review of each kept task, where it is of the task's current version.

    A review of an earlier version (one that has since been regenerated) is left out, so that
    task counts as not reviewed yet.

    Returns the {seed_id: review record} map.
    """
    reviews = latest_records(track_artefact_path("03_agent_reviews.jsonl", track))
    return {
        task["seed_id"]: reviews[task["seed_id"]]
        for task in generated_tasks(track)
        if reviews.get(task["seed_id"], {}).get("task_hash") == task["task_hash"]
    }


def agent_kept_tasks(
    track: str = "reuse",
) -> list[dict]:
    """The tasks of a track that the agent review kept: the pool that human review looks at.

    Returns the task records.
    """
    reviews = agent_reviews(track)
    return [
        task
        for task in generated_tasks(track)
        if reviews.get(task["seed_id"], {}).get("verdict") == "keep"
    ]


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


# --- reuse value ------------------------------------------------------------------------

# a task is only in the "high" tier if its standard-library solution is also at least this many
# times as long as its library solution - so that a long solution which a library barely
# shortens (e.g. 120 lines against 100) doesn't count as high value. the tiers' bounds in lines
# saved are set in libreusebench.evaluate (REUSE_TIER_MEDIUM and REUSE_TIER_HIGH)
HIGH_TIER_MIN_RATIO = 1.5


def reuse_stats(
    stdlib_solution: str,
    extlib_solution: str,
) -> dict:
    """Measure how much code a library saves on a task, from its two gold solutions.

    The reuse value is the standard-library solution's logical lines of code minus the library
    solution's (one logical line per statement, measured with radon). The tier is the harness's
    tier for that value, except that "high" also needs the ratio of HIGH_TIER_MIN_RATIO.

    Returns {stdlib_lloc, extlib_lloc, reuse_value, reuse_ratio, reuse_tier}.
    """
    stdlib_lloc, _ = code_metrics(stdlib_solution)
    extlib_lloc, _ = code_metrics(extlib_solution)
    if stdlib_lloc is None or extlib_lloc is None:
        raise ValueError("a gold solution could not be parsed")

    value = stdlib_lloc - extlib_lloc
    ratio = stdlib_lloc / max(extlib_lloc, 1)
    tier = reuse_tier(value)
    if tier == "high" and ratio < HIGH_TIER_MIN_RATIO:
        tier = "medium"
    return {
        "stdlib_lloc": stdlib_lloc,
        "extlib_lloc": extlib_lloc,
        "reuse_value": value,
        "reuse_ratio": round(ratio, 2),
        "reuse_tier": tier,
    }
