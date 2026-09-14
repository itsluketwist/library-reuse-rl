"""Build the library-usage dataset - the main focus of this project.

Each of the 66 code categories (taxonomy.yaml) pairs mature python libraries with concrete
feature tasks (tasks.yaml) whose natural solution reaches for one of those libraries. Every
task is wrapped in three modes (templates.yaml) - asked inside one vendored python repo's
context (taxonomy.yaml's repo_hint), as a bare "how do I implement x", or as a stdin/stdout
script reminiscent of the LibReuseBench eval benchmark - so the model sees the same
library-reuse decision worded several ways. The reward is programmatic: any used external
import (after subtracting stdlib and the repo's own local_modules) counts as reuse, so no
labels or execution are needed.

The hybrid LLM layer (augment.py) adds one extra claude-generated wording per mode from the
committed simple_prompts.json / script_prompts.json / repo_prompts.json caches when present;
without them the template-only build is fully deterministic.

Usage: python data/build/library/build.py
"""

import random
import re
from pathlib import Path

import yaml
from llm_cgr import save_jsonl

from data.build.contexts import Context, build_contexts
from data.build.library.augment import load_prompt_cache
from data.build.paths import (
    LIBRARY_REPOS_DIR,
    LIBRARY_TAXONOMY_FILE,
    LIBRARY_USAGE_DIR,
    REPOS_DIR,
)
from data.build.split import assign_task_ids, split_disjoint_test


SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# reward-reference version recorded on every row so a reward stays reproducible (matches the
# amplify allowlists version - both are regenerated together when the library surface changes)
VERSION = "2026-08"

# how the dataset was constructed, stamped on every row. v1 was the deterministic template-only
# build. v2 is the 3-mode (repo/simple/script) x 2-source (template/claude) restructure: every
# task gets exactly one row per (mode, source) - 6 rows/task once the claude caches are
# populated, 3 rows/task (template-only) while they're still empty.
DATASET_VERSION = "v2"

_HERE = Path(__file__).parent
TASKS_FILE = _HERE / "tasks.yaml"
TEMPLATES_FILE = _HERE / "templates.yaml"


def _task_lower(
    task: str,
) -> str:
    """Drop a task's trailing full stop and lowercase its first letter, for mid-sentence use.

    An acronym / all-caps first word (e.g. "POST") is left as-is - lowercasing it to "pOST"
    would be wrong.

    Returns the reworded task fragment.
    """
    fragment = task.rstrip(".")
    first = fragment.split(" ", 1)[0]
    if first.isupper() and len(first) > 1:
        return fragment
    return fragment[:1].lower() + fragment[1:]


def _task_stdio(
    task: str,
) -> str:
    """Reword a task fragment for stdin/stdout framing.

    Tasks are written function-style ("... and return the result."), which reads wrong inside
    a stdin/stdout ask - there is nothing to "return", only to print. Strips a trailing
    "return"-style clause on top of `_task_lower`'s stop/case handling. Only strips when there's
    a clause before it to keep (e.g. not "Return the prime factorisation..." - that has nothing
    left to say without it). This is a best-effort, deterministic text rewrite (no LLM) - it
    only needs to not read absurdly inside the generic script template; a concrete, unambiguous
    stdin/stdout spec is the claude variant's job.

    Returns the reworded fragment.
    """
    fragment = _task_lower(task)
    match = re.search(r",?\s*(?:and\s+)?returns?\b.*$", fragment, flags=re.IGNORECASE)
    if match and match.start() > 0:
        return fragment[: match.start()].rstrip(", ")
    return fragment


def _pick_one(
    candidates: list[str],
    task: str,
    salt: str,
) -> str | None:
    """Deterministically pick one candidate wording for a task.

    Reproducible per (SEED, salt, task) via a seeded RNG, so re-running the build without
    changing inputs always assigns the same wording to the same task - "random" only in the
    sense of spreading tasks evenly across the available wordings, not in being unstable
    across runs.

    Returns the picked candidate, or None if `candidates` is empty.
    """
    if not candidates:
        return None
    return random.Random(f"{SEED}:{salt}:{task}").choice(candidates)


def _render(
    template: str,
    task: str,
    context: Context | None,
) -> str:
    """Fill a phrasing template with the task (and repo context, for repo-mode templates).

    Returns the rendered prompt.
    """
    return template.format(
        context=context.summary if context else "",
        task=task,
        task_lower=_task_lower(task),
        task_stdio=_task_stdio(task),
    )


def _row(
    prompt: str,
    category_id: str,
    task: str,
    useful_libs: list[str],
    context: Context | None,
    phrasing_mode: str,
    source: str,
) -> dict:
    """Assemble one library-usage row.

    Returns the row dict.
    """
    return {
        "prompt": prompt,
        "task_type": "library_usage",
        "category_id": category_id,
        # the canonical task this row is a wording of - used to keep every wording of a task
        # in the same split, so the test set can hold out whole (unseen) tasks
        "task": task,
        "language": "python",
        "context_id": context.context_id if context else "none",
        "local_modules": context.local_modules if context else [],
        "useful_libs": useful_libs,
        "phrasing_mode": phrasing_mode,
        "allowlist_version": VERSION,
        "dataset_version": DATASET_VERSION,
        "source": source,
    }


def build_library_usage_rows(
    taxonomy: dict,
    tasks: dict[str, list[str]],
    templates: dict[str, list[str]],
    contexts_by_id: dict[str, Context],
    prompt_caches: dict[str, dict[str, list[str]]],
) -> list[dict]:
    """Build library-usage rows: each task, in each of the 3 modes, from each of the 2 sources.

    Every task gets exactly one template row and (once the cache has an entry) one claude row
    per mode - 6 rows/task in steady state, 3 rows/task (template-only) while the claude caches
    are still empty. Repo-mode rows are set in the one repo named by the task's category's
    taxonomy.yaml repo_hint; simple- and script-mode rows carry no context.

    Returns the library-usage rows.
    """
    rows = []
    for entry in taxonomy.get("code", []):
        cid = entry["id"]
        useful_libs = entry["seed_libraries"]
        repo_context = contexts_by_id[entry["repo_hint"]]

        for task in tasks.get(cid, []):
            for mode, context in (
                ("repo", repo_context),
                ("simple", None),
                ("script", None),
            ):
                # every mode has a fixed, non-empty template list, so this is never None
                template = _pick_one(templates[mode], task, salt=mode)
                assert template is not None
                rows.append(
                    _row(
                        _render(template, task, context),
                        cid,
                        task,
                        useful_libs,
                        context,
                        phrasing_mode=mode,
                        source="template",
                    )
                )

                claude_prompt = _pick_one(
                    prompt_caches[mode].get(task, []), task, salt=f"{mode}-llm"
                )
                if claude_prompt is not None:
                    rows.append(
                        _row(
                            claude_prompt,
                            cid,
                            task,
                            useful_libs,
                            context,
                            phrasing_mode=mode,
                            source="llm",
                        )
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
    """Build the library-usage dataset from the taxonomy, tasks, templates and cache."""
    taxonomy = yaml.safe_load(LIBRARY_TAXONOMY_FILE.read_text())
    tasks = yaml.safe_load(TASKS_FILE.read_text())
    templates = yaml.safe_load(TEMPLATES_FILE.read_text())
    prompt_caches = {
        mode: load_prompt_cache(mode) for mode in ("simple", "script", "repo")
    }

    # repo-mode context comes from the vendored python-api plus the committed repos we author
    # (pytoolbox, ...); each category picks exactly one of these via taxonomy.yaml's repo_hint
    contexts = {**build_contexts(REPOS_DIR), **build_contexts(LIBRARY_REPOS_DIR)}
    contexts_by_id = {
        c.context_id: c for c in contexts.values() if c.language == "python"
    }
    print(
        f"using {len(contexts_by_id)} python repo context(s) for repo-mode rows: "
        f"{sorted(contexts_by_id)}"
    )
    missing_hints = sorted(
        entry["id"]
        for entry in taxonomy.get("code", [])
        if entry["repo_hint"] not in contexts_by_id
    )
    if missing_hints:
        raise ValueError(
            f"taxonomy.yaml repo_hint doesn't match any python context for: {missing_hints}"
        )

    rows = dedupe(
        build_library_usage_rows(
            taxonomy, tasks, templates, contexts_by_id, prompt_caches
        )
    )
    # test holds out whole (unseen) tasks; val is an in-distribution, category-stratified
    # subset of the seen tasks - so val tracks learning and test tracks generalisation
    train, val, test = split_disjoint_test(
        rows,
        group_key=lambda row: row["task"],
        stratify_key=lambda row: row["category_id"],
        test_frac=TEST_FRAC,
        val_frac=VAL_FRAC,
        seed=SEED,
    )
    # label rows within the splits (test -> train -> val); the "U" tags the library-usage dataset
    train, val, test = assign_task_ids(train, val, test, prefix="LRRL/U")

    LIBRARY_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        save_jsonl(split_rows, str(LIBRARY_USAGE_DIR / f"{split_name}.jsonl"))
    print(
        f"library_usage: {len(rows)} rows -> train {len(train)}, "
        f"val {len(val)}, test {len(test)} in {LIBRARY_USAGE_DIR}/"
    )


if __name__ == "__main__":
    main()
