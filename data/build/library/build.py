"""Build the library-usage dataset - the main focus of this project.

Each of the 66 code categories (taxonomy.yaml) pairs mature python libraries with concrete
feature tasks (tasks.yaml) whose natural solution reaches for one of those libraries. Every
task is wrapped in three modes (templates.yaml) - asked inside one vendored python repo's
context (taxonomy.yaml's repo_hint), as a bare "how do I implement x", or as a stdin/stdout
script reminiscent of the LibReuseBench eval benchmark - so the model sees the same
library-reuse decision worded several ways. The reward is programmatic: any used external
import (after subtracting stdlib and the repo's own local_modules) counts as reuse, so no
labels or execution are needed.

The hybrid LLM layer (augment.py) adds claude-generated wordings from committed caches when
present: one extra wording per task for repo and simple (repo_prompts.json / simple_prompts.json),
and three staged wordings per task for script (script_stage_prompts.json, see SCRIPT_STAGES in
augment.py). Without them the template-only build is fully deterministic.

`--pilot N` instead writes the script pilot file: just the script rows of the pilot's N tasks,
both the dataset-v2 wordings and the new staged ones, so a model's library use on each can be
compared in a single eval run (library_usage_script_pilot in config/eval_datasets.yaml).

Usage: python data/build/library/build.py [--pilot N]
"""

import argparse
import random
import re
from pathlib import Path
from typing import Any

import yaml
from llm_cgr import save_jsonl

from data.build.contexts import Context, build_contexts
from data.build.library.augment import (
    SCRIPT_STAGES,
    load_prompt_cache,
    load_script_stage_cache,
    pilot_seed_tasks,
)
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
# populated, 3 rows/task (template-only) while they're still empty. v3 replaces the one claude
# script wording with three staged ones in LibReuseBench's template (outline/precise/benchmark,
# see augment.py) beside the template script row, and adds a project-framed repo row with no
# repo context (source "project", templates.yaml's repo_project) - 9 rows/task, or 6 for a task
# that can't be posed honestly as stdin/stdout (it keeps only its template script row).
DATASET_VERSION = "v3"

_HERE = Path(__file__).parent
TASKS_FILE = _HERE / "tasks.yaml"
TEMPLATES_FILE = _HERE / "templates.yaml"

# the script pilot's rows (see --pilot) - kept beside the build, away from the training splits
PILOT_FILE = _HERE / "pilot" / "script_pilot.jsonl"


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
    seed_task: str,
    salt: str,
) -> str | None:
    """Deterministically pick one candidate wording for a task.

    Reproducible per (SEED, salt, seed_task) via a seeded RNG, so re-running the build without
    changing inputs always assigns the same wording to the same task - "random" only in the
    sense of spreading tasks evenly across the available wordings, not in being unstable
    across runs.

    Returns the picked candidate, or None if `candidates` is empty.
    """
    if not candidates:
        return None
    return random.Random(f"{SEED}:{salt}:{seed_task}").choice(candidates)


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
    seed_task: str,
    task: str,
    useful_libs: list[str],
    context: Context | None,
    phrasing_mode: str,
    source: str,
    script_stage: str | None = None,
) -> dict:
    """Assemble one library-usage row.

    `script_stage` names which script wording a script row is ("template", or one of
    SCRIPT_STAGES for the staged claude wordings); it is None for repo and simple rows.

    Returns the row dict.
    """
    return {
        "prompt": prompt,
        "task_type": "library_usage",
        "category_id": category_id,
        # the taxonomy id of the task this row is a wording of ("<category_id>.<number>",
        # matching tasks.yaml's per-category numbering) - used to keep every wording of a task
        # in the same split (so test can hold out whole, unseen tasks) and as the key into the
        # claude prompt caches
        "seed_task": seed_task,
        # the task's descriptive text, for readability/rendering - not itself an id
        "task": task,
        "language": "python",
        "context_id": context.context_id if context else "none",
        "local_modules": context.local_modules if context else [],
        "useful_libs": useful_libs,
        "phrasing_mode": phrasing_mode,
        "script_stage": script_stage,
        "allowlist_version": VERSION,
        "dataset_version": DATASET_VERSION,
        "source": source,
    }


def build_library_usage_rows(
    taxonomy: dict,
    tasks: dict[str, dict[int, str]],
    templates: dict[str, list[str]],
    contexts_by_id: dict[str, Context],
    prompt_caches: dict[str, dict[str, list[str]]],
    stage_cache: dict[str, dict[str, Any]],
) -> list[dict]:
    """Build library-usage rows: each task, in each of the 3 modes, from each source.

    Every task gets one template row per mode, plus a project-framed repo row with no repo
    context (source "project"). Repo and simple add one claude row each (once `prompt_caches` has an entry);
    script adds one claude row per stage of the task's staged wording (once `stage_cache` has
    an entry that fits stdin/stdout). That is 9 rows/task in steady state. Repo-mode rows are set in the one repo named by the task's category's
    taxonomy.yaml repo_hint; simple- and script-mode rows carry no context.

    Returns the library-usage rows.
    """
    rows = []
    for entry in taxonomy.get("code", []):
        cid = entry["id"]
        useful_libs = entry["seed_libraries"]
        repo_context = contexts_by_id[entry["repo_hint"]]

        for number, task in tasks.get(cid, {}).items():
            seed_task = f"{cid}.{number}"
            for mode, context in (
                ("repo", repo_context),
                ("simple", None),
                ("script", None),
            ):
                # every mode has a fixed, non-empty template list, so this is never None
                template = _pick_one(templates[mode], seed_task, salt=mode)
                assert template is not None
                rows.append(
                    _row(
                        _render(template, task, context),
                        cid,
                        seed_task,
                        task,
                        useful_libs,
                        context,
                        phrasing_mode=mode,
                        source="template",
                        script_stage="template" if mode == "script" else None,
                    )
                )

                # every task also gets the project-framed repo row ("... so that I can use it
                # in my project"), whichever repo template was picked above - the request alone,
                # with no repo context (so no local modules either)
                if mode == "repo":
                    rows.append(
                        _row(
                            _render(templates["repo_project"][0], task, None),
                            cid,
                            seed_task,
                            task,
                            useful_libs,
                            None,
                            phrasing_mode=mode,
                            source="project",
                        )
                    )

                if mode == "script":
                    for stage, prompt in _staged_prompts(stage_cache, seed_task):
                        rows.append(
                            _row(
                                prompt,
                                cid,
                                seed_task,
                                task,
                                useful_libs,
                                context,
                                phrasing_mode=mode,
                                source="llm",
                                script_stage=stage,
                            )
                        )
                    continue

                claude_prompt = _pick_one(
                    prompt_caches[mode].get(seed_task, []),
                    seed_task,
                    salt=f"{mode}-llm",
                )
                if claude_prompt is not None:
                    rows.append(
                        _row(
                            claude_prompt,
                            cid,
                            seed_task,
                            task,
                            useful_libs,
                            context,
                            phrasing_mode=mode,
                            source="llm",
                        )
                    )
    return rows


def _staged_prompts(
    stage_cache: dict[str, dict[str, Any]],
    seed_task: str,
) -> list[tuple[str, str]]:
    """List a task's staged script wordings, loosest first.

    A task with no cache entry, or one the generator judged can't be posed honestly as
    stdin/stdout (fits_stdio false), has none.

    Returns (stage, prompt) pairs.
    """
    entry = stage_cache.get(seed_task)
    if not entry or not entry.get("fits_stdio"):
        return []
    return [(stage, entry[stage]) for stage in SCRIPT_STAGES]


def build_script_pilot_rows(
    taxonomy: dict,
    tasks: dict[str, dict[int, str]],
    templates: dict[str, list[str]],
    v2_script_cache: dict[str, list[str]],
    stage_cache: dict[str, dict[str, Any]],
    pilot_tasks: list[str],
) -> list[dict]:
    """Build the script pilot's rows: every script wording of each pilot task, old and new.

    Per task: the template script row, the dataset-v2 claude script row (picked exactly as the
    v2 build picked it, marked script_stage "v2"), and the new staged rows - so one eval run
    scores old and new wordings of the same tasks side by side. Ids are "LRRL/S####".

    Returns the pilot rows.
    """
    wanted = set(pilot_tasks)
    rows = []
    for entry in taxonomy.get("code", []):
        cid = entry["id"]
        for number, task in tasks.get(cid, {}).items():
            seed_task = f"{cid}.{number}"
            if seed_task not in wanted:
                continue

            # the template row and the v2 claude row, picked with the same seeds as before
            template = _pick_one(templates["script"], seed_task, salt="script")
            assert template is not None
            wordings = [("template", "template", _render(template, task, None))]
            v2_prompt = _pick_one(
                v2_script_cache.get(seed_task, []),
                seed_task,
                salt="script-llm",
            )
            if v2_prompt is not None:
                wordings.append(("llm", "v2", v2_prompt))
            wordings += [
                ("llm", stage, prompt)
                for stage, prompt in _staged_prompts(stage_cache, seed_task)
            ]

            for source, stage, prompt in wordings:
                row = _row(
                    prompt,
                    cid,
                    seed_task,
                    task,
                    entry["seed_libraries"],
                    None,
                    phrasing_mode="script",
                    source=source,
                    script_stage=stage,
                )
                rows.append({"task_id": f"LRRL/S{len(rows):04d}", **row})
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="write only the script pilot file, for the pilot's sample of N tasks",
    )
    args = parser.parse_args()

    taxonomy = yaml.safe_load(LIBRARY_TAXONOMY_FILE.read_text())
    tasks = yaml.safe_load(TASKS_FILE.read_text())
    templates = yaml.safe_load(TEMPLATES_FILE.read_text())
    stage_cache = load_script_stage_cache()

    if args.pilot is not None:
        pilot_tasks = pilot_seed_tasks(taxonomy, tasks, args.pilot)
        missing = [
            seed_task for seed_task in pilot_tasks if seed_task not in stage_cache
        ]
        if missing:
            raise ValueError(
                f"no staged wording cached for pilot task(s) {missing} - run "
                f"`augment.py --type script --pilot {args.pilot}` first"
            )
        rows = build_script_pilot_rows(
            taxonomy,
            tasks,
            templates,
            v2_script_cache=load_prompt_cache("script"),
            stage_cache=stage_cache,
            pilot_tasks=pilot_tasks,
        )
        PILOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(rows, str(PILOT_FILE))
        print(
            f"script pilot: {len(rows)} rows for {len(pilot_tasks)} tasks in {PILOT_FILE}"
        )
        return

    prompt_caches = {mode: load_prompt_cache(mode) for mode in ("simple", "repo")}
    aligned_categories = {
        entry["id"]
        for entry in taxonomy.get("code", [])
        if entry.get("benchmark_aligned")
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
            taxonomy, tasks, templates, contexts_by_id, prompt_caches, stage_cache
        )
    )
    # a task without a cached staged wording silently gets no staged rows, so report coverage
    n_tasks = sum(len(category_tasks) for category_tasks in tasks.values())
    n_fit = sum(1 for entry in stage_cache.values() if entry.get("fits_stdio"))
    print(
        f"staged script wordings: {len(stage_cache)}/{n_tasks} tasks cached, "
        f"{n_fit} posed as stdin/stdout"
    )
    # test holds out whole (unseen) tasks; val is an in-distribution, category-stratified
    # subset of the seen tasks - so val tracks learning and test tracks generalisation. the
    # held-out test categories are balanced by benchmark_aligned so test's aligned/unaligned
    # row split tracks train/val's, instead of just the (lower) aligned share of categories
    train, val, test = split_disjoint_test(
        rows,
        group_key=lambda row: row["seed_task"],
        stratify_key=lambda row: row["category_id"],
        test_frac=TEST_FRAC,
        val_frac=VAL_FRAC,
        seed=SEED,
        balance_key=lambda row: row["category_id"] in aligned_categories,
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
