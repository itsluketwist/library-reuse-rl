"""Build the library-usage dataset - the main focus of this project.

Each of the ~50 code categories (taxonomy.yaml) pairs mature python libraries with concrete
feature tasks (tasks.yaml) whose natural solution reaches for one of those libraries. Every
task is wrapped in several phrasings (templates.yaml) across two modes - asked inside a
vendored python repo's context, or as a bare "how do I implement x" - so the model sees the
same library-reuse decision worded many ways. The reward is programmatic: any used external
import (after subtracting stdlib and the repo's own local_modules) counts as reuse, so no
labels or execution are needed.

The hybrid LLM layer (augment.py) adds extra natural bare-mode wordings from the committed
phrasings.json cache when present; without it the template-only build is fully deterministic.

Usage: python data/build/library/build.py
"""

from pathlib import Path

import yaml
from llm_cgr import save_jsonl

from data.build.contexts import Context, build_contexts
from data.build.library.augment import load_phrasings_cache
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

# how the dataset was constructed, stamped on every row. v1 is the deterministic
# template-only build; v2 will fold in the LLM-diversified phrasings (augment.py -> phrasings.json)
# and any further expansion work. bump this when that lands.
DATASET_VERSION = "v1"

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
    python_contexts: list[Context],
    phrasings_cache: dict[str, list[str]],
) -> list[dict]:
    """Build library-usage rows: each task, in each phrasing mode and wording.

    Repo-mode rows are set in each vendored python repo's context; bare-mode rows carry no
    context. Template wordings are deterministic; cached LLM wordings add extra bare-mode rows.

    Returns the library-usage rows.
    """
    rows = []
    for entry in taxonomy.get("code", []):
        cid = entry["id"]
        useful_libs = entry["seed_libraries"]
        for task in tasks.get(cid, []):
            # repo-context wordings, one set per vendored python repo
            for context in python_contexts:
                for template in templates["repo"]:
                    rows.append(
                        _row(
                            _render(template, task, context),
                            cid,
                            task,
                            useful_libs,
                            context,
                            phrasing_mode="repo",
                            source="template",
                        )
                    )

            # bare wordings from the fixed templates
            for template in templates["bare"]:
                rows.append(
                    _row(
                        _render(template, task, context=None),
                        cid,
                        task,
                        useful_libs,
                        context=None,
                        phrasing_mode="bare",
                        source="template",
                    )
                )

            # bare wordings from the hybrid LLM cache (each is already a full ask)
            for phrasing in phrasings_cache.get(task, []):
                rows.append(
                    _row(
                        f"{phrasing} Write it in Python.",
                        cid,
                        task,
                        useful_libs,
                        context=None,
                        phrasing_mode="bare",
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
    phrasings_cache = load_phrasings_cache()

    # repo-mode context comes from the vendored python-api plus the committed repos we author
    # (pytoolbox, ...), so every category has a plausible repo it could be added to
    contexts = {**build_contexts(REPOS_DIR), **build_contexts(LIBRARY_REPOS_DIR)}
    python_contexts = [c for c in contexts.values() if c.language == "python"]
    print(
        f"using {len(python_contexts)} python repo context(s) for repo-mode rows: "
        f"{sorted(c.context_id for c in python_contexts)}"
    )

    rows = dedupe(
        build_library_usage_rows(
            taxonomy, tasks, templates, python_contexts, phrasings_cache
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
