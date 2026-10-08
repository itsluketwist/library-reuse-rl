# Datasets

All dataset download/construction code and the built datasets live here (the exception is the
**LibReuseBench** eval benchmark, which is built and bundled in `benchmark/`). The folder is
called `data` (not `datasets`) so it never shadows the HuggingFace `datasets` library.

- **`train/`** — built training splits (`{train,val,test}.jsonl` each). `library_usage` is
  committed; `skywork` is a large download and `product_choice` depends on an upstream seed
  without a redistribution licence, so those two are gitignored and must be built locally.
- **`eval/`** — the committed evaluation sets (`livecodebench.jsonl`, `apps_introductory.jsonl`).
- **`build/`** — all the construction code (an installed package, `data.build.*`, run by path).
  `audit.py` writes one report per pipeline: `build/library/docs/REPORT.md` (library_usage, the
  main focus) and `build/amplify/docs/REPORT.md` (everything else, e.g. product_choice).
- **`build/library/docs/`** — `CONSTRUCTION.md` (how the library_usage dataset is built),
  `REPORT.md`, `PROMPT_COMPARISON.md`.

The `product_choice` split and the amplify seed are derived from amplifying.ai's **claude-code-picks**
study — [blog](https://amplifying.ai/research/claude-code-picks) ·
[repo](https://github.com/amplifying-ai/claude-code-picks). Credit to them for the prompts, the
test repos, and the tool-frequency data we build on. The split is not redistributed here because
the upstream seed has no licence.

## Three task styles

| Task | Split | Model produces | Reward | Focus |
| --- | --- | --- | --- | --- |
| `implement_tested` | `train/skywork/` | code (stdin/stdout) | execution vs tests | option |
| `product_choice` | `train/product_choice/` | prose pick | allowlist prose match, no execution | option |
| `library_usage` | `train/library_usage/` | code (in/around a repo) | **import classification, no execution** | **main focus** |

`library_usage` is the one we care about: 66 code categories where a mature python library
exists but a model is tempted to hand-roll it. Reuse is detected programmatically — extract the
solution's imports, subtract the standard library and the repo's own modules (`local_modules`),
and any remaining *used* import is genuine reuse (no labels, no execution). Each task is asked
in 3 modes — set **in a python repo's context** (`repo`), as a **bare "how do I implement X"**
(`simple`), or as a **stdin/stdout script** (`script`, in the LibReuseBench eval benchmark's
template) — each from 2 sources, a deterministic template and claude-generated wordings (one for
`repo` and `simple`, three staged ones for `script`) - plus a project-framed `repo` template
row with no repo context (`source: "project"`), so every task has up to 9 rows. See
`src/grpo/rewards/library_usage.py`.

**Ids.** Every row carries a unique `task_id` in the benchmark's style — `LRRL/U####` for
`library_usage`, `LRRL/P####` for `product_choice`. Ids are assigned within the splits, numbered
in test → train → val order, so they are contiguous per split and the test set starts at `…0000`.

**Versioning.** Every `library_usage` row carries a `dataset_version`. **v1** was the
deterministic, template-only build. **v2** is the 3-mode (`repo`/`simple`/`script`) x
2-source (`template`/`claude`) build: every task gets one row per (mode, source) - 6 rows/task
once the claude caches (`build/library/*_prompts.json`) are populated via `augment.py`'s CLI, 3
rows/task (template-only) while a cache is still empty. **v3** (current) replaces the one claude
`script` wording with three staged ones in LibReuseBench's template (`outline` / `precise` /
`benchmark`, recorded as `script_stage`), because the v2 wording gave almost no library-use
signal, and adds a project-framed `repo` row with no repo context ("... so that I can use it in my
project") -
9 rows/task, or 6 for a task that can't be posed as stdin/stdout. Bump
`DATASET_VERSION` in `build/library/build.py` when the construction method changes again.
Eval results record the `dataset_version` they scored, since a rebuild renumbers every id.

**Splits.** Made by `build/split.py:split_disjoint_test`, with two deliberately different holdouts:

- **test** holds out **whole, unseen tasks** — every wording of a held-out task is removed from
  training, so test measures generalisation to tasks the model never trained on. Test tasks are
  spread across categories (at most one per category, only from categories with >1 task, so no
  category is emptied). With v2's 194 tasks across 66 categories this covers ~19 categories.
- **val** is an in-distribution, **`category_id`-stratified** subset of the *seen* tasks (held-out
  rows/wordings of tasks that are also in train, across all 66 categories) — so val tracks whether
  the model is learning the training distribution, while test is the final unseen check.

## `build/`

Shared helpers (used by both pipelines):

- **`paths.py`** — every shared path (vendor dir, taxonomies, output dirs, report).
- **`contexts.py`** — turn a repo into prompt context (+ `local_modules`).
- **`split.py`** — stratified train/val/test split.
- **`download_skywork.py`** — the `implement_tested` / skywork code split → `train/skywork/`.
- **`download_eval.py`** — the LiveCodeBench + APPS eval sets → `eval/`.
- **`audit.py`** — summarise the built splits, one report per pipeline: `library/docs/REPORT.md`
  and `amplify/docs/REPORT.md`.

**`build/amplify/`** — the amplify (`product_choice`) pipeline (an option, not expanded):

- **`vendor_seed.py`** — fetch the amplify seed (prompts + tool-frequency extractions + repos)
  into `build/vendor/` (gitignored, pinned revision). Shared prerequisite — run it first; the
  library pipeline reuses the vendored python-api repo as one of its contexts.
- **`taxonomy.yaml`** — the 20 amplify product categories.
- **`allowlists.py`** — build `config/library_allowlists.yaml` from the taxonomies + amplify data.
- **`build.py`** — build the `product_choice` split → `train/product_choice/`.
- **`docs/REPORT.md`** — `audit.py`'s report for every dataset besides library_usage.

**`build/library/`** — the library-usage pipeline (the main focus):

- **`taxonomy.yaml`** — the 66 code categories (`seed_libraries` are pip import names;
  `repo_hint` picks which repo below a category's repo-mode rows are set in).
- **`tasks.yaml`** — the concrete feature tasks per category.
- **`templates.yaml`** — 2 deterministic phrasing templates per mode (`repo`/`simple`/`script`);
  build.py deterministically picks one per task.
- **`repos/`** — committed repos we author (e.g. `pytoolbox`, a deliberately broad toolbox) used
  as repo-mode contexts, so every category has a plausible repo it could be added to.
- **`augment.py`** — LLM seam that adds claude-generated wordings (one per task for `repo` and
  `simple`, three staged ones for `script`); also the CLI (`python data/build/library/augment.py
  --type simple|script|repo|all`) that regenerates the caches offline (needs `ANTHROPIC_API_KEY`;
  build.py itself never calls an LLM).
- **`simple_prompts.json` / `repo_prompts.json` / `script_stage_prompts.json`** — committed
  caches of claude-generated wordings the build folds in (empty ⇒ templates only).
  `script_prompts.json` holds the retired v2 script wordings, kept only for the script pilot.
- **`build.py`** — build the `library_usage` split → `train/library_usage/`; with `--pilot N`,
  the script pilot file instead (`pilot/script_pilot.jsonl`, see `docs/CONSTRUCTION.md`).
- **`docs/`** — `CONSTRUCTION.md` (how this all fits together), `REPORT.md`, `PROMPT_COMPARISON.md`
  (sample prompts from each mode/source, for reviewing `augment.py`'s output by eye).

## Run order (from the repo root)

```
# training data
python data/build/amplify/vendor_seed.py    # -> data/build/vendor/ (needed by both pipelines)
python data/build/amplify/allowlists.py     # -> config/library_allowlists.yaml
python data/build/amplify/build.py          # -> data/train/product_choice/
python data/build/library/build.py          # -> data/train/library_usage/   (the focus)
python data/build/audit.py                  # -> */docs/REPORT.md (library + amplify)
python data/build/download_skywork.py       # -> data/train/skywork/  (large, optional)

# evaluation benchmarks
python data/build/download_eval.py          # -> data/eval/
```

## Training / eval

```
run_grpo -m <model> -d data/train/library_usage  -rp lu-v1
run_grpo -m <model> -d data/train/product_choice -rp product_choice
run_eval -m <model> -d library_usage,product_choice
```

The eval dataset configuration stores an explicit `evaluator` beside every path. The
library-usage evaluator examines every fenced answer code block and reports valid-code
rate, reuse rate, appropriate-library rate, average distinct libraries used, library counts,
and per-phrasing-mode breakdowns (`repo`/`simple`/`script`). LibReuseBench uses its package's complete official
evaluator:
it discovers and installs the generated responses' dependencies in an isolated environment,
runs the official tests, and saves the official correctness and reuse statistics together with
the requested packages and resolved environment versions.
