# Datasets

All dataset download/construction code and the built datasets live here (the exception is the
**LibReuseBench** eval benchmark, which is built and bundled in `benchmark/`). The folder is
called `data` (not `datasets`) so it never shadows the HuggingFace `datasets` library.

- **`train/`** — built training splits (`{train,val,test}.jsonl` each). `library_usage` is
  committed; `skywork` is a large download and `product_choice` depends on an upstream seed
  without a redistribution licence, so those two are gitignored and must be built locally.
- **`eval/`** — the committed evaluation sets (`livecodebench.jsonl`, `apps_introductory.jsonl`).
- **`build/`** — all the construction code (an installed package, `data.build.*`, run by path).
- **`REPORT.md`** — auto-generated audit of the built library-reuse splits (`build/audit.py`).

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

`library_usage` is the one we care about: `~50` code categories where a mature python library
exists but a model is tempted to hand-roll it. Reuse is detected programmatically — extract the
solution's imports, subtract the standard library and the repo's own modules (`local_modules`),
and any remaining *used* import is genuine reuse (no labels, no execution). Each task is asked in
several wordings, in two modes: set **in a python repo's context**, or a **bare "how do I
implement X"**. See `src/grpo/rewards/library_usage.py`.

**Ids.** Every row carries a unique `task_id` in the benchmark's style — `LRRL/U####` for
`library_usage`, `LRRL/P####` for `product_choice`. Ids are assigned within the splits, numbered
in test → train → val order, so they are contiguous per split and the test set starts at `…0000`.

**Versioning.** Every `library_usage` row carries a `dataset_version`. **v1** (current) is the
deterministic, template-only build — no LLM calls. **v2** (future) will fold in the
LLM-diversified phrasings via `build/library/augment.py` → `phrasings.json`, plus any further
expansion work; bump `DATASET_VERSION` in `build/library/build.py` when that lands.

**Splits.** Made by `build/split.py:split_disjoint_test`, with two deliberately different holdouts:

- **test** holds out **whole, unseen tasks** — every wording of a held-out task is removed from
  training, so test measures generalisation to tasks the model never trained on. Test tasks are
  spread across categories (at most one per category, only from categories with >1 task, so no
  category is emptied). With v1's 2–3 tasks/category this covers ~10 categories.
- **val** is an in-distribution, **`category_id`-stratified** subset of the *seen* tasks (held-out
  rows/wordings of tasks that are also in train, across all 50 categories) — so val tracks whether
  the model is learning the training distribution, while test is the final unseen check.

## `build/`

Shared helpers (used by both pipelines):

- **`paths.py`** — every shared path (vendor dir, taxonomies, output dirs, report).
- **`contexts.py`** — turn a repo into prompt context (+ `local_modules`).
- **`split.py`** — stratified train/val/test split.
- **`download_skywork.py`** — the `implement_tested` / skywork code split → `train/skywork/`.
- **`download_eval.py`** — the LiveCodeBench + APPS eval sets → `eval/`.
- **`audit.py`** — summarise the built library-reuse splits → `REPORT.md`.

**`build/amplify/`** — the amplify (`product_choice`) pipeline (an option, not expanded):

- **`vendor_seed.py`** — fetch the amplify seed (prompts + tool-frequency extractions + repos)
  into `build/vendor/` (gitignored, pinned revision). Shared prerequisite — run it first; the
  library pipeline reuses the vendored python-api repo as one of its contexts.
- **`taxonomy.yaml`** — the 20 amplify product categories.
- **`allowlists.py`** — build `config/library_allowlists.yaml` from the taxonomies + amplify data.
- **`build.py`** — build the `product_choice` split → `train/product_choice/`.

**`build/library/`** — the library-usage pipeline (the main focus):

- **`taxonomy.yaml`** — the ~50 code categories (`seed_libraries` are pip import names).
- **`tasks.yaml`** — the concrete feature tasks per category.
- **`templates.yaml`** — deterministic phrasing templates for the repo and bare modes.
- **`repos/`** — committed repos we author (e.g. `pytoolbox`, a deliberately broad toolbox) used
  as repo-mode contexts, so every category has a plausible repo it could be added to.
- **`augment.py`** — LLM seam that diversifies task wordings (hybrid); writes `phrasings.json`.
- **`phrasings.json`** — committed cache of LLM wordings the build folds in (empty ⇒ templates only).
- **`build.py`** — build the `library_usage` split → `train/library_usage/`.

## Run order (from the repo root)

```
# training data
python data/build/amplify/vendor_seed.py    # -> data/build/vendor/ (needed by both pipelines)
python data/build/amplify/allowlists.py     # -> config/library_allowlists.yaml
python data/build/amplify/build.py          # -> data/train/product_choice/
python data/build/library/build.py          # -> data/train/library_usage/   (the focus)
python data/build/audit.py                  # -> data/REPORT.md
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
and bare/repository-phrasing breakdowns. LibReuseBench uses its package's complete official
evaluator:
it discovers and installs the generated responses' dependencies in an isolated environment,
runs the official tests, and saves the official correctness and reuse statistics together with
the requested packages and resolved environment versions.
