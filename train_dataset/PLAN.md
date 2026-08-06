# Library-reuse datasets

Datasets that teach models to **reuse existing libraries instead of reimplementing
functionality**, built by expanding the [amplifying.ai "claude-code-picks"](https://github.com/amplifying-ai/claude-code-picks)
study into RL training data. Where the project's original Skywork data (self-contained
stdin/stdout puzzles) rarely needs a library, here **library choice is the task**, so the
reward is verifiable without executing code.

## Three tasks

| Task | Model produces | Verified by | Dataset |
| --- | --- | --- | --- |
| `implement_tested` | code (stdin/stdout) | executing vs test cases | existing `data/skywork/` |
| `library_usage` | code (in a repo) | import classification, no execution | `train_dataset/data/library_usage/` |
| `product_choice` | prose pick | allowlist prose match, no execution | `train_dataset/data/product_choice/` |

Each has a reward class under [`src/grpo/rewards/`](../src/grpo/rewards) (base
`LibraryReward` → `ProductChoiceReward`, and `CodeReward` →
`TestedImplementationReward` / `LibraryUsageReward`), selected by
`build_reward`. Reward profiles live in [`config/rewards.yaml`](../config/rewards.yaml).

**Verification asymmetry.** Recommendation is prose, so an **allowlist** is the core
verifier (was the primary pick an external tool vs DIY?). Library-implementation is code,
so reuse is detected **programmatically**: extract imports, subtract stdlib + the project's
own/relative modules (`local_modules`), and any remaining, actually-used import is a
genuine external library — no allowlist needed. The allowlist there is only an optional
`appropriate_lib_used` bonus.

## Pipeline (run in order, from the repo root)

Numbered build stages live in [`pipeline/`](pipeline) and run as scripts (their filenames
start with a digit, so they are invoked by path, not `-m`). `train_dataset` is an installed
package (see pyproject), so no `PYTHONPATH` is needed — run from the repo root:

```
python train_dataset/pipeline/01_vendor_seed.py       # -> train_dataset/vendor/ (gitignored)
python train_dataset/pipeline/02_build_allowlists.py  # -> config/library_allowlists.yaml
python train_dataset/pipeline/03_build_dataset.py     # -> train_dataset/data/{recommend,implement}/{train,val,test}.jsonl
python train_dataset/pipeline/04_audit.py             # -> train_dataset/REPORT.md
```

Shared, non-stage code sits at the package root:
- [`common.py`](common.py) — the shared paths every stage resolves from (the stages live
  one level down in `pipeline/`).
- [`taxonomy.yaml`](taxonomy.yaml) — the category taxonomy: 20 amplify product categories
  (→ product choice) and code-level library categories (→ library usage).
- [`contexts.py`](contexts.py) — turns each vendored repo into prompt context (+ `local_modules`).
- [`augment.py`](augment.py) — the LLM seam for scaling up (generated phrasings, synthetic
  contexts, real-OSS ingestion). **Inert until a generation model is wired in** — this is the
  main open decision for scale-up; the seed build above needs no LLM.

Generated data and the vendored seed are gitignored (regenerable, mirroring
`data/skywork/`); the curated `taxonomy.yaml` and `config/library_allowlists.yaml` are
committed.

## Row schema

```json
// product_choice
{ "prompt": "<repo context>\n\n<question>", "task_type": "product_choice",
  "category_id": "databases", "language": "typescript", "context_id": "...",
  "allowlist": ["postgresql", "postgres", ...], "diy_terms": ["json file", ...],
  "allowlist_version": "2026-08", "source": "amplify" }

// library_usage
{ "prompt": "<python repo context>\n\n<feature task>", "task_type": "library_usage",
  "category_id": "pdf", "language": "python", "context_id": "...",
  "local_modules": ["src"], "useful_libs": ["pypdf", "pdfplumber"],
  "allowlist_version": "2026-08", "source": "template" }
```

Reward-reference fields ride along as dataset columns, so the reward reads them from the
batch without any execution or file lookup. The loader ([`src/data.py`](../src/data.py))
detects this schema (no `ground_truth`) and passes the columns through. Eval scores these
splits with the same deterministic checks (see [`src/eval/scoring.py`](../src/eval/scoring.py)),
reporting `external_pick_rate`/`diy_rate` (product choice) and
`external_lib_used_rate`/`appropriate_lib_rate` (library usage) — comparable to the amplify study.

## Training / eval

```
run_grpo -m <model> -d train_dataset/data/product_choice -rp product_choice
run_grpo -m <model> -d train_dataset/data/library_usage  -rp library_usage
run_eval -m <model> -d product_choice,library_usage
```
