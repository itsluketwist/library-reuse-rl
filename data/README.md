# Datasets

The training and evaluation datasets, and the code that builds them. The LibReuseBench benchmark
is built separately, in `benchmark/`. The folder is called `data`, not `datasets`, so it never
shadows the HuggingFace `datasets` library.

- **`train/`**: built training splits, each as `{train,val,test}.jsonl`.
- **`eval/`**: committed eval sets, two regression tests of general coding ability:
  `humaneval_plus.jsonl`, and `livecodebench.jsonl` (the easy and medium function problems of
  LiveCodeBench releases v5 and v6).
- **`build/`**: the code that builds them, run by path from the repo root.

## The training datasets

| Dataset | Model produces | Reward | Status |
| --- | --- | --- | --- |
| `library_usage` | code | whether the code uses a library (from its imports, nothing is run) | **main focus**, committed |
| `implement_tested` (`train/skywork/`) | a stdin/stdout program | execution against tests | option, gitignored (large download) |
| `product_choice` | a prose recommendation | whether it names an allowlisted tool | option, gitignored (see below) |

### `library_usage`

- **Tasks:** 194 small feature tasks in 66 categories, each a job where a mature Python library
  exists but a model might write the code by hand. They are defined in `build/library/`
  (`taxonomy.yaml`, `tasks.yaml`).
- **Reward:** the response's imports, minus the standard library and the project's own modules
  (`local_modules`). Any remaining import the code uses counts as library use. The target
  components count only the libraries on the row's `target_libs` list. See
  `src/grpo/rewards/library_usage.py`.
- **Wordings:** each task is asked in three modes, up to 8 rows per task in the current
  version (v5, worded as v4):
  - `repo`, 3 rows: a template inside a real project's files, the same request with no project
    shown, and a claude-written request inside the project.
  - `simple`, 2 rows: a template and a claude-written request.
  - `function`, 3 rows: claude-written requests for one named function, in LibReuseBench's
    style: `name`, `signature` and `example`.
- **Splits** (`build/split.py`): **test** holds out 19 whole tasks, one per category, to measure
  generalisation to unseen tasks. **val** holds out some wordings of the tasks in train, to track
  learning.
- **Ids:** `LRRL/U####`, numbered test, then train, then val. Every rebuild renumbers them, so
  eval results record the `dataset_version` they scored.
- **More:** `build/library/HISTORY.md` describes each version and why it changed;
  `build/library/docs/CONSTRUCTION.md` explains how the rows are built.

### `product_choice`

Built from amplifying.ai's **claude-code-picks** study ([blog](https://amplifying.ai/research/claude-code-picks)
· [repo](https://github.com/amplifying-ai/claude-code-picks)), which we credit for the prompts,
test repos and tool-frequency data. The upstream seed has no licence, so the split is not
redistributed and must be built locally.

## `build/`

- **Shared:** `paths.py` (every path), `contexts.py` (a repo as prompt context, with its
  `local_modules`), `split.py`, `audit.py` (writes each pipeline's `docs/REPORT.md`),
  `download_skywork.py`, `download_eval.py`, `llm.py` (the claude client).
- **`amplify/`:** the `product_choice` pipeline. `vendor_seed.py` fetches the amplify seed into
  `build/vendor/` (gitignored); the library pipeline reuses its python-api repo, so run it first.
  `allowlists.py` writes `config/library_allowlists.yaml`.
- **`library/`:** the `library_usage` pipeline.
  - Inputs: `taxonomy.yaml`, `tasks.yaml`, `templates.yaml`, and the authored repo `repos/pytoolbox/`.
  - `augment.py` generates the claude-written wordings into committed caches
    (`simple_prompts.json`, `repo_prompts.json`, `function_stage_prompts.json`). It needs
    `ANTHROPIC_API_KEY`; `build.py` reads the caches and never calls a model.
  - `script_prompts.json` and `script_stage_prompts.json` hold the stdin/stdout wordings of v2
    and v3. Only the script pilot (`build.py --pilot N`) still uses them.

## Run order (from the repo root)

```
# training data
python data/build/amplify/vendor_seed.py    # -> data/build/vendor/ (needed by both pipelines)
python data/build/amplify/allowlists.py     # -> config/library_allowlists.yaml
python data/build/amplify/build.py          # -> data/train/product_choice/
python data/build/library/build.py          # -> data/train/library_usage/
python data/build/audit.py                  # -> */docs/REPORT.md
python data/build/download_skywork.py       # -> data/train/skywork/  (large, optional)

# evaluation sets
python data/build/download_eval.py          # -> data/eval/
```

## Training and eval

```
run_grpo -m <model> -d data/train/library_usage  -rp lu-usage
run_grpo -m <model> -d data/train/product_choice -rp product_choice
run_eval -m <model> -d library_usage,product_choice
```

Eval datasets are named in `config/eval_datasets.yaml`, each with its path and evaluator. The
library-usage evaluator reports valid-code, library-use and target-library rates, overall
and by mode. LibReuseBench uses its own package's evaluator, which installs what the responses
import and runs the tests.
