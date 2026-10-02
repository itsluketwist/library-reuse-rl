# Iteration 01: GRPO on Qwen3-8B, with and without the `ext` system prompt

Two GRPO runs that are identical except for the system prompt used during training. Each run's training metrics are in `{run_id}/grpo/` and each epoch's evals in `{run_id}/epoch{n}/`. Plots for either run come from `notebooks/01_grpo_run_analysis.ipynb` (set `RUN_ID`).

## Training choices

| | `qwen3-8b-4796` | `qwen3-8b-7405` |
|---|---|---|
| system prompt in training | none (`base`) | `ext`: nudges towards third-party libraries (`src/prompts.py`) |

Both runs share everything else:

- **model**: Qwen3-8B, full fine-tuning, thinking mode on
- **data**: the training split of `library_usage` v2, 916 prompts (see [Training data](#training-data))
- **reward**: `lu-v1`, scored by lure 0.1.2. It gives 0.25 for correct format, 0.25 for code that parses, 0.5 for importing a third-party library and 1.0 for using it (max 2.0).
- **GRPO** (`config/training.yaml`): 5 epochs, 140 steps, 32 prompts x 8 rollouts per step. The learning rate is 3e-6, three times the recipe default, with cosine decay to 10%. The KL penalty is 0.01, and the limit is 8192 new tokens at temperature 1.0.
- **compute**: about 17 hours per run on 4 GPUs (Isambard), seed 42

## Training data

This is the `library_usage` dataset as it stood for these runs. It will be changed for the next iteration.

- **version**: every row is tagged `dataset_version: v2` and `allowlist_version: 2026-08`. The files were last changed in commit `c362305` (2026-09-15, "Update library usage dataset."). To restore them: `git checkout c362305 -- data/train/library_usage`.
- **SHA-256**: train `e0c3d59b156f…`, val `6cea171eb0c9…`, test `052b9d4cccf7…`
- **how it was built**: `data/build/library/build.py`, from hand-written categories (`taxonomy.yaml`), tasks (`tasks.yaml`) and wording templates, plus cached wordings written by Claude. See `data/build/library/docs/CONSTRUCTION.md`.
- **what a prompt asks for**: a small Python feature that a mature library already solves, but that a model may be tempted to write by hand. Each row lists the libraries that would count as good reuse (`useful_libs`).
- **wordings**: each of the 194 tasks is worded in 3 styles. Each style has one template wording, plus one Claude wording if it passed a check that it doesn't name the library.
  - `repo`: a request inside one of two small projects, with one of its files shown
  - `simple`: a short, bare request
  - `script`: a stdin/stdout exercise, in the style of LibReuseBench
- **benchmark alignment**: 26 of the 66 categories are "benchmark-aligned", meaning their skills were picked from what LibReuseBench tests.

| split | prompts | tasks | categories | repo / simple / script | written by Claude | benchmark-aligned |
|---|---|---|---|---|---|---|
| train | 916 | 175 | 66 | 302 / 313 / 301 | 442 (48%) | 549 (60%) |
| val | 101 | 92 | 66 | 30 / 35 / 36 | 50 (50%) | 60 (59%) |
| test | 108 | 19 | 19 | 36 / 36 / 36 | 51 (47%) | 66 (61%) |

Val holds out other wordings of training tasks, so it measures the same tasks phrased differently. Test holds out 19 whole tasks (one per category) that training never sees.

**Known weakness:** `script` prompts give GRPO almost nothing to learn from. After training, 82% of held-out script prompts still never used a library in any of their 5 samples (`4796`). Fixing this is the main goal of the next dataset version.

## Results

All evals are run without a system prompt and sample 5 responses per task. pass@5 counts a task as solved if any of its 5 samples passes. Library use is the share of responses that import and actually use a third-party library. LibReuseBench has 179 test tasks, seeded from problems whose solution used a library, and 79 control tasks, where no library should be needed.

| LibReuseBench | test pass@1 | test pass@5 | test library use | control pass@1 | control pass@5 | control library use |
|---|---|---|---|---|---|---|
| Qwen3-8B (before training) | 45.7% | 58.7% | 1.9% | 37.7% | 60.8% | 0.0% |
| `4796` (no prompt), epoch 5 | 53.7% | 68.7% | 4.1% | 45.6% | 59.5% | 0.0% |
| `7405` (`ext` prompt), epoch 5 | 52.6% | 69.3% | 3.8% | 47.1% | 68.4% | 0.0% |

On the held-out library-usage prompts (the training task), library use rose from 28.7% to 50.6% for `4796` and to 43.1% for `7405` (test split).

## Takeaways

- **The pass@1 gain comes from shorter responses, not better code.** On LibReuseBench test, the share of responses cut off at the token limit fell from 41% to 24% (`4796`) and 30% (`7405`). On tasks that finished both before and after training, the pass rate didn't change. `4796`'s best pass@1 is at epoch 1 (57.9%), when the fewest responses are cut off.
- **Library use barely transfers to LibReuseBench.** It went from 1.9% to about 4%, and the script-style training prompts closest to the benchmark only went from 6% to 12%. Library use on control stayed at 0%, so the model isn't forcing libraries where none are needed.
- **Training without the system prompt works better.** `4796` beats `7405` by 6.6 points on held-out library use (95% interval 4.0 to 9.2, resampling tasks), and all of the gap is on repo-style prompts. This counts against an `ext`-then-no-prompt curriculum.
- **Both runs stop improving after epoch 2.** The share of prompts whose 8 rollouts all score the same, and so teach nothing, rises from about 45% to 60-70%. The training data, not the learning rate or the number of epochs, now limits progress.
- **Caveats.**
  - Some of the library-use gain is cheap: the library appears only in a side snippet, such as a `click` CLI wrapper, while the solution itself uses only the standard library.
  - Imports of the project's own files, such as `from tasks import ...`, still count as third-party libraries.
  - Each configuration was run with one seed.
