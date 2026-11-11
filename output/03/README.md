# Iteration 03: three library rewards (import, usage, target), 8 epochs

Three GRPO runs on Qwen3-8B with the same recipe and data. They differ only in the reward. Each
run's training metrics are in `{run_id}/grpo/` and each epoch's evals in `{run_id}/epoch{n}/`.

- **Figures per run:** `{run_id}/lure-{reward}-summary.png`, made by
  `notebooks/01_grpo_training_summary.ipynb` (set `RUN_ID`).
- **Prompt comparison figures:** `prompt_engineering_summary.png` and
  `prompt_engineering_details.png`, made by `notebooks/05_prompt_engineering_comparison.ipynb`.
- **Base model results:** these stay in `output/baselines/qwen3-8b/`, because later iterations use
  them too. A rerun that records timings will replace those files. The numbers below are from the
  files as they stood in the commit that created this folder.

## What changed since iteration 02

- **Rewards:** three rewards replace `lu-v1` (see [The rewards](#the-rewards)). They are scored by
  lure 0.2.0, which adds the target-library components.
- **Benchmark:** the final LibReuseBench, with 216 test tasks (`libreusebench_test`) and 84
  control tasks (`libreusebench_control`), following BigCodeBench's split. Its numbers can't be compared with iteration 02's provisional `libreusebench_v2_full`.
- **Training data:** v5. The prompts are v4's, word for word. Only each row's list of target
  libraries (`target_libs`, previously `useful_libs`) changed, after an audit (see
  `data/build/library/HISTORY.md`).
- **Recipe:** 16 rollouts per prompt instead of 8. More groups then contain both library and
  non-library answers, which is what GRPO learns from. The runs were 8 epochs long. The held-out
  splits of the training data were scored every 5 steps during training.
- **Regression benchmarks:** HumanEval+ (164 tasks) and LiveCodeBench (87 easy and medium tasks)
  check that general coding ability doesn't drop.
- **Prompt-engineering baselines:** instead of training, a sentence is added to the end of each
  task: "Always use an external library in your solution." (`always`) or "Use an external
  library when appropriate." (`appropriate`).

## The rewards

Each answer earns the weight of every component it satisfies:

- `correct_format`: the answer contains a code block.
- `passes_syntax_check`: the code parses.
- `external_lib_imported` / `external_lib_used`: the code imports a third-party library, or
  imports it and actually calls it.
- `target_lib_imported` / `target_lib_used`: the same, but only for libraries listed for the
  task's category (its `target_libs`). For example, HTML parsing lists `beautifulsoup4`, `lxml`,
  `pandas` and five others.

| profile | run | format | syntax | library imported | library used | target imported | target used |
|---|---|---|---|---|---|---|---|
| `lu-import` | `qwen3-8b-4474` | 0.5 | 0.5 | 1.0 | 0 | 0 | 0 |
| `lu-usage` | `qwen3-8b-cc26` | 0.25 | 0.25 | 0.5 | 1.0 | 0 | 0 |
| `lu-target` | `qwen3-8b-1f71` | 0.25 | 0.25 | 0 | 0 | 0.5 | 1.0 |

- **import:** any third-party import earns the library credit, used or not.
- **usage:** importing earns part of the credit, and using the library earns the rest.
- **target:** as usage, but only a library on the task's target list counts.

GRPO compares the rewards within each prompt's group of 16 rollouts and normalises them. So only
the ratios between weights matter, not the maximum. A zero-weighted component is still logged, so
it can be checked, for example whether the import-only run uses what it imports.

## Training choices

All three runs share everything except the reward:

- **model:** Qwen3-8B, full fine-tuning, thinking mode on.
- **data:** the training split of `library_usage` v5, 1,209 prompts.
- **GRPO** (`config/training.yaml`):
  - 8 epochs of 37 steps (296 steps).
  - 32 prompts x 16 rollouts per step.
  - Learning rate 3e-6, warmed up over the first 2% of steps, then a cosine decay to 10%.
  - KL penalty 0.01.
  - Up to 8,192 new tokens at temperature 1.0, with no system prompt.
- **evaluation during training:** the val and test splits were scored with 5 answers per prompt
  before training and every 5 steps.
- **compute:** 4 GPUs (Isambard), seed 42. Each run logged about 59 hours of training steps, about
  12 minutes a step.
  - Saving a checkpoint left GPU memory that only a fresh process could free. So training stopped
    after each epoch's save, and the job relaunched it from that checkpoint.
  - Jobs requeued themselves at an epoch boundary when the next epoch wouldn't fit in the 24-hour
    limit.
  - `1f71` crashed during its epoch-6 save and was resumed under its own run id.
  - Each run's logged learning rate follows the schedule exactly across every resume.

**Why 8 epochs.** In iteration 02, run `8672` was still improving at its last epoch (6). So
these runs were given 8 epochs, with every epoch evaluated, to see how far the gain goes.

**What the 8 epochs showed.** LibReuseBench library use peaked at epoch 2-4 in every run, then
fell back. Library use on the training data kept rising (see [Results](#results)). The learning
rate was still high at the peak: 87% of its maximum at the end of epoch 2, 72% at epoch 3 and 55%
at epoch 4. So an epoch here can't be compared with the same epoch of a shorter run. The next
iterations therefore use 4 epochs for every run, so all runs share one learning-rate schedule.
They report the final epoch, chosen in advance rather than picked on the test set.

## Training data

| split | prompts | tasks held out |
|---|---|---|
| train | 1,209 | - |
| val | 135 | other wordings of training tasks |
| test | 145 | 19 whole tasks training never sees |

- **v5** files: commit `2e9714d` (2026-10-28). To restore them: `git checkout 2e9714d --
  data/train/library_usage`. SHA-256: train `35483b180f49…`, val `16d50eecfbe9…`, test
  `894e4e3e29e0…`.
- The rows are v4's, word for word, apart from `target_libs` and `dataset_version`. The audit
  removed `argparse` and added 166 import names across 61 categories. 21 of those came from what
  the model used, and the rest are well-known alternatives (for example `polars` wherever `pandas`
  is listed). Generic names such as `schema` or `parse` were left out on purpose, so a made-up
  local module can't earn target credit.

## Results

All benchmark evals use no system prompt, 5 samples per task and up to 16,384 new tokens.
- **Library use** is the share of responses that import and use a third-party library.
- **… high value** is the same, on the 52 test tasks where a library saves 15 or more logical
  lines.
- **Lines** is the average number of logical lines of code per response.
- One standard error on LibReuseBench test library use is about 2.3 points.

**Each epoch, plain prompt:**

| | test library use | … high value | test pass@1 | lines | control library use | control pass@1 | HumanEval+ | LiveCodeBench |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (before training) | 15.7% | 39.6% | 75.5% | 25.6 | 0.0% | 80.5% | 89.6% | 66.9% |
| import, epoch 1 | 31.7% | 64.2% | 78.1% | 22.9 | 0.2% | 83.6% | 89.8% | 69.4% |
| import, epoch 2 | **37.3%** | 71.5% | 79.2% | 22.2 | 0.0% | 81.9% | 88.7% | 69.2% |
| import, epoch 3 | 33.0% | 65.4% | 77.3% | 23.3 | 0.2% | 81.9% | 89.8% | 70.1% |
| import, epoch 4 | 35.4% | 66.9% | 76.2% | 23.9 | 1.9% | 80.5% | 88.3% | 69.9% |
| import, epoch 5 | 30.2% | 58.1% | 78.1% | 24.4 | 2.1% | 81.7% | 88.9% | 69.9% |
| import, epoch 6 | 28.0% | 56.9% | 78.0% | 24.7 | 1.4% | 83.1% | 89.1% | 70.3% |
| import, epoch 7 | 28.0% | 58.1% | 78.9% | 24.9 | 1.9% | 80.2% | 90.6% | 70.1% |
| import, epoch 8 | 25.9% | 52.3% | 78.6% | 24.7 | 2.6% | 80.5% | 89.5% | 67.1% |
| usage, epoch 1 | 29.0% | 60.8% | 79.8% | 22.5 | 0.0% | 80.7% | 89.4% | 68.3% |
| usage, epoch 2 | 32.8% | 63.5% | 78.2% | 23.1 | 0.2% | 82.4% | 90.7% | 69.0% |
| usage, epoch 3 | **37.5%** | 71.9% | 75.0% | 23.1 | 0.5% | 79.0% | 90.1% | 69.7% |
| usage, epoch 4 | 31.8% | 64.2% | 74.4% | 22.9 | 3.8% | 76.4% | 88.5% | 69.4% |
| usage, epoch 5 | 27.7% | 56.9% | 74.3% | 23.5 | 4.3% | 71.0% | 86.8% | 67.1% |
| usage, epoch 6 | 28.1% | 59.2% | 73.4% | 23.6 | 4.5% | 71.9% | 89.1% | 72.0% |
| usage, epoch 7 | 26.6% | 56.2% | 73.6% | 23.9 | 4.3% | 75.2% | 87.6% | 68.0% |
| usage, epoch 8 | 26.3% | 54.6% | 74.3% | 23.7 | 4.3% | 73.6% | 88.8% | 68.5% |
| target, epoch 1 | 27.2% | 56.2% | 79.3% | 23.5 | 0.0% | 84.3% | 89.5% | 67.8% |
| target, epoch 2 | 35.4% | 67.3% | 79.8% | 22.1 | 0.5% | 85.7% | 90.0% | 68.3% |
| target, epoch 3 | 37.5% | 70.8% | 78.0% | 22.2 | 0.2% | 82.1% | 88.7% | 70.3% |
| target, epoch 4 | **37.6%** | 69.6% | 78.0% | 22.3 | 0.2% | 82.6% | 87.8% | 69.4% |
| target, epoch 5 | 34.2% | 65.0% | 79.3% | 22.8 | 1.4% | 78.3% | 87.0% | 72.4% |
| target, epoch 6 | 31.8% | 60.4% | 74.7% | 23.5 | 2.9% | 73.3% | 86.5% | 69.7% |
| target, epoch 7 | 29.2% | 56.9% | 76.4% | 23.8 | 5.2% | 73.8% | 87.3% | 68.0% |
| target, epoch 8 | 28.8% | 55.8% | 76.9% | 23.8 | 6.0% | 75.0% | 87.0% | 69.7% |

**The training data's held-out splits**, scored during training (5 answers per prompt). "Any
library" is the share of answers using a third-party library. "Target library" is the share using
one on the task's target list.

| | val, any library | val, target library | test, any library | test, target library |
|---|---|---|---|---|
| before training (step 0) | 36-37% | 30-31% | 28-29% | 24-25% |
| import, end (step 296) | 89% | 52% | 91% | 49% |
| usage, end | 92% | 77% | 89% | 75% |
| target, end | 90% | 86% | 85% | 81% |

**Prompt engineering, with and without training.** Prompt suffixes were only scored on epochs
1-3.

| | prompt | test library use | test pass@1 | control library use | HumanEval+ | LiveCodeBench |
|---|---|---|---|---|---|---|
| Qwen3-8B | plain | 15.7% | 75.5% | 0.0% | 89.6% | 66.9% |
| Qwen3-8B | `always` | 85.1% | 74.0% | 10.0% | 74.8% | 63.2% |
| Qwen3-8B | `appropriate` | 46.2% | 78.1% | 0.0% | 90.9% | 65.5% |
| import, epoch 1-3 | `appropriate` | 65.1-71.0% | 76.0-79.7% | 0.2-2.1% | 88.2-89.4% | 69.7-71.5% |
| usage, epoch 1-3 | `appropriate` | 64.1-71.6% | 76.0-78.1% | 0.2-3.1% | 87.8-89.6% | 65.5-69.9% |
| target, epoch 1-3 | `appropriate` | 63.4-74.6% | 78.1-78.6% | 1.2-3.1% | 86.7-88.8% | 68.5-70.6% |
| any run, epoch 1-3 | `always` | 88.0-96.8% | 71.0-78.2% | 9.3-32.9% | 68.5-71.8% | 65.7-70.6% |

## Takeaways

- **Library use on the benchmark peaks early, then falls.** All three rewards reach the same peak
  on LibReuseBench test, about 37.5% against 15.7% before training: import at epoch 2, usage at
  epoch 3 and target at epochs 3-4. By epoch 8 all three are down to 26-29%. Over the same epochs,
  library use on the training data's held-out splits kept rising, to 85-92%. So the later epochs
  fit the training prompts, not the benchmark. This is why the next runs use 4 epochs.
- **Late epochs start using libraries where none are needed.** Up to epoch 3, control library use
  stays at 0.5% or less. By epoch 8 it is 2.6% (import), 4.3% (usage) and 6.0% (target). For
  usage and target, control pass@1 also falls, from about 80% to 71-75%.
- **The rewards differ in which libraries they teach, not how often.** By the end, every run uses
  a library in about 90% of held-out answers. But only the target reward keeps the choice on the
  target list: 86% of val answers use a target library, against 77% for usage and 52% for import.
  In the import run's final-epoch rollouts, 41% of all answers use only libraries off the list. On
  LibReuseBench, the three rewards peak at the same level.
- **The import-only reward did not produce unused imports.** Its final-epoch rollouts imported a
  library in 94.0% of answers and used one in 92.8%. So the usage credit isn't needed to stop
  that hack, at least at this scale.
- **General coding ability holds.** HumanEval+ stays at 86.5-90.7% (base 89.6%), and
  LiveCodeBench at 67.1-72.4% (base 66.9%). LibReuseBench pass@1 rises a little early on (to
  78-80%), then drops back for the usage run (73-74% from epoch 4). Answers get shorter, from 25.6
  logical lines to 22-25.
- **Asking for libraries "when appropriate" beats any single checkpoint.** The `appropriate`
  suffix alone gives 46.2% library use, with 0% on control and HumanEval+ unchanged. `always`
  reaches 85.1%, but uses a library on 10% of control tasks and drops HumanEval+ to 74.8%.
- **Training and the prompt add up.** Trained checkpoints with the `appropriate` suffix reach
  63-75% library use, with control use at 3.1% or less and HumanEval+ at 87-90%. The best is the
  target run at epoch 3 (74.6%). With `always`, training pushes control use up to 33%.
- **Shaping was barely needed.** Format and syntax failures were 5-8% of rollouts in epoch 1 and
  1-2% by epoch 8. The share of groups where all 16 rollouts scored the same, and so taught
  nothing, rose from about 45% to 54-70% (import highest). The shaping ablations in iteration 04
  test whether the format and syntax weights matter at all.
- **One-step KL spikes.** All three runs had spikes, for example at usage step 66, target step 262
  and import step 279 (KL 2.5e7). The ones checked (usage step 66, import step 24) each came
  from a single token whose trainer and vLLM probabilities differed by a ratio of about 1e-12. Training recovered at once each time. The
  import run's spikes grew later in training. Masking such tokens is an option for future runs.
- **Caveats.**
  - Each reward was run with one seed. Neighbouring epochs of one run differ by up to 6 points,
    so differences of a few points between rewards are within noise.
  - The peaks above are picked on the test split. The next iterations report a fixed final epoch
    instead.
  - Prompt suffixes were only scored on epochs 1-3.
