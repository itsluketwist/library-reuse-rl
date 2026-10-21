# Iteration 02: new training data, and a new benchmark to measure it

Two GRPO runs on Qwen3-8B with the same recipe. They differ only in the training data and the number
of epochs. Each run's training metrics are in `{run_id}/grpo/` and each epoch's evals in
`{run_id}/epoch{n}/`. Plots for either run come from `notebooks/01_grpo_run_analysis.ipynb` (set
`RUN_ID`).

The iteration also has the first results for a hosted frontier model (`deepseek-flash`) and for
the control split. Results for models that weren't trained here are in `baselines/{model}/`. They
were moved there from `output/baselines/`, where the eval writes them. The eval still finds them
here, so a rerun skips them unless `--update` is set.

## What changed since iteration 01

Iteration 01 found two problems: library use barely transferred to LibReuseBench (1.9% to about
4%), and the benchmark-style `script` prompts gave GRPO almost nothing to learn from. The fixes
came in this order:

1. **Training data v3** made the script prompts match the old benchmark more closely. It still
   gave little signal (run `e650`: 4.6% library use on LibReuseBench v1).
2. **So the benchmark was rebuilt** as LibReuseBench v2.
3. **Training data v4** was then matched to the new benchmark (run `8672`).

**The benchmark (LibReuseBench v1 to v2).** v1's tasks were stdin/stdout programs, and most gave no
reason to use a library: the median task saved 1-2 lines with one. Rewording v1's tasks as
functions didn't change library use (within 2 points), so the format was not the bottleneck. v2 is
a rebuild:

- **Tasks:** function tasks (`Write a function name(args) that ... For example, call returns
  result.`), reworded from BigCodeBench seeds. The generator was asked for *contested* tasks, ones
  a developer could reasonably solve either with a library or by hand, and to skip seeds that
  aren't.
- **Checks:** each task has a library solution and a standard-library solution. The expected
  results come from running the library one, and the standard-library one must give the same
  results.
- **Reuse value:** the logical lines the library saves. It sets each task's tier: low (under 5),
  medium (5-14) or high (15 or more).
- **Status:** `libreusebench_v2_full` is the 233 tasks an agent review kept: 63 low, 116 medium
  and 54 high. It still needs manual review and a split into test and dev, so its numbers may
  change.

**The training data (`library_usage` v2 to v3 to v4).** See `data/build/library/HISTORY.md`.

- **v3 (run `e650`):** the Claude-written script row became three stdin/stdout exercises in the
  old benchmark's format, at increasing detail (`outline`, `precise`, `benchmark`). It also added
  a project-framed repo row with no project shown.
- **v4 (run `8672`):** because the benchmark moved to functions, the four script rows became three
  requests for one named function: `name` (the name only), `signature` (its parameters, and what it
  takes and returns) and `example` (the signature and one call with its result, as in the
  benchmark). Each request was written to leave the choice between a library and the standard
  library open. Repo and simple rows are unchanged from v3.

## Training choices

| | `qwen3-8b-e650` | `qwen3-8b-8672` |
|---|---|---|
| training data | `library_usage` v3, 1,362 prompts | `library_usage` v4, 1,209 prompts |
| epochs (steps) | 3 (126) | 6 (222) |
| training time | about 14 hours | about 23 hours (requeued and resumed) |

Both runs share everything else:

- **model**: Qwen3-8B, full fine-tuning, thinking mode on
- **reward**: `lu-v1`, scored by lure 0.1.2. It gives 0.25 for correct format, 0.25 for code that
  parses, 0.5 for importing a third-party library and 1.0 for using it (max 2.0).
- **GRPO** (`config/training.yaml`): 32 prompts x 8 rollouts per step. The learning rate is 3e-6,
  with cosine decay to 10%. The KL penalty is 0.01, and the limit is 8,192 new tokens at
  temperature 1.0. No system prompt.
- **compute**: 4 GPUs (Isambard), seed 42

## Training data

| version | split | prompts | tasks | repo / simple / script or function | written by Claude | benchmark-aligned |
|---|---|---|---|---|---|---|
| v3 | train | 1,362 | 175 | 448 / 317 / 597 | 743 (55%) | 820 (60%) |
| v3 | val | 166 | 129 | 59 / 31 / 76 | 85 (51%) | 96 (58%) |
| v3 | test | 167 | 19 | 55 / 36 / 76 | 91 (54%) | 99 (59%) |
| v4 | train | 1,209 | 175 | 452 / 309 / 448 | 742 (61%) | 730 (60%) |
| v4 | val | 135 | 106 | 55 / 39 / 41 | 77 (57%) | 77 (57%) |
| v4 | test | 145 | 19 | 55 / 36 / 54 | 88 (61%) | 88 (61%) |

- **v3** files: commit `712828d` (2026-10-08). To restore: `git checkout 712828d --
  data/train/library_usage`. SHA-256: train `159aa8b36a60…`, val `95cae9daae75…`, test
  `fc2ae1dd20be…`.
- **v4** files: commit `b009847` (2026-10-17). SHA-256: train `51bc8519ffa0…`, val `2fc26b3e5d6d…`,
  test `3f64d0bacd1a…`.
- Both versions hold out the same 19 test tasks, which training never sees. Val holds out other
  wordings of training tasks.

## Results

**LibReuseBench v2** (`libreusebench_v2_full`, 233 tasks, before manual review). Library use is the
share of responses that import and use a third-party library. Evals use no system prompt and up to
16,384 new tokens. The base model, `4796` and `e650` drew 25 samples per task, and `8672` drew 5;
pass@5 is estimated the same way for both.

| | pass@1 | pass@5 | library use | … low value | … medium value | … high value |
|---|---|---|---|---|---|---|
| Qwen3-8B (before training) | 76.3% | 92.5% | 15.6% | 4.9% | 9.7% | 40.8% |
| `4796` (iteration 01, v2 data), epoch 5 | 79.8% | 94.3% | 26.4% | 10.9% | 21.4% | 55.0% |
| `e650` (v3 data), epoch 3 | 79.7% | 94.2% | 27.6% | 11.9% | 22.9% | 56.2% |
| `8672` (v4 data), epoch 2 | 80.9% | 94.8% | 32.7% | 14.9% | 28.1% | 63.3% |
| `8672` (v4 data), epoch 6 | 77.9% | 93.6% | 34.2% | 13.3% | 30.5% | 66.7% |

**LibReuseBench v1** (old stdin/stdout benchmark, 5 samples, 8,192 tokens; `8672` was not scored on
it):

| | test pass@1 | test pass@5 | test library use | control pass@1 | control library use |
|---|---|---|---|---|---|
| Qwen3-8B (before training) | 45.7% | 58.7% | 1.9% | 37.7% | 0.0% |
| `e650`, epoch 3 | 55.3% | 69.8% | 4.6% | 46.8% | 0.0% |

**Held-out `library_usage` test split.** Each run's numbers are on its own data version, so they
are not directly comparable. There is no base-model run on v4 yet.

- **`e650`**, base to epoch 3: library use 22.8% to 42.0%. By mode: repo 38.2% to 69.5%, simple
  38.9% to 53.9%, script 3.9% to 16.6%.
- **`8672`**, epoch 1 to epoch 6: library use 44.1% to 57.2%. By mode: repo 62.5% to 82.2%, simple
  50.6% to 61.1%, function 21.1% to 29.3%. By function level: `name` 38% to 53%, `signature` 13% to
  17%, `example` 11% to 16%.
- **On the repo and simple test prompts both versions share**, `8672` (epoch 6) beats `e650` (epoch
  3): repo 82.2% against 69.5%, simple 61.1% against 53.9%. `e650`'s evals used 8,192 tokens, not
  16,384, so treat this as approximate.

**A frontier model on LibReuseBench v2** (`deepseek-flash`, scored on the provisional test split,
`libreusebench_v2_test`). The table compares it with the open models on the 233 tasks it shares
with `libreusebench_v2_full`, matched by seed id. All use 5 samples and 16,384 tokens.

| | pass@1 | library use | … high value | … medium value | … low value | cut off |
|---|---|---|---|---|---|---|
| `deepseek-flash` | 96.8% | 3.4% | 7.0% | 2.9% | 1.3% | 2.5% |
| Qwen3-4B | 71.2% | 18.9% | 42.3% | 14.1% | 7.8% | 6.6% |
| Qwen3-8B | 76.3% | 15.6% | 40.8% | 9.7% | 4.9% | 6.0% |
| Qwen3-14B | 80.5% | 19.0% | 48.7% | 11.6% | 7.2% | 2.6% |
| Qwen3.5-9B | 55.5% | 10.1% | 26.3% | 6.4% | 3.1% | 2.8% |
| domyn-small | 54.2% | 21.4% | 50.3% | 15.0% | 8.4% | 0.0% |
| `8672` (v4 data), epoch 6 | 77.9% | 34.2% | 66.7% | 30.5% | 13.3% | 1.4% |

Its first run stopped partway when the API account ran out of credit. The finished result comes
from retrying only the failed requests (see `src/eval/evaluate.py`), with no request left failed.

**LibReuseBench v2 control** (`libreusebench_v2_control`: 96 provisional control tasks, after the
stricter agent review and before manual review; 5 samples, 16,384 tokens):

| | pass@1 | pass@5 | library use |
|---|---|---|---|
| `deepseek-flash` | 98.5% | 100.0% | 0.0% |
| Qwen3-4B | 79.4% | 88.5% | 0.0% |
| Qwen3-8B | 81.0% | 89.6% | 0.0% |
| Qwen3-14B | 83.8% | 92.7% | 0.0% |
| Qwen3.5-9B | 76.2% | 95.8% | 0.0% |
| domyn-small | 66.7% | 86.5% | 0.0% |
| `4796` (v2 data), epoch 5 | 83.1% | 89.6% | 0.2% |
| `e650` (v3 data), epoch 3 | 85.0% | 93.8% | 0.0% |
| `8672` (v4 data), epoch 6 | 80.2% | 89.6% | 0.2% |

The only library use in all 4,320 control responses is `cryptography`, twice, on one PBKDF2
password-hashing task (LRB2/C042). `4796`'s control result stays in `output/01/` with the rest of
that run.

## Takeaways

- **The v4 data transfers best.** On LibReuseBench v2, library use rose from 15.6% to 34.2% with
  v4, against 27.6% with v3 and 26.4% with v2. Each of `8672`'s last three epochs beats every
  earlier checkpoint. Most of the gain is on the medium- and high-value tasks, where a library
  saves the most code.
- **It is a change of choice, not fewer truncated responses.** Before training, only 6% of
  responses were cut off at the token limit, and 1.4-2.4% after training (`8672` at epoch 6:
  1.4%). So finishing more often adds at most a point or two of library use, far less than the
  19-point gain. Counting only responses that contain code, `8672` uses a library 34.7% of the
  time, against 28.1% for `e650` and 16.5% for the base model. pass@1 rose only a little, to
  78-81%, about what fewer cut-offs would give, so the code is not better.
- **More library use, shorter code.** Average response length fell from 25.6 logical lines
  (base) to 23.1 (`8672`, epoch 6). On the same task, a response that uses a library is 10-13
  lines shorter than one written by hand, which accounts for most of the drop.
- **v3's staged script rows learned, but didn't transfer.** On its own data, the `outline` rows went
  from 2.9% to 17.1% library use. On the stdin/stdout benchmark they were meant to match, library
  use only reached 4.6%, no better than iteration 01. They were also long: 9-13% of training
  rollouts were cut off, against 2-6% for `8672`.
- **The more a request spells out, the less often a library is used.** This holds in both versions:
  the v3 script stages and the v4 function levels (`name` 53%, `signature` 17%, `example` 16% at
  epoch 6). The rows most like the benchmark move the least, so the transfer seems to come from
  the plainer wordings and the function format.
- **`8672` trained cleanly and was still improving.** Rollouts using a library rose from 37% to 61%
  over the 6 epochs, and there were no KL spikes, unlike `4796` and `e650`. On the benchmark,
  though, most of the gain had arrived by epoch 2 (32.7%), and pass@1 was lowest at epoch 6.
- **A frontier model writes almost everything by hand.** `deepseek-flash` passes 96.8% of the test
  tasks, but uses a library on only 3.4% of responses, and 7.0% even where a library saves 15 or
  more lines. The open base models use one on 10-21%, and the trained runs on 26-34%. Being the
  best coder doesn't make a model reuse more. Its code is longer instead: 30.6 logical lines per
  response, against 25.6 for Qwen3-8B and 23.1 for `8672`.
- **No model over-reaches on control.** Library use on the control tasks is 0.0-0.2% for every
  model, including `8672`, which uses a library on 34% of test responses. Training raised library
  use where a library helps without making the models import one regardless. The control split
  doesn't separate the models, though: even the stricter "plausible but unnecessary" review left
  tasks that tempt none of them.
- **Caveats.**
  - Each configuration was run with one seed. Epochs of the same run differ by 3-4 points on the
    benchmark, which is far more than sampling noise (about 0.5 points at 5 samples per task).
  - `libreusebench_v2_full` has not been manually reviewed or split into test and dev.
  - Library use also rose on low-value tasks (4.9% to 13.3%), where a library saves little.
  - Imports of made-up modules (`from your_module import ...`) in side snippets still earn library
    credit: 1-3% of held-out responses, for both runs.
