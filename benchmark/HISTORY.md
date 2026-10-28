# LibReuseBench: how it was built

LibReuseBench has one version: the 216 test and 84 control tasks in `libreusebench/data/`, built
from the tasks both human reviewers approved. This file records the attempts that led to it, and
what each showed. Library use is the share
of responses that use a library; pass@1 is the share that pass the tests. "Base" is Qwen3-8B
before training; "4796" and "e650" are its GRPO checkpoints (`output/01/`, `output/02/`).

## An earlier design: stdin/stdout programs (archived in `benchmark/v1/`)

- BigCodeBench seeds reworded as library-agnostic stdin/stdout programs: 179 test and 79 control
  tasks.
- **Problem:** most tasks gave no reason to use a library. On the median task a library saved
  only 1-2 lines, and models used one in 2-5% of responses.
- **Function pilot** (2026-10-10, `benchmark/v1/pipeline/function_pilot/`): the same tasks
  reworded as requests for one named function. pass@1 rose 21-25 points, but library use did not
  change (within 2 points), so the format was not the bottleneck.

## The current design: function tasks, with their reuse value measured

A rebuild (`benchmark/pipeline/`): one model call per BigCodeBench seed writes a short function
task, a library solution and a standard-library solution. Both are run, and the library one sets
the expected results. The lines the library saves (reuse value) set each task's tier: low (under
5), medium (5-14) or high (15 or more). An agent review then keeps or rejects each task.

Five pilots, each changing the request that writes the tasks:

| Pilot | Seeds | Tasks kept | Tiers (low / medium / high) | What changed | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 60 | 47 + 10 control | 0 / 0 / 47 | asked for tasks where a library saves a lot of code | texts long (median 1,120 characters); 38-81% of responses cut off before any code; base pass@1 13% |
| 2 | 20 (from pilot 1) | 11 | 3 / 2 / 6 | short, simple rewordings of the seed, no target for lines saved | reuse values spread out; library choice all-or-nothing per task |
| 3 | 40 (pilot 2's + 20 new) | 13 | 3 / 6 / 4 | asked for *contested* tasks (as reasonable by hand as with a library); a small change to a seed allowed; other seeds skipped | base library use 34%; trained 43-45% |
| 4 | 50 new | 18 | 14 / 4 / 0 | a task needing a numerical method or a parser by hand counts as too hard; don't fix a convention the library's usual call doesn't follow | almost no high-value tasks; not evaluated |
| 5 | 100 (pilot 4's + 50 new) | 26 | 8 / 14 / 4 | the request explains the benchmark's purpose: models write too many custom implementations, so parsing, cryptography and utility tasks are wanted | base library use 13%, pass@1 80%; trained 20% |

**Full run** (2026-10-15, pilot 5's request, every seed):

- **Reuse track:** 813 seeds, 252 tasks kept (75 low, 121 medium, 56 high). The agent review
  kept 233 of them, which form `libreusebench_v2_full` (`benchmark/artefacts/full_run/`).
  - Base: 15.6% library use, 76.3% pass@1.
  - Training: up to 34.2% library use (run 8672), mostly on high-value tasks.
  - After one fix round (2026-10-19), 234 are kept and 18 rejected.
- **Control track:** 327 standard-library seeds, 181 tasks kept.
  - The first review kept almost all of them (176): it only asked whether a library would
    help, which the tasks were written to rule out.
  - So the control review was made stricter (2026-10-19). A control task is only useful if a
    library is *plausible but unnecessary*: the job is one a developer might use a library for
    (a format such as CSV or JSON, hashing, dates), but the standard library does it directly.
    A task so plain that nobody would think of a library is rejected, since every model writes
    it by hand.
  - The stricter review kept 96 and rejected 84. One task (BigCodeBench/644) has no review,
    because the model refuses the rubric call itself.
  - A blind solve the model refuses (it declines some hashing tasks) is now recorded as giving
    no solution, and the task is judged on everything else.

**Provisional splits** (2026-10-19, `05_build.py --provisional`, in
`benchmark/artefacts/provisional/`):

- **Test:** 234 tasks (63 low, 117 medium, 54 high). These are the 233 `v2_full` tasks word for
  word, plus BigCodeBench/845; adding that task renumbered the split.
- **Control:** 96 tasks, every one the stricter review kept. A cap of one task per category and
  15 per domain was tried (it gave 71), and dropped: it saved little, and an uncapped split is
  simpler to explain.

The pilots' task files were removed on 2026-10-19 (they are in git history, up to commit
`4e67fc6`); their eval results are still in `output/` as `libreusebench_v2_pilot*`. A
deduplication stage was built and tried (2026-10-17), then dropped: tasks need not be unique, and
using every BigCodeBench seed is simpler to defend than a rule for what counts as a duplicate. For
the same reason the splits follow BigCodeBench (library seeds give `test`, standard-library seeds
give `control`), with no dev split, no low-value split and no caps; the reuse tier is a field of
each test task instead.

**Human review and the final build** (2026-10-27, `04_review.ipynb`, then `05_build.py`):

- Both reviewers read every task the agent review kept, without seeing each other's verdicts. A
  task ships only if neither rejects it. `docs/REPORT.md` has the full counts.
- **Test:** of 234 tasks, both kept 216 (one with an edited example, written out to full
  precision) and both rejected 8; they disagreed on 10, which are rejected. Raw agreement 95.7%,
  Cohen's kappa 0.59.
- **Control:** of 96 tasks, both kept 84 and both rejected 2; they disagreed on 10. Raw agreement
  89.6%, Cohen's kappa 0.25. Kappa is low beside the raw agreement because nearly every task is
  kept, so even a few disagreements weigh heavily.
- **The splits:** `test` has 216 tasks (58 low, 106 medium, 52 high; median 8 lines saved, from
  -6 to 60), `control` has 84. Task ids are `LRB/T000`-`LRB/T215` and `LRB/C000`-`LRB/C083`,
  assigned after a seeded shuffle; they differ from the ids of the builds above, so results on
  those match these tasks by seed id (`source.seed_id`) only. BigCodeBench/845 has no results
  from the full run, which came before its fix.
- **Checks:** every reference solution passes through the evaluation harness, and a wrong answer
  fails on every task (`06_audit.py --check-golds`). The eval keys are `libreusebench_test` and
  `libreusebench_control`; results under the earlier keys stay in `output/` (the archived design's
  as `libreusebench_v1_test` and `libreusebench_v1_control`).

## Still to do

- Release: the workflow (`.github/workflows/release-libreusebench.yaml`) uploads the dataset to
  Hugging Face, removing the earlier design's `dev.jsonl` there, and publishes the package
  (`libreusebench` 0.2.0) to PyPI.
