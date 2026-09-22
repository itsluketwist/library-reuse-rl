# LibReuseBench — construction

How the [LibReuseBench](HF_README.md) benchmark is built, and how the pieces fit together.
This is the internal/reviewer-facing doc; `PYPI_README.md` / `HF_README.md` are the release cards.

## Why this benchmark exists

This project studies whether GRPO can teach LLMs to reuse libraries instead of reimplementing
functionality. Training uses the amplify-derived splits in `train_dataset/`, verified
*deterministically* (prose allowlist match, AST import classification) with no code execution — so
they give no independent correctness signal. The legacy Skywork `implement_tested` data executes
real tests but rarely needs a library.

LibReuseBench is the missing **external** signal: real executable test cases **and** a genuine
temptation to reach for a library — while never telling the model which one to use. The harness
reports both correctness (test pass rate) and library reuse on the same problems.

A third **control** track (split `control`, task ids `LRB/C000...`) is the complementary case:
problems seeded from BigCodeBench solutions that used *no* external library, so reaching for one
isn't expected to help. It exists so the benchmark can show library reuse is rewarded *where it
helps*, not wherever it's used — without it, a model that reaches for a library unconditionally
would look indistinguishable from one that reaches for one appropriately.

## Two design decisions

1. **Pure stdin/stdout, no scaffolding.** Each problem is a plain spec: read stdin, print stdout.
   No function signature, no importable test harness — any scaffolding leaks a library hint. The
   model has complete internal freedom, and the tests (just `(stdin, expected_stdout)` pairs) are
   library-agnostic for free.
2. **Keep only dual-solvable problems.** A problem that *requires* a library leaves nothing to
   measure. The problems that test reuse are solvable **both** with an external library and with the
   standard library. A stdlib-only reference solution passing the tests is what proves it.

## Layout

```
benchmark/
  libreusebench/       # the installable framework package (pip install libreusebench)
    libraries.py        # extract code/imports, import->pip map, stdlib/external, prompt scan
    environment.py      # run stdin/stdout programs; venv that installs response libraries
    load.py             # load the bundled splits
    evaluate.py         # the harness: scan -> install -> run -> aggregate
    cli.py              # `libreuseeval`
    data/               # bundled, committed splits (test/dev/control.jsonl) + CANARY.txt
  pipeline/             # dataset construction (build-time; calls into libreusebench + repo src)
    01_fetch_bcb.py … 08_audit.py (06_review_test/06_review_control are notebooks)
    build.py            # stage paths, BCB analysis, hard-exclude buckets
    generation.py       # LLM prompts + parsers (Anthropic)
  docs/                 # CONSTRUCTION.md, PYPI_README.md, HF_README.md, REPORT.md
  pyproject.toml        # standalone package spec (+ LICENSE)
```

`libreusebench` is the self-contained core (depends only on `llm_cgr`), shared by the build and the
harness and publishable on its own. The pipeline is build-time only and imports `libreusebench`
plus a few repo `src` utilities. Tests live in the main repo under `tests/test_benchmark/`.

## Build pipeline

Run in order from the repo root (no `PYTHONPATH` needed — the scripts import sibling build helpers
and the installed `libreusebench` package). Each stage reads the previous stage's artefact under
`benchmark/artefacts/` (gitignored, so runs are resumable). The LLM stages (02, 03, 04, 05) are
resumable — only records that have not yet succeeded are retried, so a rerun (optionally with a
stronger model) picks up just the failures — and take `--limit N` / `--model M`. Stage 05 also
takes `--attempts N` (best-of-N): apparent failures there are dominated by sampling variance, so a
task is kept as soon as any sample reproduces the outputs. Model defaults: the judge uses Claude
Haiku (`EVAL_DATASET_JUDGE_MODEL`); rewrite, gold generation, and validation use a stronger Claude
model (`EVAL_DATASET_MODEL`, default Sonnet). The shipped set was built by sampling each of stages
04/05 up to three times with Sonnet, then up to three more with Opus, recording the resolved model
against every gold solution it produced.

```
python benchmark/pipeline/01_fetch_bcb.py    # -> artefacts/01_bcb_{extlibs,stdlibs}.jsonl
python benchmark/pipeline/02_filter.py       # -> artefacts/02_candidates.jsonl
python benchmark/pipeline/03_rewrite.py      # -> artefacts/03_rewritten.jsonl
python benchmark/pipeline/04_make_tests.py   # -> artefacts/04_tests.jsonl
python benchmark/pipeline/05_validate.py     # -> artefacts/05_validated.jsonl
# 06 review is an interactive notebook (06_review_test.ipynb) -> artefacts/06_reviews.jsonl
python benchmark/pipeline/07_build.py        # -> libreusebench/data/{test,dev}.jsonl  (bundled)
python benchmark/pipeline/08_audit.py        # -> docs/REPORT.md
```

Stages 02/03/04/07 also take `--track control`, running the same steps over the stdlib-only seed
pool instead (`01_bcb_stdlibs.jsonl`) to build the control track:

```
python benchmark/pipeline/02_filter.py --track control      # -> artefacts/02_candidates_control.jsonl
python benchmark/pipeline/03_rewrite.py --track control     # -> artefacts/03_rewritten_control.jsonl
python benchmark/pipeline/04_make_tests.py --track control  # -> artefacts/04_tests_control.jsonl
# 06 review: 06_review_control.ipynb -> artefacts/06_reviews_control.jsonl
python benchmark/pipeline/07_build.py --track control        # -> libreusebench/data/control.jsonl
```

The control track has no stage-05 equivalent - stage 04's minted stdlib gold + tests is already its
fully validated pool, since these problems don't need a library solution to prove anything. Stage
08 (audit) always reports both tracks in one `REPORT.md`, no `--track` flag needed.

- **01 fetch** — load BigCodeBench (`bigcode/bigcodebench`, pinned `v0.1.4`).
- **02 filter** — a minimal hard-exclude drops only the impossible (live-network, plot/GUI-only);
  everything else goes to a permissive LLM judge scoring use-case fit (deterministic,
  text-serialisable, and for the reuse track dual-solvable) as keep / maybe / drop, using
  structured outputs. For the control track the dual-solvable criterion is dropped - those
  problems have no library solution to be dual-solvable with.
- **03 rewrite** — the model rewrites the problem into a single self-contained, natural-language
  stdin/stdout task that names no library and no function. The original BigCodeBench task is shown
  only so the model understands the required behaviour; only the reworded prompt is carried forward
  (there is no adapter and BigCodeBench's own solution is not used downstream).
- **04 make_tests** — the model writes a standard-library-only gold solution plus diverse test
  inputs; we run that gold to mint each expected output, running every input twice and keeping only
  the reproducible ones (rejecting any gold that depends on randomness or the clock). This fixes
  the ground truth and already proves stdlib-solvability.
- **05 validate** — the prompt must name no library, and an independent solution that reaches for a
  third-party library (best-of-N samples) must reproduce the stage-04 outputs exactly. Two
  independent solutions agreeing cross-validates the ground truth and proves the problem is
  genuinely dual-solvable. The declared PyPI packages are installed into an isolated venv, run
  under a 30s cap (headroom for slow imports like sklearn/scipy), then fully uninstalled; each drop
  is recorded with a granular reason (output mismatch / runtime error / timeout / install failed)
  and the failing solution, so failures are diagnosable without re-running. **Reuse track only** -
  the control track has nothing to dual-validate, so stage 04's output ships as-is.
- **06 review** — a light human coherence check, not a deep audit. An interactive notebook
  (`06_review_test.ipynb`, or `06_review_control.ipynb` for the control track) shows each validated
  problem one at a time (prompt, sample tests, and its gold solution(s)); each author skims that the
  task makes sense and reads cleanly, and catches anything the automated gates cannot — a confusing
  prompt, an odd expected output, something that slipped through — then records keep / update (ship
  an edited prompt) / reject. Both authors independently review the whole pool - never a split of
  it, and never shown each other's verdict - and a problem only ships once both have weighed in and
  neither rejected it (reject wins on any disagreement). Verdicts persist to
  `artefacts/06_reviews.jsonl` (`06_reviews_control.jsonl` for the control track - the two tracks
  are reviewed and shipped independently; resumable per author, and each track's keep/update/reject
  summary is captured in the audit REPORT), and the build refuses to run until every problem in
  that track has a verdict from both authors.
- **07 build** — write the author-approved survivors straight into the package's `data/`. The reuse
  track holds out a small dev split (`test.jsonl` / `dev.jsonl`); the control track ships as a
  single split (`control.jsonl`), since there's no dual-solvability signal to hold a dev set out
  for.
- **08 audit** — write `docs/REPORT.md`: the drop funnel (incl. the review outcome) and coverage by
  category / seed library, for both tracks.

## Row schema

Each row carries a dataset-unique `task_id` (`LRB/T000...` test, `LRB/D000...` dev, `LRB/C000...`
control, all assigned after shuffling), the task and its tests, a `ground_truth` block with the
gold solution(s), and a `source` provenance block pointing back to BigCodeBench.

```json
{ "task_id": "LRB/T000", "category": "html-parsing",
  "prompt": "<library-agnostic spec: read stdin, print stdout>",
  "test_inputs": ["...", "..."], "expected_outputs": ["...", "..."],
  "ground_truth": {
    "gold_stdlib": "<stdlib reference solution>",
    "gold_extlib": "<third-party-library reference solution>",
    "gold_extlib_packages": ["pandas==3.0.5"] },
  "source": { "seed_id": "BigCodeBench/52", "seed_libs": ["pandas", "regex"] } }
```

A control row has no `gold_extlib*` keys and an empty `seed_libs` (its seed used no external
library):

```json
{ "task_id": "LRB/C000", "category": "text-processing",
  "prompt": "<library-agnostic spec: read stdin, print stdout>",
  "test_inputs": ["...", "..."], "expected_outputs": ["...", "..."],
  "ground_truth": { "gold_stdlib": "<stdlib reference solution>" },
  "source": { "seed_id": "BigCodeBench/107", "seed_libs": [] } }
```

## The harness (parse → install → run)

`libreusebench.evaluate_responses` (CLI: `libreuseeval responses.jsonl`) is the released evaluator, and
responses are keyed by `task_id` (`{task_id, response}` or `{task_id, responses: [...]}`). It first
**parses every response** and gathers the union of third-party packages they import, mapping each
import name to its pip distribution (`sklearn` → `scikit-learn`, via `IMPORT_TO_PIP`). It then asks
for confirmation before it **creates the venv, installs the packages, and runs** each response
against its task's tests (`--unsafe-installs` skips the interactive checkpoint and installs
automatically), aggregating **pass@1 / pass@k**, overall **per-problem library-reuse
statistics**, and radon's average **logical lines of code** and **maintainability index** (over all
responses, and over passing ones only). Point `--env-dir` at an isolated location. The saved result records the environment
directory, requested packages, originating import names, and the resolved version of every
installed distribution.

For control and reproducibility the environment can instead be pinned with `--env env.txt` (one pip
package per line), which skips both collection and confirmation. If confirmation is declined or a
package fails to install, the harness writes that editable `env.txt` for correction —
remove unwanted or un-installable libraries, fix names/versions, and re-run with `--env`.

The main repo's eval pipeline invokes this same complete evaluator, passing `confirm=None` to install
automatically since it runs unattended, and uses a separate cached environment for each evaluation
output.
