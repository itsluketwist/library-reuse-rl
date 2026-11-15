# library_usage — construction

How the `library_usage` dataset is built. This is the internal/reviewer-facing doc; see
[../../../README.md](../../../README.md) for how the whole `data/` pipeline fits together,
[../HISTORY.md](../HISTORY.md) for what each dataset version changed and why, and
`PROMPT_COMPARISON.md` for real sample output.

## The idea

Each row asks for a small Python feature whose natural solution reaches for a mature library,
but which a model could also be tempted to hand-roll. The reward needs no execution: it just
extracts the response's imports, subtracts the standard library and any local project modules,
and checks what's left. So the whole dataset is built from four small, hand-authored inputs plus
one offline LLM pass - no scraping, no seed dataset.

## The four inputs

- **`taxonomy.yaml`** - 66 categories. Each has `seed_libraries` (the task's target libraries,
  which become each row's `target_libs`), a `diy_temptation` (what hand-rolling it looks like),
  and `repo_hint` (which of the two repos below its tasks get framed in). 26 categories are
  `benchmark_aligned: true` - their skills were chosen by clustering the 129 test problems of
  LibReuseBench's archived stdin/stdout design (`benchmark/v1/`) by the library call doing the
  real work, so training deliberately overlaps what that design measured. How much they overlap
  the released benchmark, built later from BigCodeBench, has not been measured.
- **`tasks.yaml`** - 194 concrete tasks, a `{number: text}` mapping per category (e.g.
  `datetime: {1: "Parse an ISO-8601 timestamp...", 2: ...}`). A task's `seed_task` id is
  `"<category_id>.<number>"`, e.g. `"datetime.1"` - the thing every wording of that task traces
  back to.
- **`templates.yaml`** - 2 fixed wordings each for `repo` (framed inside a real project) and
  `simple` (a short, bare ask), and for `script` (stdin/stdout, used only by the script pilot
  since v4). One of the 2 is picked per task per mode, deterministically (seeded by `seed_task`
  and mode), rather than emitting both. The `function` mode has no template, only the claude
  wordings below.
- **`repos/pytoolbox/`** (authored here) and the vendored `python-api` (shared with the amplify
  pipeline) - two small real Python projects `repo`-mode tasks get framed inside. Each category
  picks one via `repo_hint`.

## The LLM layer (`augment.py`)

Alongside each mode's template, claude-generated wordings are cached offline - one per task for
`simple` and `repo`, three staged ones for `script`, and three staged `function` requests:

- **`simple`** - a short, natural chat-style request ("Can you write a function that...?"),
  optionally naming a plausible signature in plain built-in types only (never a type that would
  hint at a library). If two attempts both leak a library name, it retries without asking for a
  signature at all - some tasks' library name is also the natural English verb for the task
  (e.g. `slugify`), and dropping the signature ask is what actually fixes that, not rewording.
- **`script`** (dataset v3; since v4 only in the script pilot) - one stdin/stdout exercise per
  task, in LibReuseBench's old template (an
  opening "Write a program that reads ... from standard input", `Input format:` and `Output
  format:` sections, and exactly one `Example:`), written at three levels of detail: `outline`
  (the behaviour and the formats, briefly), `precise` (plus output rules such as rounding, and how
  to treat empty or invalid input) and `benchmark` (as long and formal as the benchmark). Each row
  records its stage as `script_stage`. The generation prompt deliberately mirrors the one that
  design was built with (`benchmark/v1/pipeline/03_rewrite.py`): the model first writes a
  short private solution that uses a suitable third-party library (kept in the cache as
  `solution`), then describes that program's behaviour - what it reads and prints - without naming
  the library, describing an algorithm, or defining standard concepts (it names them instead:
  "EAN-13", "population variance"). So, as in the benchmark, a library's natural output is a
  correct answer, but the task can be solved with or without one, leaving the model free to
  choose. Two statements from that design's dev split (`benchmark/v1/data/dev.jsonl`, never
  test) are style examples. A task that can't be
  a deterministic text-in, text-out program (binary output, network, randomness, spawning
  processes, ...) is marked `fits_stdio: false` and keeps only its template script row. Every
  statement passes a deterministic check (`check_script_stage`: template shape, no library names
  or talk of libraries, no code, length), with a failed attempt retried with its problems fed
  back.
  How it got here (worth knowing before changing it): the v2 wording pre-digested inputs until no
  library helped (an audio clip became a list of integers) and gave almost no library signal. A
  first v3 pilot with many more rules - keep the "hand-rolled part", choose the variant that's the
  most work by hand - produced long, exacting statements; models ran out of tokens on them and,
  when an exact output wasn't what a library produces, hand-rolled. Library use fell as the
  statements got more specific, so the simpler, behaviour-only prompt replaced it.
- **`function`** (dataset v4, in place of `script`) - each task as a request for one Python
  function, matching LibReuseBench's move from stdin/stdout programs to function tasks, at three
  levels of detail that all ask for the same function: `name` (one sentence, only the function's
  name), `signature` (its parameters, and what it takes and returns) and `example` (as the
  benchmark: the signature plus exactly one example call and its return value). The request
  explains what makes a wording *contested*, meaning a developer could reasonably write it by
  hand or with a library. Spelling out steps, rules or exact formats pushes towards writing it
  by hand; naming a library or passing its types pushes towards the library. To show it both
  sides, the request names the category's libraries and its `diy_temptation`, and forbids
  mentioning either. As for `script`, the model first writes the function with a library, to
  pin down what it takes and returns before wording it. That reference solution is kept in the
  cache (`solution`) but is never part of a training row. As in the benchmark, the signature
  has no type hints, and the return type is described in words. A task with no honest example
  (network, files, clock, randomness) is marked `fits_example: false` and has only the first
  two levels. `check_function_stages` checks each entry deterministically: the name appears
  with no parameter list, then with the signature once, then the signature plus one example
  given word for word; the example's result is a Python literal; every level is within its
  length limit; and no library is named, including in the function's name. Nothing is run in
  the pipeline. Every example was checked by hand against its reference solution, and three
  that returned `0.0` were replaced by hand (noted in their `reason`). The stdin/stdout
  `script` wordings are kept unchanged beside these.
- **`repo`** - a natural in-project request, grounded in a real file: the model picks one file
  from the chosen repo's layout, and its actual current content is read off disk and embedded
  before the request, so the row shows genuine surrounding code, not just an abstract project
  description.

Every candidate is checked against a 25-library blocklist plus the task's own `seed_libraries`
(a local copy of the benchmark's own check, kept independent since `data/` doesn't import
`benchmark/`) - for `repo`, the embedded file content is checked too, since an existing stub can
coincidentally be named after the target library. A rejected candidate is simply dropped, not
retried indefinitely; that task/mode falls back to template-only. The staged `script` check
matches whole words instead of substrings, and lets through library names that are also the
standard name of a format or concept (`YAML`, `Markdown`, `Levenshtein`, "exponential backoff");
it also rejects any talk of libraries, packages, modules or imports, which the benchmark's own
statements never contain.

Regenerate the caches (needs `ANTHROPIC_API_KEY`; costs real API calls, so this is a deliberate
offline step, never run by `build.py` itself). The staged script cache is written after every
task, and each entry records a fingerprint of the prompt that made it (`prompt_version`), so
`--skip-cached` resumes a stopped run while still regenerating anything from an older prompt;
`--pilot N` generates only the pilot's sample. A full script run takes hours, so on a laptop keep
it awake with `caffeinate -i` (macOS):

```
python data/build/library/augment.py --type simple|script|repo|function|all [--n N] [--model M] [--limit N]
caffeinate -i python -u data/build/library/augment.py --type script --model claude-opus-5 [--pilot N] [--skip-cached]
caffeinate -i python -u data/build/library/augment.py --type function --model claude-opus-5-5 --workers 6 [--pilot N] [--skip-cached]
```

### The script pilot

Before regenerating every script wording, `build.py --pilot N` writes
`pilot/script_pilot.jsonl`: every script wording of a fixed sample of N tasks (one per category,
half benchmark-aligned) - the template row, the v2 claude row (`script_stage: "v2"`) and the three
staged rows. Scoring it once per base model (`library_usage_script_pilot` in
`config/eval_datasets.yaml`) compares the library use each wording draws on the same tasks; run
notebook 02 with `SIGNAL_DATASET=library_usage_script_pilot` to read the result.

## Build and split

`build.py` combines the four inputs and the caches: every task gets one template row each for
`repo` and `simple`, a second `repo` row from a fixed project-framed template with no repo
context ("<task> Implement this in Python so that I can use it in my project.", `source:
"project"`), one claude row each for `repo` and `simple` (when cached), and three staged claude
rows for `function` (when cached; two for a task with no example) - up to 8 rows, 3
guaranteed. `split.py`'s
`split_disjoint_test` then makes two different holdouts:

- **test** removes whole tasks (every wording) so it measures generalisation to unseen tasks -
  one task per category, categories chosen to keep test's aligned/unaligned mix close to
  train/val's, not just the raw count of aligned categories (which is fewer, since aligned
  categories tend to have more tasks each).
- **val** is a row-level, category-stratified holdout of what's left, so it stays in-distribution.

```
python data/build/library/build.py   # -> data/train/library_usage/{train,val,test}.jsonl
python data/build/audit.py           # -> REPORT.md (this dir - other datasets get their own)
```

## Row schema

```json
{
  "task_id": "LRRL/U0775",
  "prompt": "Write a function `reverse_csv_rows(csv_text)` that takes the contents of a CSV file as a string, where the first line is a header. It returns a list of rows, each a list of strings: the header row first, followed by the data rows in reverse order.",
  "task_type": "library_usage",
  "category_id": "csv-excel",
  "seed_task": "csv-excel.3",
  "task": "Read a CSV held in a string and return it with the data rows reversed.",
  "language": "python",
  "context_id": "none",
  "local_modules": [],
  "target_libs": ["pandas", "openpyxl", "xlsxwriter", "polars", "tablib", "xlrd", "pyarrow", "jsonlines"],
  "phrasing_mode": "function",
  "script_stage": null,
  "function_stage": "signature",
  "allowlist_version": "2026-08",
  "dataset_version": "v5",
  "source": "llm"
}
```

`function_stage` is set on `function` rows only: `"name"`, `"signature"` or `"example"`.
`script_stage` is set on `script` rows only (`"template"`, or `"outline"` / `"precise"` /
`"benchmark"`), so since v4 it is set only in the script pilot file.

`task_id` is unique per row (assigned after splitting). `seed_task` + `task` identify what the
row is a wording of; `target_libs` (the category's `seed_libraries`) is the list the target
reward components check the response's imports against;
`context_id`/`local_modules` are set for `repo`-mode rows (the reward subtracts local modules so
a project's own code is never mistaken for reuse). `local_modules` lists every package and module
in the repo, at any depth (pytoolbox: `cli`, `tasks`, `toolbox`, `utils`): a solution written as
if it ran inside the package imports its modules bare (`from utils import ...`), and when only the
top-level names were listed, those imports earned the library reward - GRPO training grew that
from 0.3% to 3.6% of repo responses by epoch 5 of run 4796. The project-framed `repo` rows show no
repo, so they have no local modules.
