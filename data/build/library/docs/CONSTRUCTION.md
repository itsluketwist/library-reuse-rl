# library_usage — construction

How the `library_usage` dataset is built. This is the internal/reviewer-facing doc; see
[../../../README.md](../../../README.md) for how the whole `data/` pipeline fits together and
`PROMPT_COMPARISON.md` for real sample output.

## The idea

Each row asks for a small Python feature whose natural solution reaches for a mature library,
but which a model could also be tempted to hand-roll. The reward needs no execution: it just
extracts the response's imports, subtracts the standard library and any local project modules,
and checks what's left. So the whole dataset is built from four small, hand-authored inputs plus
one offline LLM pass - no scraping, no seed dataset.

## The four inputs

- **`taxonomy.yaml`** - 66 categories. Each has `seed_libraries` (the libraries that count as
  reuse), a `diy_temptation` (what hand-rolling it looks like), and `repo_hint` (which of the two
  repos below its tasks get framed in). 26 categories are `benchmark_aligned: true` - their
  skills were chosen by clustering the LibReuseBench eval benchmark's own test problems, so
  training deliberately overlaps what that benchmark measures.
- **`tasks.yaml`** - 194 concrete tasks, a `{number: text}` mapping per category (e.g.
  `datetime: {1: "Parse an ISO-8601 timestamp...", 2: ...}`). A task's `seed_task` id is
  `"<category_id>.<number>"`, e.g. `"datetime.1"` - the thing every wording of that task traces
  back to.
- **`templates.yaml`** - 2 fixed wordings for each of 3 modes: `repo` (framed inside a real
  project), `simple` (a short, bare ask), `script` (stdin/stdout, reminiscent of the benchmark's
  register). One of the 2 is picked per task per mode, deterministically (seeded by `seed_task`
  and mode), rather than emitting both.
- **`repos/pytoolbox/`** (authored here) and the vendored `python-api` (shared with the amplify
  pipeline) - two small real Python projects `repo`-mode tasks get framed inside. Each category
  picks one via `repo_hint`.

## The LLM layer (`augment.py`)

Alongside each mode's template, one claude-generated wording is cached offline:

- **`simple`** - a short, natural chat-style request ("Can you write a function that...?"),
  optionally naming a plausible signature in plain built-in types only (never a type that would
  hint at a library). If two attempts both leak a library name, it retries without asking for a
  signature at all - some tasks' library name is also the natural English verb for the task
  (e.g. `slugify`), and dropping the signature ask is what actually fixes that, not rewording.
- **`script`** - a LibReuseBench-register stdin/stdout exercise statement, generated against a
  real embedded benchmark example so it matches that register's length and precision rather than
  hedging every edge case.
- **`repo`** - a natural in-project request, grounded in a real file: the model picks one file
  from the chosen repo's layout, and its actual current content is read off disk and embedded
  before the request, so the row shows genuine surrounding code, not just an abstract project
  description.

Every candidate is checked against a 25-library blocklist plus the task's own `seed_libraries`
(a local copy of the benchmark's own check, kept independent since `data/` doesn't import
`benchmark/`) - for `repo`, the embedded file content is checked too, since an existing stub can
coincidentally be named after the target library. A rejected candidate is simply dropped, not
retried indefinitely; that task/mode falls back to template-only.

Regenerate the caches (needs `ANTHROPIC_API_KEY`; costs real API calls, so this is a deliberate
offline step, never run by `build.py` itself):

```
python data/build/library/augment.py --type simple|script|repo|all [--n N] [--model M] [--limit N]
```

## Build and split

`build.py` combines the four inputs and the three caches: every task gets one template row and
(when cached) one claude row per mode - up to 6 rows, 3 guaranteed. `split.py`'s
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
  "task_id": "LRRL/U1044",
  "prompt": "Can you write a function that takes a CSV string, keeps the header where it is, but reverses the order of the data rows, and returns the result as a string?",
  "task_type": "library_usage",
  "category_id": "csv-excel",
  "seed_task": "csv-excel.3",
  "task": "Read a CSV held in a string and return it with the data rows reversed.",
  "language": "python",
  "context_id": "none",
  "local_modules": [],
  "useful_libs": ["pandas", "openpyxl", "xlsxwriter"],
  "phrasing_mode": "simple",
  "allowlist_version": "2026-08",
  "dataset_version": "v2",
  "source": "llm"
}
```

`task_id` is unique per row (assigned after splitting). `seed_task` + `task` identify what the
row is a wording of; `useful_libs` is what the reward checks the response's imports against;
`context_id`/`local_modules` are set for `repo`-mode rows (the reward subtracts local modules so
a project's own code is never mistaken for reuse).
