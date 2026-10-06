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

Alongside each mode's template, claude-generated wordings are cached offline - one per task for
`simple` and `repo`, three staged ones for `script`:

- **`simple`** - a short, natural chat-style request ("Can you write a function that...?"),
  optionally naming a plausible signature in plain built-in types only (never a type that would
  hint at a library). If two attempts both leak a library name, it retries without asking for a
  signature at all - some tasks' library name is also the natural English verb for the task
  (e.g. `slugify`), and dropping the signature ask is what actually fixes that, not rewording.
- **`script`** (dataset v3) - one concrete stdin/stdout exercise per task, written three times in
  LibReuseBench's template (an opening "Write a program that reads ... from standard input",
  `Input format:` and `Output format:` sections, and exactly one `Example:`), at increasing
  detail: `outline` (the job and the formats, just unambiguous), `precise` (plus every output and
  edge-case rule) and `benchmark` (the benchmark's full register). Each row records its stage as
  `script_stage`. The generator is told to keep the library's job in the problem without ever
  naming or hinting at a library: real-world text input (timestamps with zone names, HTML/XML/YAML
  documents, URLs, ...) rather than pre-parsed numbers, laid out line by line like most of the
  benchmark's inputs (CSV only when the task is about tabular data - an early pilot that listed
  CSV among its example formats got CSV input on 73% of benchmark-aligned tasks, against 11% in
  the benchmark), the variant of the task where hand-rolling takes the most work, standard
  concepts by their usual names ("ISO 8601", "Welch's t-test"), and never a description of the
  core computation. It sees the category's
  `diy_temptation` but not its libraries, and two real benchmark statements from the *dev*
  split (never test) as style anchors. A task that can't honestly be posed as text in, text out
  (binary media, files, network, running commands, a terminal UI, ...) is marked
  `fits_stdio: false` and keeps only
  its template script row. Every statement passes a deterministic template check
  (`check_script_stage`), with a failed attempt retried with its problems fed back.
  This replaced the v2 wording - one statement against a benchmark *test* example, which
  pre-digested the input until no library was useful (e.g. an audio clip became a list of
  integers) and gave almost no library-use signal.
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
python data/build/library/augment.py --type simple|script|repo|all [--n N] [--model M] [--limit N]
caffeinate -i python -u data/build/library/augment.py --type script --model claude-opus-5 [--pilot N] [--skip-cached]
```

### The script pilot

Before regenerating every script wording, `build.py --pilot N` writes
`pilot/script_pilot.jsonl`: every script wording of a fixed sample of N tasks (one per category,
half benchmark-aligned) - the template row, the v2 claude row (`script_stage: "v2"`) and the three
staged rows. Scoring it once per base model (`library_usage_script_pilot` in
`config/eval_datasets.yaml`) compares the library use each wording draws on the same tasks; run
notebook 02 with `SIGNAL_DATASET=library_usage_script_pilot` to read the result.

## Build and split

`build.py` combines the four inputs and the caches: every task gets one template row per mode,
one claude row each for `repo` and `simple` (when cached), and three staged claude rows for
`script` (when cached and posed as stdin/stdout) - up to 8 rows, 3 guaranteed. `split.py`'s
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
  "script_stage": null,
  "allowlist_version": "2026-08",
  "dataset_version": "v3",
  "source": "llm"
}
```

`script_stage` is set on `script` rows only: `"template"` for the template row, or `"outline"` /
`"precise"` / `"benchmark"` for the staged claude rows.

`task_id` is unique per row (assigned after splitting). `seed_task` + `task` identify what the
row is a wording of; `useful_libs` is what the reward checks the response's imports against;
`context_id`/`local_modules` are set for `repo`-mode rows (the reward subtracts local modules so
a project's own code is never mistaken for reuse).
