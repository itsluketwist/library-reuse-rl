# library_usage: version history

The runs on v2-v4 all used one reward (`lu-v1`): a response earns library credit when its code
uses a library outside the standard library and the project's own modules. What changes between
those versions is which tasks are asked, and how they are worded. v5 changes only each task's
list of target libraries, which the target reward of its runs counts. Every row records its
`dataset_version`.

| Version | Date | Wordings per task | Rows (train / val / test) | Run trained on it |
| --- | --- | --- | --- | --- |
| v1 | 2026-08-11 | `repo` ×4, `bare` ×3 (7, templates only) | 592 / 52 / 70 | none kept |
| v2 | 2026-09-14 | `repo`, `simple`, `script` (6) | 916 / 101 / 108 | 4796 (and 7405) |
| v3 | 2026-10-08 | `repo` ×3, `simple` ×2, `script` ×4 (9) | 1,362 / 166 / 167 | e650 |
| v4 | 2026-10-17 | `repo` ×3, `simple` ×2, `function` ×3 (8) | 1,209 / 135 / 145 | 8672 |
| v5 | 2026-10-24 | as v4 | 1,209 / 135 / 145 | cc26, 4474, 1f71 |

The runs below are all qwen3-8b, trained with the lu-v1 reward, the base prompt and lr 3e-6.
Each was scored with the same settings on the provisional build of LibReuseBench
(`libreusebench_v2_full`: 233 tasks, before human review). None of them was scored on the final
benchmark (216 test tasks), and these numbers can't be compared with its numbers. The v5 runs
were scored on the final benchmark: see [v5](#v5-audited-target-libraries).

| Model | Uses a library | … on high-value tasks | pass@1 |
| --- | --- | --- | --- |
| base qwen3-8b | 15.6% | 40.8% | 76.3% |
| 4796, v2 data, epoch 5 | 26.4% | 55.0% | 79.8% |
| e650, v3 data, epoch 3 | 27.6% | 56.2% | 79.7% |
| 8672, v4 data, epoch 6 | 34.2% | 66.7% | 77.9% |

## v1: templates only

- 102 tasks, each with 7 template rows: inside a project (`repo`: 2 templates × 2 repos) and as
  a bare request (`bare`: 3 templates).
- Superseded before any run we report.

## v2: three modes, two sources

- 194 tasks in 66 categories. 26 of the categories are "benchmark-aligned": chosen to cover
  what LibReuseBench measured at the time (its 129 stdin/stdout test problems, now archived in
  `benchmark/v1/`, clustered by the library call doing the real work).
- Each task was asked in 3 modes: inside a project (`repo`), as a bare request (`simple`), and as
  a stdin/stdout program (`script`), the format of the benchmark at the time. Each mode had a
  template row and a claude-written row.
- The test split holds out 19 whole tasks, one per category. They have been the same in every
  version since.
- **Result (run 4796):** library use on test and val rose from 27.8% to 49.3%, then plateaued
  after epoch 2. Script rows almost never used a library, and on the stdin/stdout benchmark,
  library use only went from 1.9% to about 4%.

## v3: staged script wordings

- **Why:** the script rows were the closest to the benchmark, yet gave almost no learning signal.
- The claude script row became three stdin/stdout exercises in the benchmark's format, at
  increasing detail (`outline`, `precise`, `benchmark`).
- A project-framed repo row was added, with no repo shown: "... so that I can use it in my
  project".
- Every module of a repo now counts as local (it used to be only the top-level ones), closing a
  loophole where importing the project's own files earned library credit.
- **Result (run e650):** the staged script rows did learn (`outline` 2.9% to 17.1%), but they were
  long and were cut off 15-28% of the time. On the stdin/stdout benchmark, library use was no
  better than 4796's. On prompts shared with v2, repo rows were 10 points lower than under 4796.

## v4: function requests instead of script rows

- **Why:** LibReuseBench moved from stdin/stdout programs to function tasks
  (`Write a function name(args) that ...`).
- The four script rows became three requests for one named function, at increasing detail:
  `name` (the name only), `signature` (its parameters, and what it takes and returns), and
  `example` (as in the benchmark: the signature and one call with its result). Like the
  benchmark's, the signatures have no type hints.
- The generator was told explicitly to keep each request *contested*: as reasonable to write by
  hand as with a library.
- 39 tasks have no honest example (network, files, the clock, randomness), so they get only the
  first two levels.
- Repo and simple rows, and the test tasks, are the same as in v3. The script wordings are kept,
  but only the script pilot uses them.
- **Result (run 8672, 6 epochs):** the best library use on the provisional benchmark of any v2-v4
  run: 34.2%, against e650's 27.6%. Most of the gain is on high-value tasks, where a library saves the most code. pass@1 did
  not change. Training was clean: about 2% of rollouts were cut off by the end. On its own test
  split, the `example` rows barely moved (11% to 16%) while `name` rows rose from 38% to 53%: the
  more a request spells out, the less often a library is used.

## v5: audited target libraries

- **Why:** experiment 01's `target` reward only pays for a library on the task's list, so the
  lists had to be complete and correct. In v4 they only fed a zero-weighted bonus.
- `useful_libs` is renamed `target_libs`. Every other field of every row is word for word v4's
  (checked row by row when it was built).
- The lists (each category's `seed_libraries` in `taxonomy.yaml`) were audited in
  `notebooks/04_target_library_audit.ipynb`, against every saved `library_usage` evaluation:
  - **Removed:** `argparse` (cli-parsing). It is standard library, so it could never count.
  - **Added, from what models used:**
    - old names of listed libraries: `fuzzywuzzy`, `PyPDF2`;
    - real alternatives: `Crypto`, `argon2`, `librosa`, `mutagen`, `py7zr`, `unidecode`,
      `dns`, `hdbscan`, `dateutil` for log timestamps, `packaging` for version strings,
      `fpdf`, `deepmerge`, `scipy` for reading WAV files;
    - numpy for polynomial fits (regression-fitting's own reference solution uses it);
    - pandas where it does the task's real work: `to_datetime` and business-day offsets
      (datetime), `tz_localize`/`tz_convert` (timezones), `.corr()` (hypothesis-testing),
      `read_html` (html-parsing), `.rolling()` (signal-processing).
  - **Added, as other well-known libraries for the tasks:** 145 more, e.g. `polars`
    wherever pandas is listed for table work, `dateparser` and `humanize` for dates, `rich`
    for terminal tables, `toposort` and `rustworkx` for graphs, `w3lib` and `tldextract` for
    URLs, `segno` for QR codes.
  - **Left out on purpose** (used by models, but not a library for the task): app plumbing a
    repo prompt pulls in (fastapi, click, pydantic, sqlalchemy...), `requests` where it only
    fetches the data, numpy or pandas as a container for hand-written maths (e.g. numpy in
    clustering, pandas in numerics), `matplotlib` beside the real work, `psycopg2` (a database
    driver, not a SQL builder), and bs4 for stripping rendered markdown.
- Rows now list 0-8 more libraries each (most 1-4). Only the target reward and the target
  rates see the change; every v4 library is still listed, so the other rewards are unaffected.
- **Result (runs 4474, cc26 and 1f71: the import, usage and target rewards, 8 epochs; see
  `output/03/README.md`):** on the final LibReuseBench test split, library use rose from 15.7%
  to a peak of about 37.5% under every reward (import at epoch 2, usage at 3, target at 3-4),
  then fell to 26-29% by epoch 8. On this dataset's held-out splits it kept rising, to 85-92% by
  the end, so the later epochs fit the training prompts rather than the benchmark; the next runs
  use 4 epochs. Only the target reward kept the choice on the target lists: at the end, 86% of
  val answers used a target library, against 77% for usage and 52% for import.
