# LibReuseBench v1 (archived)

This folder is the first version of LibReuseBench, kept as a frozen record. Version 2 replaces
it: see `benchmark/docs/V2_PLAN.md` for why, and for what changes.

Version 1 rewrote BigCodeBench tasks as stdin/stdout programs that name no library. Each task
can be solved with a third-party library or with the standard library alone.

## What is here

| Folder | Contents |
|---|---|
| `data/` | The released splits: `test.jsonl` (179 tasks), `dev.jsonl` (10) and `control.jsonl` (79). |
| `pipeline/` | The build scripts, stages 01 to 08. |
| `pipeline/function_pilot/` | The function-style pilot: the same tasks reworded as a request for one named function (177 test and 79 control tasks). |
| `artefacts/` | The output of every build stage, and the reviewers' verdicts. |
| `docs/` | `CONSTRUCTION.md` (how v1 was built) and `REPORT.md` (the build audit). |
| `tests/` | The unit tests for the build scripts. |

The data is evaluation-only. `data/CANARY.txt` marks it as such.

## The same data is on PyPI

`pip install libreusebench==0.1.1` ships exactly these three splits, with the harness that
scores them. Later releases of the package ship version 2.

## Running v1 evaluations from this repository

These dataset names in `config/eval_datasets.yaml` still score against the archived files, so
results stay comparable with the existing files in `output/`:

| Dataset name | File |
|---|---|
| `libreusebench_test` | `data/test.jsonl` |
| `libreusebench_control` | `data/control.jsonl` |
| `libreusebench_function_test` | `pipeline/function_pilot/test.jsonl` |
| `libreusebench_function_control` | `pipeline/function_pilot/control.jsonl` |

All four use the `libreusebench` evaluator, which scores a task as a function task when its
record has a `function_name`, and as a stdin/stdout task otherwise.

## What no longer runs

The build scripts and their tests are not maintained, linted or collected by `pytest`. They
were written to run from `benchmark/pipeline/`, so they will not run from here unchanged:

- `pipeline/build.py` looks for the data folder and the fetched seed beside its parent folder.
- `pipeline/function_pilot/generate.py` imports `src/eval/scoring/libreusebench_function.py`,
  which has moved into the package as `libreusebench/functions.py`.

The version 2 pipeline in `benchmark/pipeline/` copies what it needs from here.
