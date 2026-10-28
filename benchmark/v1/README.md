# LibReuseBench v1 (archived)

This folder is an earlier design of LibReuseBench, kept as a frozen record. It is not the
benchmark: LibReuseBench is the dataset in `benchmark/libreusebench/`, which replaced this design.
See `benchmark/HISTORY.md` for why.

This design rewrote BigCodeBench tasks as stdin/stdout programs that name no library. Each task
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
scores them. Later releases of the package ship the current benchmark instead.

## Results in this repository

These files are no longer in `config/eval_datasets.yaml`, whose LibReuseBench keys
(`libreusebench_test`, `libreusebench_control`) now score the current benchmark. Earlier results on
them stay in `output/`, under these names:

| Results name | File |
|---|---|
| `libreusebench_v1_test` | `data/test.jsonl` |
| `libreusebench_v1_control` | `data/control.jsonl` |
| `libreusebench_function_test` | `pipeline/function_pilot/test.jsonl` |
| `libreusebench_function_control` | `pipeline/function_pilot/control.jsonl` |

To score against one of them again, add an entry with its results name to
`config/eval_datasets.yaml`. All four use the `libreusebench` evaluator, which scores a task as a function task when its
record has a `function_name`, and as a stdin/stdout task otherwise.

## What no longer runs

The build scripts and their tests are not maintained, linted or collected by `pytest`. They
were written to run from `benchmark/pipeline/`, so they will not run from here unchanged:

- `pipeline/build.py` looks for the data folder and the fetched seed beside its parent folder.
- `pipeline/function_pilot/generate.py` imports `src/eval/scoring/libreusebench_function.py`,
  which has moved into the package as `libreusebench/functions.py`.

The pipeline in `benchmark/pipeline/` copies what it needs from here.
