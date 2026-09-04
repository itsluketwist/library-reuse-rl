# **library-reuse-rl**

<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://creativecommons.org/licenses/by/4.0/">
        <img alt="CC-BY-4.0 License" src="https://img.shields.io/badge/Licence-CC_BY_4.0-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
</div>

## *about*

Research code studying whether reinforcement learning can teach LLMs to reuse existing
libraries when writing code, instead of reimplementing functionality from scratch. Models
are trained and evaluated on stdin/stdout coding problems, with a reward function that can
optionally reward library imports/usage on top of plain correctness (see
[`src/grpo/rewards.py`](src/grpo/rewards.py)).

## *installation*

The project was built using Python, and is managed with `uv`, ensure it is installed or get it from [*uv docs*](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/library-reuse-rl
cd library-reuse-rl
```

Install the project. The GPU training/inference stack (torch, vLLM, unsloth, flash-attn, ...) only
builds on linux/CUDA, so it lives in an optional, linux-marked `hpc` extra - the core install (data
handling, code analysis, the `benchmark` pipeline, tooling) works anywhere.

On the HPC (full training + evaluation stack):

```shell
module load cuda/12.2.1-gcc-13.2.0
uv sync --locked --extra hpc
source .venv/bin/activate
```

Locally (core only - for the `benchmark` pipeline, `dataset/` build, and tests; the GPU stack is
linux-marked, so it is skipped automatically):

```shell
uv sync --locked
source .venv/bin/activate
```

Create the isolated venv used by the ordinary stdin/stdout code benchmarks:

```shell
bash setup/setup_eval_venv.sh v1
```

LibReuseBench does not use this venv. Its official evaluator discovers the packages
imported by the generated responses, builds an isolated environment under
`.libreuseeval-venvs/`, installs those packages, and then runs the benchmark tests. Results record
the requested packages, their originating imports, and all resolved distribution versions.

That environment is disposable: each run gets its own and deletes it afterwards, so
evaluations running at the same time (a whole slurm submission, say) can never share and
corrupt one venv. Pass `libreuseeval --env-dir <path>` to reuse a named venv across runs
instead, trading that isolation for not reinstalling the same packages every time.

## *usage*

After [*installation*](#installation), training and evaluation are run via the `run_grpo`
and `run_eval` console scripts (installed into the venv by `uv sync`, see
[`src/run_grpo.py`](src/run_grpo.py) and [`src/run_eval.py`](src/run_eval.py)):

```shell
run_grpo -m qwen3-8b -d data/train/library_usage -rp lu-v1

run_eval -m qwen3-8b
```

Run either with `--help` to see all options (base/decoding/reward configuration, which
datasets to evaluate against, etc).

On a slurm cluster, use the submission scripts in [`scripts/`](scripts/) instead - edit the
configuration block at the top of each, then run e.g. `bash scripts/submit_grpo.sh`. Submit
evaluation separately after the trained run ids have been registered in `config/models.yaml`.

There are two eval launchers, each with its own configuration block at the top (models,
datasets, decoding profiles, prompt styles, debug), so the two can be pointed at different
model sets: [`scripts/submit_eval.sh`](scripts/submit_eval.sh) submits one slurm job per
combination, and [`scripts/run_eval_local.sh`](scripts/run_eval_local.sh) runs them
sequentially on the current node.

Each invocation gets its own timestamp, and results are saved under
`output/eval/{run_timestamp}/{model}/`, so re-running a combination never overwrites an
earlier run. The launchers pass their timestamp through with `--run-timestamp`, so results
and the matching `logs/{run_timestamp}/` folder share a name. Summarise a finished run
into per-task comparison tables (and a `summary.csv`) with:

```shell
python scripts/summarize_eval.py output/eval/260816T174336
```

### *api-served models*

Models can be evaluated over an OpenAI-compatible API instead of local weights, by giving
their `config/models.yaml` entry `provider: api` and an `api_key_env` naming the `.env`
variable that holds the key (copy [`.env.example`](.env.example) to `.env` and fill it in).
`base_url` selects the provider, and `max_tokens_param` covers models that reject the
legacy `max_tokens` field. These runs never import vllm, and `submit_eval.sh` routes them
to a CPU partition since they only need network access. Adapters do not apply to a hosted
model. See [`src/eval/generate_api.py`](src/eval/generate_api.py).

Pass `--debug` to either command to smoke-test the code end-to-end: it truncates the
dataset(s) to a handful of samples, and defaults to the tiny `qwen3-debug` model (see
[`config/models.yaml`](config/models.yaml)) if `-m`/`--model` isn't given.

```shell
run_grpo -d data/train/skywork --debug

run_eval --debug
```

The same flag is available as the `debug` setting in
[`scripts/submit_grpo.sh`](scripts/submit_grpo.sh) and
[`scripts/submit_eval.sh`](scripts/submit_eval.sh).

Set `libraries_requirements_file` in the recipe config
([`config/training.yaml`](config/training.yaml)) to a requirements file from
[`setup/requirements/`](setup/requirements/) to include the available libraries in the
prompt, so the model knows what it can import and reuse - see
[`src/prompts.py`](src/prompts.py). `run_eval` reads this straight from the same
recipe file too, so training and evaluation always agree on it.

Responses generated outside this repo (e.g. by [`fireworks/`](fireworks/), which only
generates - see its README) are scored with the `score_generations` console script
(see [`src/score_generations.py`](src/score_generations.py)), instead of `run_eval`'s
in-process vllm generation:

```shell
score_generations -d library_usage -g output/fireworks/qwen3-8b/library_usage-qwen3.jsonl
```

`-d`/`--dataset` names a dataset from `config/eval_datasets.yaml`, same as `run_eval`, so
the dataset path, evaluator, and execution environment are never duplicated. `-g`/`--generations`
is a JSONL file with one record per (`task_id`, `sample_index`) carrying at least `content`
and `finish_reason` (the schema `fireworks/fw_rlvr/inference.py` writes). Output defaults
alongside the generations file, replacing its suffix with `.json`.

## *rewards*

GRPO training scores each response with a reward function built from named **reward
profiles** in [`config/rewards.yaml`](config/rewards.yaml). Each profile maps reward
component names to the weight they contribute when satisfied - a component left out
of a profile never contributes for that profile:

```yaml
full:
  correct_format: 0.1
  passes_syntax_check: 0.05
  passed_tests: 1.0

correctness_only:
  passed_tests: 1.0
```

Pick a profile by name with `-rp`/`--reward-profile` (default `full`). The resolved
profile (name + weights) is recorded in the run's `train_results.json` and in the
trained model's registry `properties` (see [`src/run_grpo.py`](src/run_grpo.py)), so a
run's reward function stays recoverable.

The reward logic lives in [`src/grpo/rewards/`](src/grpo/rewards/). The dataset row's
`task_type` selects the reward class automatically; `-rp` selects that class's component
weights. Incompatible combinations fail before training instead of silently becoming no-op
components.

**To add a new reward profile**, add a top-level entry to `config/rewards.yaml` using
existing component names.

**To add a new reward component**, in `src/grpo/rewards/`:

1. Add its name to `REWARD_COMPONENTS` and the relevant task's supported-component set.
2. Score it in the relevant reward class.
3. Reference the new name in a `config/rewards.yaml` profile.

## *structure*

- [`config/`](config/) - YAML config: the model registry (base models, and trained models
  once `run_grpo` registers one), decoding profiles, and each eval dataset's path, scorer,
  and execution environment.
- [`data/`](data/) - Built training/evaluation datasets and their construction pipelines;
  see [`data/README.md`](data/README.md).
- [`notebooks/`](notebooks/) - One-off, numbered notebooks for exploratory work that doesn't
  belong in `src/`.
- [`output/`](output/) - Trained model outputs, named and registered automatically by
  `run_grpo` (see `trained_model_path` in [`src/run_grpo.py`](src/run_grpo.py)), and
  evaluation results under `output/eval/{run_timestamp}/`. Debug runs are gitignored;
  finished results are committed as the experimental record.
- [`scripts/`](scripts/) - Job launchers (`setup_job.sh`, `submit_grpo.sh`,
  `submit_eval.sh` for slurm, `run_eval_local.sh` for the current node), and
  `summarize_eval.py` for reading results back.
- [`setup/`](setup/) - Defines the isolated venv used to sandbox LLM-generated code during
  evaluation, kept separate from the project's own venv so we control exactly which
  libraries generated code can import (`setup_eval_venv.sh`, `requirements/`).
- [`src/`](src/) - The main project code:
  - `run_grpo.py` - the RL training entry point: CLI parsing, dataset loading, output
    naming, and model registration.
  - `grpo/` - the GRPO training loop itself (reward function, trainer setup).
  - `run_eval.py` - the evaluation entry point: CLI parsing, dataset/output-path
    naming.
  - `score_generations.py` - the entry point for scoring responses generated outside
    this repo (e.g. by `fireworks/`): CLI parsing, dataset/output-path naming.
  - `eval/` - the evaluation loop itself (model/tokenizer loading, generation,
    scoring each response, saving results), plus `score_generations.py`'s alignment
    of externally generated responses onto a dataset for the same scorers.
  - `models.py` - model/tokenizer loading, shared by training and evaluation.
  - `prompts.py` - builds the system prompt from the available-libraries list.
  - `response/` - parses an LLM response into candidate code, sandboxes and runs it
    against a record's test cases, and computes a reward from the outcome.
  - `utils/` - shared config loading, logging, and I/O helpers.
- [`tests/`](tests/) - Unit tests, mirroring the structure of `src/`.

## *development*

```shell
# run tests
pytest tests/

# run linting
pre-commit run --all-files

# update dependency lockfile (uv.lock) after changing pyproject.toml
uv lock --upgrade
```

## *paper*

todo

## *citation*

todo
