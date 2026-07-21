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

Install the project:

```shell
uv sync --locked
source .venv/bin/activate
```

Create the isolated venv used to run llm-generated code during evaluation:

```shell
bash setup/setup_eval_venv.sh v1
```

## *usage*

After [*installation*](#installation), training and evaluation are run via the `run_grpo`
and `run_eval` console scripts (installed by `pip install -e .`, see
[`src/run_grpo.py`](src/run_grpo.py) and [`src/run_eval.py`](src/run_eval.py)):

```shell
run_grpo -m qwen3-8b -d data/skywork

run_eval -m qwen3-8b
```

Run either with `--help` to see all options (base/decoding/reward configuration, which
datasets to evaluate against, etc).

On a slurm cluster, use the submission scripts in [`scripts/`](scripts/) instead - edit the
configuration block at the top of each, then run e.g. `bash scripts/submit_grpo.sh`. This
also takes care of automatically queuing a `run_eval` job once `run_grpo` finishes.

Pass `--debug` to either command to smoke-test the code end-to-end: it truncates the
dataset(s) to a handful of samples, and defaults to the tiny `qwen3-debug` model (see
[`config/models.yaml`](config/models.yaml)) if `-m`/`--model` isn't given.

```shell
run_grpo -d data/skywork --debug

run_eval --debug
```

The same flag is available as the `debug` setting in
[`scripts/submit_grpo.sh`](scripts/submit_grpo.sh) and
[`scripts/submit_eval.sh`](scripts/submit_eval.sh).

Set `libraries_requirements_file` (training, in the recipe config) or
`--libraries-requirements-file` (eval) to a requirements file from
[`setup/requirements/`](setup/requirements/) to include the available libraries in the
prompt, so the model knows what it can import and reuse - see
[`src/prompts.py`](src/prompts.py).

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

Pick a profile by name with `-rp`/`--reward-profile` (default `full`), and it's baked
into the trained model's output directory name (see `trained_model_path` in
[`src/run_grpo.py`](src/run_grpo.py)) so a run's reward function is always recoverable
from its path. The resolved profile (name + weights) is also saved into the run's
`train_results.json`, alongside the rest of its training metrics.

The actual reward logic lives in [`src/grpo/rewards.py`](src/grpo/rewards.py):

- `REWARD_COMPONENTS` is the set of valid component names - the names you're allowed
  to use as keys in a `config/rewards.yaml` profile.
- `compute_reward` is a single function with one `if weight := weights.get("name"):`
  block per component, each checking that component's condition and adding its weight
  to the total if satisfied.

**To add a new reward profile**, just add a new top-level entry to
`config/rewards.yaml` using any of the existing component names.

**To add a new reward component**, in `src/grpo/rewards.py`:

1. Add its name to the `REWARD_COMPONENTS` frozenset.
2. Add a matching `if weight := weights.get("your_component_name"):` block in
   `compute_reward`, with a short comment explaining what it checks and `reward +=
   weight` if the condition is met.
3. Reference the new name in a profile in `config/rewards.yaml` to use it.

## *structure*

- [`config/`](config/) - YAML config: the model registry (base models, and trained models
  once `train` registers one), decoding profiles, and the eval dataset name → file mapping.
- [`data/`](data/) - Scripts that download and format the datasets used by the project
  ([`download_skywork.py`](data/download_skywork.py) for RL training data,
  [`download_eval.py`](data/download_eval.py) for the held-out eval benchmarks). Run them
  directly (e.g. `python data/download_skywork.py`) - their outputs are written alongside
  them (`data/skywork/`, `data/eval/`) and are not committed.
- [`notebooks/`](notebooks/) - One-off, numbered notebooks for exploratory work that doesn't
  belong in `src/`.
- [`output/`](output/) - Trained model outputs, named and registered automatically by `train`
  (see `trained_model_path` in [`src/run_grpo.py`](src/run_grpo.py)). Gitignored.
- [`scripts/`](scripts/) - Bash scripts for running jobs on a slurm cluster
  (`setup_job.sh`, `submit_grpo.sh`, `submit_eval.sh`).
- [`setup/`](setup/) - Defines the isolated venv used to sandbox LLM-generated code during
  evaluation, kept separate from the project's own venv so we control exactly which
  libraries generated code can import (`setup_eval_venv.sh`, `requirements/`).
- [`src/`](src/) - The main project code:
  - `run_grpo.py` - the RL training entry point: CLI parsing, dataset loading, output
    naming, and model registration.
  - `grpo/` - the GRPO training loop itself (reward function, trainer setup).
  - `run_eval.py` - the evaluation entry point: CLI parsing, dataset/output-path
    naming.
  - `eval/` - the evaluation loop itself (model/tokenizer loading, generation,
    scoring each response, saving results).
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
