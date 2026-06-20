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
[`src/response/reward.py`](src/response/reward.py)).

## *installation*

The code requires Python 3.12 to run.
Ensure you have it installed with the command below, otherwise download and install it from
[here](https://www.python.org/downloads/).

```shell
python3.12 --version
```

Now clone the repository code:

```shell
git clone https://github.com/itsluketwist/library-reuse-rl
```

Once cloned, install the requirements locally in a virtual environment:

```shell
python3.12 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

pip install -e .
```

## *usage*

After [*installation*](#installation), training and evaluation are run via the `train` and
`eval` console scripts (installed by `pip install -e .`, see
[`src/cli_train.py`](src/cli_train.py) and [`src/cli_eval.py`](src/cli_eval.py)):

```shell
train -m qwen3-8b -d data/skywork

eval -m qwen3-8b
```

Run either with `--help` to see all options (base/decoding/reward configuration, which
datasets to evaluate against, etc).

On a slurm cluster, use the submission scripts in [`scripts/`](scripts/) instead - edit the
configuration block at the top of each, then run e.g. `bash scripts/submit_train.sh`. This
also takes care of automatically queuing an `eval` job once `train` finishes.

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
  (see `trained_model_path` in [`src/run_train.py`](src/run_train.py)). Gitignored.
- [`scripts/`](scripts/) - Bash scripts for running jobs on a slurm cluster
  (`setup_job.sh`, `submit_train.sh`, `submit_eval.sh`).
- [`setup/`](setup/) - Defines the isolated venv used to sandbox LLM-generated code during
  evaluation, kept separate from the project's own venv so we control exactly which
  libraries generated code can import (`setup_eval_venv.sh`, `requirements/`).
- [`src/`](src/) - The main project code:
  - `cli_train.py` / `run_train.py` - the RL training entry point and loop.
  - `cli_eval.py` / `run_eval.py` - the evaluation entry point and loop.
  - `response/` - parses an LLM response into candidate code, sandboxes and runs it
    against a record's test cases, and computes a reward from the outcome.
  - `utils/` - shared config loading, logging, and I/O helpers.
- [`tests/`](tests/) - Unit tests, mirroring the structure of `src/`.

## *development*

```shell
pip install -e ".[dev]"

# run tests
pytest tests/

# run linting
pre-commit run --all-files

# update dependency lockfiles
uv pip compile requirements.txt --output-file requirements.frozen --upgrade

# create the isolated venv used to run llm-generated code during evaluation, with a
# controlled set of libraries - see setup/requirements/ for the available versions
bash setup/setup_eval_venv.sh v1
```

## *paper*

todo

## *citation*

todo
