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

The project was built using Python 3.11, any other version does not guarantee reproducibility of the experiments.
Check if you have it installed with the command below, otherwise download and install it from
[here](https://www.python.org/downloads/).

```shell
python3.11 --version
```

Now clone the repository code:

```shell
git clone https://github.com/itsluketwist/library-reuse-rl
```

If on a compatible HPC, you can load the exact modules used:

```shell
module load python/3.11.6-gcc-13.2.0
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
```

Configure pip to use the correct PyTorch index before installing:

```shell
mkdir -p ~/.config/pip
echo -e "[global]\nextra-index-url = https://download.pytorch.org/whl/cu121" > ~/.config/pip/pip.conf
```

Now create a virtual environment and install the requirements:

```shell
python3.11 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

pip install -e .
```

Create the isolated venv used to run llm-generated code during evaluation.

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

Pass `--debug` to `run_grpo` to smoke-test the training code end-to-end: it truncates the
train/val splits to a handful of samples, and defaults to the tiny `qwen3-debug` model
(see [`config/models.yaml`](config/models.yaml)) if `-m`/`--model` isn't given.

```shell
run_grpo -d data/skywork --debug
```

The same flag is available as the `debug` setting in [`scripts/submit_grpo.sh`](scripts/submit_grpo.sh).

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

# update dependency lockfiles
uv pip compile requirements.in --output-file requirements.txt --upgrade
```

## *paper*

todo

## *citation*

todo
