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
[`src/grpo/rewards/`](src/grpo/rewards/)).

The project releases a benchmark and a reward package, both built from this repository:

- **LibReuseBench**, the library reuse benchmark: the dataset is on
  [Hugging Face](https://huggingface.co/datasets/itsluketwist/LibReuseBench), and its evaluator is
  on PyPI as [`libreusebench`](https://pypi.org/project/libreusebench/). Source in
  [`benchmark/`](benchmark/).
- **lure** (Library-Use Rewards), the reward functions used for training: on PyPI as
  [`lure`](https://pypi.org/project/lure/). Source in [`reward/`](reward/).

## *installation*

The project was built using Python, and is managed with `uv`, ensure it is installed or get it from [*uv docs*](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/library-reuse-rl
cd library-reuse-rl
```

Install the project. The GPU training/inference stack (torch, vLLM, TRL, transformers, ...) only
runs on linux/CUDA, so it lives in an optional, linux-marked `hpc` extra - the core install (data
handling, code analysis, the `benchmark` pipeline, tooling) works anywhere.

On **Isambard** (Arm GH200 nodes), install inside a GPU allocation and check the stack works.
The `hpc` pins use CUDA 12.9 builds of torch and vLLM, because the nodes' driver can't run the
CUDA 13 builds that PyPI now defaults to - see the comments in `pyproject.toml`:

```shell
uv sync --locked --extra hpc
python scripts/check_gpu_stack.py                               # cuda, triton, training step, vllm
torchrun --nproc-per-node=4 scripts/check_gpu_stack.py --nccl   # in a 4-gpu allocation
```

`scripts/discover_environment.sh` reports a node's hardware, driver, modules and Slurm limits,
without needing the project installed.

Isambard's storage areas serve different purposes, and the jobs are wired up accordingly by
[`scripts/setup_job.sh`](scripts/setup_job.sh):

| Area | What goes there | Why |
| --- | --- | --- |
| `$HOME` (100 GiB) | The repo, `.venv`, `logs/`, and the small files under `output/` | Wanted in git, and small |
| `$PROJECTDIR` | Model downloads (`HF_HOME`, set in `.env`) and training checkpoints | Hundreds of GB that must persist - `$HOME` is too small |
| `$SCRATCHDIR` | Compile caches, wandb staging, the disposable evaluation venvs | Regenerable churn; scratch is purged after 60 days of inactivity |

Jobs refuse to start if `HF_HOME` is unset or inside `$HOME`, since a single model cache
would fill that quota.

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
accelerate launch --config_file config/accelerate_configs/fsdp2.yaml --num_processes 4 \
    --module src.run_grpo --model qwen3-8b -d data/train/library_usage -rp lu-v1

run_eval -m qwen3-8b
```

Training is **full-weight** GRPO: the model, its gradients and its optimizer state are
sharded across the job's GPUs with FSDP2 (see
[`config/accelerate_configs/fsdp2.yaml`](config/accelerate_configs/fsdp2.yaml)), while a
vLLM engine colocated on the same GPUs generates the rollouts. Evaluation needs only one
GPU. Run either command with `--help` to see all options (base/decoding/reward
configuration, which datasets to evaluate against, etc).

Every epoch leaves a checkpoint, registered in `config/trained_models.yaml` as
`{run-id}-epoch{n}` (debug runs use the git-ignored `config/debug_models.yaml` instead),
so each epoch can be evaluated and compared afterwards. Only the
newest checkpoint keeps the optimizer state needed to resume, so a job stopped by the
slurm time limit is requeued and picks up from the last epoch.

On a slurm cluster, use the submission scripts in [`scripts/`](scripts/) instead - edit the
configuration block at the top of each, then run e.g. `bash scripts/submit_grpo.sh`, which
also queues one eval job per epoch checkpoint, held until training succeeds. Untrained
base models are evaluated by running `bash scripts/submit_eval.sh` directly.

Before a real run - or after changing the training code - work through the staged checks
in [`scripts/submit_debug.sh`](scripts/submit_debug.sh) in order, each of which submits one
job (eval checks the checkpoint smoke trains):

```shell
bash scripts/submit_debug.sh smoke   # 4 gpus, ~10 minutes - training end to end, small model
bash scripts/submit_debug.sh eval    # 1 gpu, ~10 minutes - the evaluation path
bash scripts/submit_debug.sh scale   # 4 gpus, ~30 minutes - memory and timings at 8b
bash scripts/submit_debug.sh clean   # no job - deletes the checks' checkpoints and results
```

`eval` and `scale` also take a model key, to check a new candidate model before using it
(e.g. `bash scripts/submit_debug.sh eval seed-coder-8b`, then `scale seed-coder-8b`).

There are two eval launchers, each with its own configuration block at the top (models,
datasets, decoding profiles, prompt styles, debug, update), so the two can be pointed at different
model sets: [`scripts/submit_eval.sh`](scripts/submit_eval.sh) submits one slurm job per
combination, and [`scripts/run_eval_local.sh`](scripts/run_eval_local.sh) runs them
sequentially on the current node.

Results are saved in the model's folder - `output/baselines/{model}/` for a base model, and
`output/{base-model}/{run-id}/epoch{n}/` for a training run's epoch (see
[`src/utils/paths.py`](src/utils/paths.py)) - keyed by dataset/decoding-profile/prompt-style,
as two files per combination, both committed as the experimental record: `{name}.json`
holds the run's settings and its aggregate analysis, and `{name}.generations.jsonl` holds
one line per response - its full text and its scores. The code isn't stored separately:
`lure.parse_response` (see [`reward/`](reward/)) re-extracts it from the text.
If scoring a dataset fails (e.g. its code can't be run), its generations are kept in a
git-ignored `{name}.unscored.jsonl` beside where its results would go, and the job carries on
with the other datasets - running the same evaluation again scores those saved generations
instead of generating them again.
Re-running the same combination overwrites its previous result - a combination whose
result file already exists is skipped instead, unless the `update` setting (`--update` on
`run_eval`) is set. Each invocation's logs still get their own `logs/{run_timestamp}/`
folder, with what each job ran in its log's name (e.g.
`eval-{model}-{decoding-profile}-{prompt-style}-{job-id}.out`) - and a `submit_grpo.sh`
submission puts its training log (`train-{model}-{dataset}-{job-id}.out`) and the logs of
the eval jobs it queues in one such folder. Summarise results into per-task comparison tables (and a `summary.csv`) with:

```shell
python scripts/summarize_eval.py
```

### *api-served models*

A saved [LibReuseBench comparison of ML4, GLM-5.3 and DeepSeek Flash across direct
API calls, Vibe and mini-swe](output/05_harness_ablation/README.md)
includes per-task submissions, corrected library-use metrics, token statistics,
source/runtime audits, plots and regeneration instructions.

A follow-up non-agentic run of the same three models (high reasoning, temperature 1,
top-p 1, five samples per task) is saved as baselines under the `default_32k` decoding
profile, in `output/baselines/{mistral-large-4,glm-5.3,deepseek-v4.1-flash}/`. Its
original per-replica export is in the git history (commit 52076df).

Models can be evaluated over an OpenAI-compatible API instead of local weights, by giving
their `config/models.yaml` entry `provider: api` and an `api_key_env` naming the `.env`
variable that holds the key (copy [`.env.example`](.env.example) to `.env` and fill it in).
`base_url` selects the provider, `max_tokens_param` covers models that reject the
legacy `max_tokens` field, and `max_workers` (default 8) caps how many requests are in flight
at once, for a provider with a low concurrency limit. A request that still fails is scored as a
wrong answer, so check the run's warnings before trusting its results. These runs never
import vllm, and only need network access - so
they are not submitted to slurm (Isambard has no CPU partition); run them with
[`scripts/run_eval_local.sh`](scripts/run_eval_local.sh). A hosted model has no local
weights, so its registry `path` stays blank. See
[`src/eval/generate_api.py`](src/eval/generate_api.py).

Pass `--debug` to either command to smoke-test the code end-to-end: it truncates the
dataset(s) to a handful of samples, and defaults to the tiny `qwen3-debug` model (see
[`config/models.yaml`](config/models.yaml)) if `-m`/`--model` isn't given. Debug eval results
are written under the git-ignored `output/debug/`, so they never replace real results.

```shell
run_grpo -d data/train/library_usage --debug

run_eval --debug
```

The same flag is available as the `debug` setting in
[`scripts/submit_grpo.sh`](scripts/submit_grpo.sh) and
[`scripts/submit_eval.sh`](scripts/submit_eval.sh), and as the `smoke` and `eval` modes of
[`scripts/submit_debug.sh`](scripts/submit_debug.sh).

Set `libraries_requirements_file` in the recipe config
([`config/training.yaml`](config/training.yaml)) to a requirements file from
[`setup/requirements/`](setup/requirements/) to include the available libraries in the
prompt, so the model knows what it can import and reuse - see
[`src/prompts.py`](src/prompts.py). `run_eval` reads this straight from the same
recipe file too, so training and evaluation always agree on it.

## *rewards*

GRPO training scores each response with a reward function built from named **reward
profiles** in [`config/rewards.yaml`](config/rewards.yaml). Each profile maps reward
component names to the weight they contribute when satisfied - a component left out
of a profile never contributes for that profile:

```yaml
lu-v1:
  correct_format: 0.25
  passes_syntax_check: 0.25
  external_lib_imported: 0.5
  external_lib_used: 1.0
  appropriate_lib_used: 0
```

Pick a profile by name with `-rp`/`--reward-profile` (default `full`). The resolved
profile (name + weights) is recorded in the run's `train_results.json` and in the
trained model's registry `properties` (see [`src/run_grpo.py`](src/run_grpo.py)), so a
run's reward function stays recoverable.

The reward logic lives in [`src/grpo/rewards/`](src/grpo/rewards/). The dataset row's
`task_type` selects the reward class automatically; `-rp` selects that class's component
weights. Incompatible combinations fail before training instead of silently becoming no-op
components. Every reward (and every evaluator) parses responses with
[lure](reward/README.md), so training and evaluation judge code, and the libraries it
imports and uses, in exactly the same way.

**To add a new reward profile**, add a top-level entry to `config/rewards.yaml` using
existing component names.

**To add a new reward component**, in `src/grpo/rewards/`:

1. Add its name to `REWARD_COMPONENTS` and the relevant task's supported-component set.
2. Judge it in that reward class's `components()`.
3. Reference the new name in a `config/rewards.yaml` profile.

Each weighted component is given to the trainer as its own reward function (see
`build_reward_functions`), so the training logs carry each component's rate -
e.g. `rewards/external_lib_used/mean` is the fraction of completions that reused a
library. The total reward is unchanged: the trainer multiplies each by its weight and sums
them.

## *structure*

- [`config/`](config/) - YAML config: the model registry (hand-written base models in
  `models.yaml`, and each epoch of a training run in `trained_models.yaml` once `run_grpo`
  registers them), the training recipe and its sharding
  config, decoding profiles, and each eval dataset's path, scorer, and execution
  environment.
- [`data/`](data/) - Built training/evaluation datasets and their construction pipelines;
  see [`data/README.md`](data/README.md).
- [`notebooks/`](notebooks/) - Analysis notebooks: the paper figures and the per-run training
  summary at the top, `ablation/` and `analysis/` for the rest. Results and figures stay in
  `output/`.
- [`reward/`](reward/) - lure (Library-Use Rewards), our response parsing and library
  analysis, released on PyPI as a standalone package (`pip install lure`); see
  [`reward/README.md`](reward/README.md). It is used by the rewards, the evaluators and the
  benchmark, and installed editable here.
- [`output/`](output/) - Base models' evaluation results under `output/baselines/{model}/`,
  and everything from a training run under `output/{base-model}/{run-id}/`: its training
  metrics in `grpo/`, and each epoch's evaluation results in `epoch1/`, `epoch2/`, ... (see
  [`src/utils/paths.py`](src/utils/paths.py)). The checkpoints themselves are far too large for
  the repo and live under `--output-root` instead (project storage on the HPC, see
  `trained_model_path` in [`src/run_grpo.py`](src/run_grpo.py)). Debug runs are
  gitignored; finished results are committed as the experimental record.
- [`scripts/`](scripts/) - Job launchers (`setup_job.sh`, `submit_grpo.sh`,
  `submit_eval.sh` and `submit_debug.sh` for slurm, `run_eval_local.sh` for the current
  node), `check_gpu_stack.py` for verifying an install, and `summarize_eval.py` for
  reading results back.
- [`setup/`](setup/) - Defines the isolated venv used to sandbox LLM-generated code during
  evaluation, kept separate from the project's own venv so we control exactly which
  libraries generated code can import (`setup_eval_venv.sh`, `requirements/`).
- [`src/`](src/) - The main project code:
  - `run_grpo.py` - the RL training entry point: CLI parsing, dataset loading, output
    naming, and model registration.
  - `grpo/` - the GRPO training loop itself (reward functions, trainer setup, and
    `checkpoints.py` for reading and tidying the per-epoch checkpoints).
  - `run_eval.py` - the evaluation entry point: CLI parsing, dataset/output-path
    naming.
  - `eval/` - the evaluation loop itself (model/tokenizer loading, generation,
    scoring each response, saving results).
  - `prompts.py` - builds the system prompt from the available-libraries list.
  - `response/` - runs a response's code (parsed by lure) in a sandbox against a record's
    test cases, measures code size and maintainability, and maps stop reasons.
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

## *authors*

Lukas Twist, Shuyin Ouyang, Helen Yannakoudakis and Jie M. Zhang, all at King's College London.

## *paper*

This repository accompanies the paper *Reinforcement Learning with Library-Use Rewards*, coming
soon.

## *citation*

If you use this code, LibReuseBench or lure, please cite:

```bibtex
@misc{twist2026reinforcement,
  title        = {Reinforcement Learning with Library-Use Rewards},
  author       = {Twist, Lukas and Ouyang, Shuyin and Yannakoudakis, Helen and Zhang, Jie M.},
  year         = {2026},
  howpublished = {\url{https://github.com/itsluketwist/library-reuse-rl}},
}
```
