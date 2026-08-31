# Fireworks RLVR pilot

This directory adapts the existing library-reuse experiment to Fireworks without
duplicating its datasets or final evaluation logic. It provides a portable structural
reward, matched base/RFT inference, a guarded RFT launcher, and a bridge back into the
project scorer.

The reward never executes generated code. It parses only final-answer content with
Python's `ast`; `reasoning_content` is saved for later analysis but is not scored.

## Layout

```text
existing data -> rft/prepare_dataset.py -> Eval Protocol / Fireworks RFT
existing prompts -> infer_base.py or infer_rft.py -> generation JSONL
generation JSONL -> score_generations.py -> existing project evaluation
```

`fw_rlvr/` is deliberately neutral: the top-level directory is not a Python package
named `fireworks`, so it cannot shadow a Fireworks SDK package. The evaluator directory
contains all reward-time code and can be uploaded independently.

## Install and configure

Keep project inference/scoring in the normal repository environment and install Eval
Protocol in the separate, ignored `.venv-eval` environment:

```bash
venv
uv venv .venv-eval --python .venv/bin/python
uv pip install --python .venv-eval/bin/python \
  --requirement fireworks/requirements.txt
export FIREWORKS_API_KEY="..."
```

The Eval Protocol SDK is pinned to the reviewed official Git commit in
`requirements.txt`. Copy `.env.example` if useful, but explicitly export or otherwise
load its values—the project inference scripts do not read `.env` files themselves.

> **Qwen3-8B deployment requirement:** `accounts/fireworks/models/qwen3-8b` is a
> valid catalog/base-model ID for creating an RFT job, but Qwen3-8B is not available on
> Fireworks serverless inference. A paid on-demand deployment is therefore required for
> base-model inference and API-backed `ep local-test` rollouts. It is **not** required
> for the offline reward tests, the RFT launcher dry run, or for submitting an RFT job.
> The no-deployment path is documented in [Test rewards and RFT without a deployment](#test-rewards-and-rft-without-a-deployment).

## 1. Prepare training data

The adapter reads the committed project splits and retains prompt text plus only the
metadata needed by the structural evaluator:

```bash
venv
python fireworks/rft/prepare_dataset.py \
  --dataset-dir data/train/library_usage \
  --output-dir fireworks/rft/data \
  --splits train,val \
  --prompt-style base
```

This writes generated train/validation JSONL under the ignored `fireworks/rft/data/`
directory and a deterministic 12-row development sample under
`rft/evaluator/development/`.

## 2. Validate the reward

Run the offline adversarial suite before making any API call:

```bash
venv
pytest -q tests/test_fireworks/test_reward.py
```

These unit tests exercise the same portable `score_response` implementation used by the
uploaded evaluator, without generating model completions or requiring a deployment.

An API-backed Eval Protocol rollout tests completion generation as well as reward
calculation. It therefore needs either a serverless model or an existing on-demand
deployment. For Qwen3-8B, complete step 3 first, set `FIREWORKS_EVAL_MODEL` to the
deployment resource with Eval Protocol's `fireworks_ai/` provider prefix, and then use
two terminals. The environment flag prevents ordinary repository tests from accidentally
making model calls.

```bash
# terminal 1
.venv-eval/bin/ep logs
```

```bash
# terminal 2
cd fireworks/rft/evaluator
source ../../../.venv-eval/bin/activate
export FIREWORKS_EVAL_MODEL="fireworks_ai/accounts/ACCOUNT_ID/deployments/DEPLOYMENT_ID"
RUN_FIREWORKS_EVAL=1 ep local-test --yes
```

This exercises the portable evaluator source tree and generates API-backed rollouts.

## 3. Prepare a Qwen3-8B base deployment

Qwen3-8B is present in the Fireworks model catalog and supports fine-tuning, but it is
not a serverless inference model. Before base inference or an API-backed Qwen3-8B
`ep local-test`, create the paid resource deliberately in the Fireworks UI or with
`firectl`:

```bash
firectl deployment create accounts/fireworks/models/qwen3-8b \
  --deployment-id library-reuse-qwen3-8b \
  --region global \
  --min-replica-count 0 \
  --max-replica-count 1 \
  --wait
firectl deployment get library-reuse-qwen3-8b
```

Confirm the exact model ID in the current Fireworks model library first; available model
IDs, shapes, accelerators, and pricing can change. Record the deployment resource returned
by Fireworks; inference must use that resource, not
`accounts/fireworks/models/qwen3-8b`. The inference scripts never create or delete
deployments.

## 4. Generate and score base outputs

Use a held-out prepared split. Both inference commands resolve the same named decoding
profile from `config/training.yaml`; any override must be applied identically to both
runs.

```bash
venv
python fireworks/infer_base.py \
  --model accounts/ACCOUNT_ID/deployments/library-reuse-qwen3-8b \
  --input fireworks/rft/data/val.jsonl \
  --output fireworks/output/qwen3-8b-base.jsonl \
  --decoding-profile qwen3 \
  --seed 42

python fireworks/score_generations.py \
  --dataset data/train/library_usage/val.jsonl \
  --generations fireworks/output/qwen3-8b-base.jsonl \
  --output fireworks/output/qwen3-8b-base-results.json \
  --evaluator library_usage
```

The streamed output is resumable with `--resume`. It records prompt messages, final
content, separate reasoning content, finish reason, usage, resolved generation config,
and raw SDK response data. Existing files are never overwritten implicitly.

## 5. Launch the pilot RFT job

Eval Protocol's current single-turn workflow registers the evaluator during the decorated
test run, then creates RFT with `eval-protocol create rft`. The installed CLI infers a
local JSONL source from the evaluator decorator. The launcher sets that source to the
full prepared training split, validates every row itself, and asks the CLI to skip its
second API-backed validation run so hundreds of redundant local rollouts are not created.

First inspect the installed interface and the exact command without creating anything:

```bash
venv
.venv-eval/bin/eval-protocol create rft --help
python fireworks/rft/launch.py \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset fireworks/rft/data/train.jsonl \
  --dry-run
```

After reviewing the printed configuration, deliberately launch it:

```bash
python fireworks/rft/launch.py \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset fireworks/rft/data/train.jsonl \
  --record fireworks/rft/runs/qwen3-8b-pilot.json \
  --yes
```

Use repeatable `--extra-argument` values only after checking the installed help, for
example `--extra-argument=--epochs --extra-argument=2`. The launcher invokes no shell,
captures stdout/stderr, and saves timestamps plus any returned dashboard/resource IDs.
It refuses a real launch without `--yes`; tests exercise only dry or mocked launches.

### Test rewards and RFT without a deployment

This is the current pilot path when base inference is intentionally deferred:

1. Prepare the training data in step 1.
2. Run the offline reward tests in step 2; skip the API-backed `ep local-test`.
3. Run the RFT launcher with `--dry-run` and inspect the resolved model, dataset, row
   count, evaluator directory, and command.
4. Launch a deliberately short RFT job with `--yes` when ready.
5. Inspect its reward curves, component metrics, and sampled trajectories in Fireworks.

The RFT job may use `accounts/fireworks/models/qwen3-8b` directly as its base-model
resource. This does not make that catalog model callable through the inference API.
Deployment remains required later for matched base inference and for querying the trained
LoRA model.

## 6. Inspect, deploy, and evaluate the RFT model

Inspect reward curves and sampled trajectories in the returned dashboard before
deploying the resulting LoRA model. Deploy through the UI or a reviewed `firectl`
command; do not reuse an unrelated base deployment. Then query its deployment resource
through the same inference implementation:

```bash
venv
python fireworks/infer_rft.py \
  --model accounts/ACCOUNT_ID/deployments/FINE_TUNED_DEPLOYMENT_ID \
  --input fireworks/rft/data/val.jsonl \
  --output fireworks/output/qwen3-8b-rft.jsonl \
  --decoding-profile qwen3 \
  --seed 42

python fireworks/score_generations.py \
  --dataset data/train/library_usage/val.jsonl \
  --generations fireworks/output/qwen3-8b-rft.jsonl \
  --output fireworks/output/qwen3-8b-rft-results.json \
  --evaluator library_usage
```

Compare the saved `generation_config` objects before comparing outcomes. Structural
library reuse is the training reward; executable correctness tests belong only in the
existing project's final evaluation stage. Stop or delete any on-demand deployment when
the experiment is finished to avoid idle cost.

## Pilot checklist

1. Set `FIREWORKS_API_KEY`.
2. Prepare Fireworks training and validation JSONL.
3. Pass the offline reward tests.
4. Dry-run and review the RFT command.
5. Launch a short RFT job and inspect rewards and sampled trajectories.
6. When matched inference is required, create a Qwen3-8B on-demand base deployment.
7. Run the API-backed Eval Protocol local test against that deployment and inspect traces.
8. Save Qwen3-8B base outputs and existing library-use metrics.
9. Deploy the resulting RFT model.
10. Save RFT outputs with exactly the base decoding configuration.
11. Feed both output files into the existing scorer.
12. Run executable correctness only during final evaluation.

Steps 6–12 may be deferred while validating only the reward and RFT workflow. Do not run
Qwen3-8B inference or its API-backed local rollout before step 6.

Changing from Qwen3-8B to a later model such as Kimi requires model/deployment IDs and,
if intentional, a named decoding profile—not changes to the adapter or reward.
