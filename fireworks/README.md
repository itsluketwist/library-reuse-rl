# Fireworks RLVR pilot

This directory adapts the existing library-reuse experiment to Fireworks without
duplicating its datasets or final evaluation logic. It provides a portable structural
reward, matched generation-only base/RFT inference, and a guarded RFT launcher. Final
evaluation remains entirely in the main repository code.

The reward never executes generated code. It parses only final-answer content with
Python's `ast`; `reasoning_content` is saved for later analysis but is not scored.

Its normalized shaping matches the main repository's `lu-v1` intent: `0.00` for no
parseable Python, `0.25` for parseable Python without external reuse, `0.50` for an
external import that is not used, and `1.00` for an imported and referenced external
library. It does not classify libraries as appropriate, inappropriate, or other.

## Layout

```text
existing data -> rft/prepare_dataset.py -> Eval Protocol / Fireworks RFT
config/eval_datasets.yaml -> main prompt loader -> infer_base.py or infer_rft.py
infer_base.py or infer_rft.py -> output/fireworks/{model}/ raw generation JSONL
```

Nothing under `fireworks/` evaluates saved generations. Use the main repository's
evaluation code for scoring so local and Fireworks generations share one implementation.

`fw_rlvr/` is deliberately neutral: the top-level directory is not a Python package
named `fireworks`, so it cannot shadow a Fireworks SDK package. The evaluator directory
contains all reward-time code and can be uploaded independently.

## Install and configure

Keep project inference and evaluation in the normal repository environment and install Eval
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

## 4. Generate base outputs

Inference reads named datasets from `config/eval_datasets.yaml` and delegates prompt
construction to `src.data.load_eval_dataset`, exactly as the main evaluation path does.
By default it generates both `library_usage` and `libreusebench`. It does not prepare RFT
rows and does not run any evaluator.

```bash
venv
python fireworks/infer_base.py \
  --model accounts/ACCOUNT_ID/deployments/library-reuse-qwen3-8b \
  --model-name qwen3-8b-base \
  --datasets library_usage,libreusebench \
  --decoding-profile qwen3 \
  --prompt-style base \
  --seed 42
```

This writes:

```text
output/fireworks/qwen3-8b-base/library_usage-qwen3.jsonl
output/fireworks/qwen3-8b-base/libreusebench-qwen3.jsonl
```

Omit `--model-name` to use the deployment ID's final segment as the directory. The
streamed output is resumable with `--resume`. It records task ID, prompt messages, final
content, separate reasoning content, finish reason, usage, resolved generation config,
and raw SDK response data. It deliberately stores no reward ground truth and existing
files are never overwritten implicitly. The named profile's `n_samples` is used unless
`--samples` overrides it; apply identical profiles and prompt settings to compared runs.

## 5. Launch the pilot RFT job

Eval Protocol's current single-turn workflow registers the evaluator during the decorated
test run, then creates RFT with `eval-protocol create rft`. The installed CLI infers a
local JSONL source from the evaluator decorator. The launcher sets that source to the
full prepared training split, validates every row itself, and asks the CLI to skip its
second API-backed validation run so hundreds of redundant local rollouts are not created.
The pinned Eval Protocol release incorrectly uses `typing.Literal` itself as its argument
converter, so it rejects every explicit loss enum. The launcher invokes the unmodified
package through a small, repository-owned compatibility entrypoint that fixes only this
conversion; `GRPO` is then passed and recorded exactly as required by the Fireworks SDK.

First inspect the installed interface and the exact command without creating anything:

```bash
venv
.venv-eval/bin/eval-protocol create rft --help
python fireworks/rft/launch.py \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset fireworks/rft/data/train.jsonl \
  --dry-run
```

After reviewing the printed configuration, first run a remote smoke job against the
12-row development sample. This validates the uploaded evaluator, rollout generation,
reward recording, and optimizer startup before the full dataset is submitted:

```bash
python fireworks/rft/launch.py \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset fireworks/rft/evaluator/development/sample.jsonl \
  --record fireworks/rft/runs/qwen3-8b-smoke-v1.json \
  --extra-argument=--force \
  --extra-argument=--rl-loss-method \
  --extra-argument=GRPO \
  --extra-argument=--epochs \
  --extra-argument=1 \
  --extra-argument=--lora-rank \
  --extra-argument=8 \
  --extra-argument=--batch-size \
  --extra-argument=32768 \
  --extra-argument=--response-candidates-count \
  --extra-argument=8 \
  --extra-argument=--max-output-tokens \
  --extra-argument=2048 \
  --yes
```

The smoke job should process 12 prompts and 96 rollouts. Do not launch the main job unless
the smoke job reaches `COMPLETED`, processes nonzero requests, records nonzero output rows,
and exposes plausible reward/component metrics and sample trajectories.

Then launch the main experiment with the full training split. Choose and record the LoRA
rank as an experimental hyperparameter; the example below uses rank 16 rather than silently
inheriting Fireworks' rank-8 default:

```bash
python fireworks/rft/launch.py \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset fireworks/rft/data/train.jsonl \
  --record fireworks/rft/runs/qwen3-8b-grpo-rank16-v1.json \
  --extra-argument=--force \
  --extra-argument=--job-id \
  --extra-argument=library-reuse-qwen3-8b-grpo-r16-v1 \
  --extra-argument=--output-model \
  --extra-argument=accounts/ACCOUNT_ID/models/library-reuse-qwen3-8b-grpo-r16-v1 \
  --extra-argument=--rl-loss-method \
  --extra-argument=GRPO \
  --extra-argument=--rl-kl-beta \
  --extra-argument=0.001 \
  --extra-argument=--epochs \
  --extra-argument=2 \
  --extra-argument=--lora-rank \
  --extra-argument=16 \
  --extra-argument=--learning-rate \
  --extra-argument=0.0001 \
  --extra-argument=--max-context-length \
  --extra-argument=32768 \
  --extra-argument=--batch-size \
  --extra-argument=32768 \
  --extra-argument=--batch-size-samples \
  --extra-argument=0 \
  --extra-argument=--gradient-accumulation-steps \
  --extra-argument=1 \
  --extra-argument=--learning-rate-warmup-steps \
  --extra-argument=0 \
  --extra-argument=--optimizer-weight-decay \
  --extra-argument=0.0 \
  --extra-argument=--response-candidates-count \
  --extra-argument=8 \
  --extra-argument=--max-output-tokens \
  --extra-argument=2048 \
  --extra-argument=--temperature \
  --extra-argument=1.0 \
  --extra-argument=--top-p \
  --extra-argument=1.0 \
  --extra-argument=--chunk-size \
  --extra-argument=200 \
  --extra-argument=--max-inference-replica-count \
  --extra-argument=1 \
  --extra-argument=--node-count \
  --extra-argument=1 \
  --yes
```

These values are written explicitly so the scientific configuration is visible rather
than inherited silently from service defaults. Use other repeatable `--extra-argument`
values only after checking the installed help. The launcher invokes no shell, captures
stdout/stderr, and saves timestamps plus any returned dashboard/resource IDs. It refuses
a real launch without `--yes`; tests exercise only dry or mocked launches.

The full-run parameter decisions are:

| Parameter | Value | Reason |
|---|---:|---|
| loss method | GRPO | make the grouped-policy objective explicit |
| KL beta | 0.001 | retain a small reference-policy constraint |
| epochs | 2 | the rank-8 baseline reward was still rising after one epoch |
| LoRA rank | 16 | double baseline capacity without jumping to the maximum rank |
| learning rate | 0.0001 | keep the successful managed-RFT baseline rate |
| max context | 32,768 tokens | retain the successful baseline limit |
| batch size | 32,768 tokens | retain the successful token-based update batch |
| sample batch size | 0 | use token batching rather than a second sample limit |
| gradient accumulation | 1 | one packed batch per optimizer update |
| warmup steps | 0 | match the baseline and avoid another simultaneous change |
| optimizer weight decay | 0.0 | match the resolved baseline configuration |
| candidates per prompt | 8 | provide the same GRPO comparison group as the baseline |
| max output | 2,048 tokens | bound rollout cost and discourage reasoning overflow |
| temperature / top-p | 1.0 / 1.0 | retain baseline exploration settings |
| chunk size | 200 prompts | retain the successful three-chunk workflow |
| inference replicas / trainer nodes | 1 / 1 | avoid changing distributed execution |

Top-k, provider-specific `extra_body`, concurrency limits, LoRA target modules/alpha, and
the trainer sharding shape remain unset because this managed interface either does not
expose them consistently or owns their safe model-specific defaults. The committed
52-row validation split remains held out for matched post-training evaluation instead of
being auto-carved from training. Eval Protocol's RFT CLI does not expose a seed for this
managed workflow, so reproducibility relies on the recorded data order and complete job
configuration rather than deterministic rollouts.

### Rank-32, ten-epoch stress run

The follow-up stress run changes only capacity, duration, and resource names relative to
the explicit main configuration above:

| Parameter | Previous run | Stress run |
|---|---:|---:|
| LoRA rank | 16 | 32 (managed RFT maximum) |
| epochs | 2 | 10 |
| planned rollouts | 9,472 | 47,360 |
| job ID | `library-reuse-qwen3-8b-grpo-r16-v1` | `library-reuse-qwen3-8b-grpo-r32-e10-v1` |
| output model | `library-reuse-qwen3-8b-grpo-r16-v1` | `library-reuse-qwen3-8b-grpo-r32-e10-v1` |

This is a deliberately aggressive exploratory run, not a controlled single-variable
comparison: both adapter capacity and training duration change. All other arguments in
the full-run command remain identical. Fireworks' managed RFT backend rejects ranks above
32 even though the current public CLI help only documents the rank as an integer.

### Test rewards and RFT without a deployment

This is the current pilot path when base inference is intentionally deferred:

1. Prepare the training data in step 1.
2. Run the offline reward tests in step 2; skip the API-backed `ep local-test`.
3. Run the RFT launcher with `--dry-run` and inspect the resolved model, dataset, row
   count, evaluator directory, and command.
4. Launch the 12-row remote smoke RFT job with `--yes` when ready.
5. Require the smoke job to complete with processed rollouts and plausible reward traces.
6. Launch the full RFT job and inspect its reward curves, component metrics, and sampled
   trajectories in Fireworks.

The RFT job may use `accounts/fireworks/models/qwen3-8b` directly as its base-model
resource. This does not make that catalog model callable through the inference API.
Deployment remains required later for matched base inference and for querying the trained
LoRA model.

## 6. Inspect, deploy, and generate with the RFT model

Inspect reward curves and sampled trajectories in the returned dashboard before
deploying the resulting LoRA model. Deploy through the UI or a reviewed `firectl`
command; do not reuse an unrelated base deployment. Then query its deployment resource
through the same inference implementation:

```bash
venv
python fireworks/infer_rft.py \
  --model accounts/ACCOUNT_ID/deployments/FINE_TUNED_DEPLOYMENT_ID \
  --model-name library-reuse-qwen3-8b-grpo-r16-v1 \
  --datasets library_usage,libreusebench \
  --decoding-profile qwen3 \
  --prompt-style base \
  --seed 42
```

The inference scripts require an existing deployment and never create, scale, stop, or
delete one. With `--min-replica-count 0`, Fireworks can scale an idle deployment to zero,
but a request that triggers a cold start may initially return `503`; retry the resumable
command after the deployment is ready. Compare the saved `generation_config` objects
before comparing outcomes, run final evaluation through the main project, and explicitly
delete an unneeded deployment with `firectl deployment delete DEPLOYMENT_ID`.

## Pilot checklist

1. Set `FIREWORKS_API_KEY`.
2. Prepare Fireworks training and validation JSONL.
3. Pass the offline reward tests.
4. Dry-run and review the RFT command.
5. Launch a short RFT job and inspect rewards and sampled trajectories.
6. When matched inference is required, create a Qwen3-8B on-demand base deployment.
7. Run the API-backed Eval Protocol local test against that deployment and inspect traces.
8. Save raw Qwen3-8B base outputs for both evaluation datasets.
9. Deploy the resulting RFT model.
10. Save raw RFT outputs with exactly the base decoding configuration.
11. Evaluate both sets of outputs only through the main repository code.
12. Scale down or delete the deployment when inference is complete.

Steps 6–12 may be deferred while validating only the reward and RFT workflow. Do not run
Qwen3-8B inference or its API-backed local rollout before step 6.

Changing from Qwen3-8B to a later model such as Kimi requires model/deployment IDs and,
if intentional, a named decoding profile—not changes to the adapter or reward.
