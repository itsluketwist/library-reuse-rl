#!/usr/bin/env bash
# staged checks to run before (or after changing) a real training run. each mode submits
# one slurm job, so the longer ones don't hold an interactive allocation.
#
#   bash scripts/submit_debug.sh smoke   # 4 gpus, minutes - wiring and the sharded launch
#   bash scripts/submit_debug.sh pilot   # 1 gpu, ~1-2 hours - the whole pipeline, small model
#   bash scripts/submit_debug.sh eval    # 1 gpu, ~10 minutes - the evaluation path
#   bash scripts/submit_debug.sh scale   # 4 gpus, ~30 minutes - memory and step time at 8b
#
# the eval mode takes an optional model key, so it can check either the debug base model
# or a checkpoint the pilot just trained:
#
#   bash scripts/submit_debug.sh eval qwen3-debug-pilot-epoch1

set -euo pipefail

###############################################################################
#                         cluster settings (isambard)                         #
###############################################################################

ACCOUNT="brics.u6yb"
PARTITION="workq"

# checkpoints go to project storage, as real runs do (see scripts/submit_grpo.sh)
OUTPUT_BASE="${PROJECTDIR:?PROJECTDIR is not set - are you on isambard?}/$USER/library-reuse-rl"

# the same training configuration the real run uses, so these checks exercise it
accelerate_config="config/accelerate_configs/fsdp2.yaml"
recipe_config="config/training.yaml"
dataset="data/train/library_usage"
reward_profile="lu-v1"
prompt_style="base"
seed=42

# datasets the eval mode scores (see config/eval_datasets.yaml)
eval_datasets="library_usage,libreusebench_test,libreusebench_control"

###############################################################################
#                            mode selection                                   #
###############################################################################

mode="${1:-}"
eval_model="${2:-qwen3-debug}"

case "$mode" in
    smoke)
        # tiny truncated run: proves imports, the fsdp2 launch and the reward wiring
        gpus=4
        time_limit="01:00:00"
        command="accelerate launch --config_file $accelerate_config --num_processes 4 \
    --module src.run_grpo --debug --model qwen3-debug -d $dataset \
    --reward-profile $reward_profile --library-prompt-style $prompt_style \
    --output-root $OUTPUT_BASE"
        ;;
    pilot)
        # one real epoch on the small model: checkpointing, registration, resuming, and
        # the validation curve. a fixed run id, so repeats overwrite one registry entry.
        gpus=1
        time_limit="04:00:00"
        command="accelerate launch --config_file $accelerate_config --num_processes 1 \
    --module src.run_grpo --model qwen3-debug --epochs 1 -d $dataset \
    --reward-profile $reward_profile --library-prompt-style $prompt_style \
    -s $seed --run-id qwen3-debug-pilot --output-root $OUTPUT_BASE"
        ;;
    scale)
        # a few steps of the real model: sharding, the vllm weight sync, peak memory and
        # step time. stops before the first epoch ends, so it registers nothing.
        gpus=4
        time_limit="01:00:00"
        command="accelerate launch --config_file $accelerate_config --num_processes 4 \
    --module src.run_grpo --model qwen3-8b --max-steps 3 -d $dataset \
    --reward-profile $reward_profile --library-prompt-style $prompt_style \
    -s $seed --run-id qwen3-8b-scale --output-root $OUTPUT_BASE"
        ;;
    eval)
        # truncated evaluation: loading weights in vllm, the execution venvs, scoring and
        # the result files - the path the post-training eval jobs depend on
        gpus=1
        time_limit="01:00:00"
        command="run_eval --debug --model $eval_model -d $eval_datasets \
    --decoding-profile default --prompt-style $prompt_style"
        ;;
    *)
        echo "Usage: bash scripts/submit_debug.sh {smoke|pilot|eval|scale} [model-key]"
        echo
        echo "  smoke  4 gpus, minutes      tiny truncated training run"
        echo "  pilot  1 gpu,  ~1-2 hours   one epoch of the real split, small model"
        echo "  eval   1 gpu,  ~10 minutes  truncated evaluation of [model-key]"
        echo "  scale  4 gpus, ~30 minutes  a few steps of qwen3-8b, for memory and timing"
        exit 1
        ;;
esac

###############################################################################
#                              submit                                         #
###############################################################################

mkdir -p logs/debug

echo
echo ">>> Submitting debug job: $mode >>>"
echo "  GPUs: $gpus"
echo "  Time limit: $time_limit"
echo "  Output: $OUTPUT_BASE"
echo

job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=debug-${mode}
#SBATCH --output=logs/debug/${mode}-%j.out
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${gpus}
#SBATCH --time=${time_limit}

source ./scripts/setup_job.sh

$command

echo "Debug run complete: $mode [\$(date '+%Y-%m-%d %H:%M:%S')]"
EOF
)

echo "Submitted job $job_id - follow it with:"
echo "  tail -f logs/debug/${mode}-${job_id}.out"
