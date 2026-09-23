#!/usr/bin/env bash
# staged checks to run before (or after changing) a real training run. each mode submits
# one slurm job, so the longer ones don't hold an interactive allocation.
#
#   bash scripts/submit_debug.sh smoke   # 4 gpus, ~10 minutes - training end to end, small model
#   bash scripts/submit_debug.sh eval    # 1 gpu, ~10 minutes - the evaluation path
#   bash scripts/submit_debug.sh scale   # 4 gpus, ~30 minutes - memory and timings at 8b
#
# run them in that order: eval defaults to the checkpoint smoke trains, and takes an
# optional model key to check another one instead (e.g. the untrained base model):
#
#   bash scripts/submit_debug.sh eval qwen3-debug
#
# afterwards, delete what the checks leave behind (it lists everything and asks first):
#
#   bash scripts/submit_debug.sh clean

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
eval_datasets="library_usage,library_usage_val,libreusebench_test,libreusebench_control"

###############################################################################
#                            mode selection                                   #
###############################################################################

mode="${1:-}"
eval_model="${2:-qwen3-debug-smoke-epoch2}"

case "$mode" in
    smoke)
        # tiny truncated run (2 epochs of 16 prompts) through the real 4-gpu setup: the
        # fsdp2 launch, vllm, rewards, checkpointing and registration. a fixed
        # run id, so repeats overwrite the same registry entries that eval then loads
        gpus=4
        time_limit="01:00:00"
        command="\$HOST_MEMBIND accelerate launch --config_file $accelerate_config --num_processes 4 \
    --module src.run_grpo --debug --model qwen3-debug -d $dataset \
    --reward-profile $reward_profile --library-prompt-style $prompt_style \
    -s $seed --run-id qwen3-debug-smoke --output-root $OUTPUT_BASE"
        ;;
    scale)
        # a few steps of the real model: sharding, the vllm weight sync, peak memory and
        # step time. stops before the first epoch ends, so it registers nothing - but the
        # trainer still saves its last step (~100GB) - run the clean mode afterwards
        gpus=4
        time_limit="01:00:00"
        command="\$HOST_MEMBIND accelerate launch --config_file $accelerate_config --num_processes 4 \
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
    clean)
        # deletes what the debug runs leave behind: checkpoints on project storage, their
        # metrics and eval results under output/, and their registry file (debug runs
        # register in config/debug_models.yaml, never alongside real runs). every debug
        # run is named after qwen3-debug or qwen3-8b-scale, which real runs never are.
        # runs here on the login node rather than as a job, and asks first.
        shopt -s nullglob
        candidates=(
            "$OUTPUT_BASE"/qwen3-debug-* "$OUTPUT_BASE"/qwen3-8b-scale
            output/grpo/qwen3-debug-* output/grpo/qwen3-8b-scale
            output/eval/qwen3-debug output/eval/qwen3-debug-* output/eval/qwen3-8b-scale-*
            config/debug_models.yaml config/debug_models.yaml.lock
        )
        leftovers=()
        for path in "${candidates[@]}"; do
            if [ -e "$path" ]; then
                leftovers+=("$path")
            fi
        done

        echo
        echo "Debug run leftovers:"
        if [ ${#leftovers[@]} -gt 0 ]; then
            du -sh "${leftovers[@]}" | sed 's/^/  /'
        else
            echo "  (no files)"
        fi
        echo "  (logs/debug/ is kept)"
        echo

        read -r -p "Delete all of these? [y/N] " answer
        if [ "$answer" != "y" ]; then
            echo "Nothing deleted."
            exit 0
        fi

        if [ ${#leftovers[@]} -gt 0 ]; then
            rm -rf "${leftovers[@]}"
        fi
        echo "Done."
        exit 0
        ;;
    *)
        echo "Usage: bash scripts/submit_debug.sh {smoke|eval|scale|clean} [model-key]"
        echo
        echo "  smoke  4 gpus, ~10 minutes  tiny truncated training run, small model"
        echo "  eval   1 gpu,  ~10 minutes  truncated evaluation of [model-key]"
        echo "  scale  4 gpus, ~30 minutes  a few steps of qwen3-8b, for memory and timing"
        echo "  clean  no job               delete the debug runs' checkpoints, results and registry entries"
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
#SBATCH --nodes=1
#SBATCH --gpus=${gpus}
#SBATCH --time=${time_limit}

source ./scripts/setup_job.sh

$command

echo "Debug run complete: $mode [\$(date '+%Y-%m-%d %H:%M:%S')]"
EOF
)

echo "Submitted job $job_id - follow it with:"
echo "  tail -f logs/debug/${mode}-${job_id}.out"
