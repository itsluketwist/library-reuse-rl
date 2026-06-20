#!/usr/bin/env bash

echo
echo ">>> Submitting train jobs to slurm >>>"

###############################################################################
#                         hardware constraints                                #
#                                                                             #
# use & for AND, | for OR (e.g., "a100&icelake" or "a100|h200")              #
###############################################################################

GPU_CONSTRAINT="h200"            # e.g., h200, b200, a100, l40s — use | for OR
CPU_MEMORY="96G"                 # cpu memory allocation

###############################################################################
#                          rl training configuration                          #
###############################################################################

# the base models to train (see config/models.yaml)
models=(
    # "qwen2-7b"
    # "qwen3-8b"
)

# training dataset directories (each containing train/val/test.jsonl) — one set of
# jobs is submitted per directory
training_datasets=(
    # "data/skywork"
)

# decoding profile used to generate responses during rl training (from config/decoding.yaml)
decoding_profile="random"

# random seed for training
seed=42

# reward components switched on for this run (see RewardBit in src/response/reward.py)
# comment out a line to switch that component off
reward_bits=(
    "CORRECT_FORMAT"
    "PASSED_TESTS"
    "INTERNAL_LIB_IMPORTED"
    "EXTERNAL_LIB_IMPORTED"
    "INTERNAL_LIB_USED"
    "EXTERNAL_LIB_USED"
)

# comma-separated eval dataset names from config/eval_datasets.yaml, and the decoding
# profile, used for the automatic post-training eval job - leave eval_datasets empty
# to evaluate every dataset in that file
eval_datasets=""
eval_decoding_profile="greedy"

###############################################################################
#                            submit jobs                                      #
###############################################################################

# join the enabled reward bit names into the comma-separated form `train` expects
reward_bits_joined=$(IFS=,; echo "${reward_bits[*]}")
reward_bits_flag="--reward-bits $reward_bits_joined"

# build eval-datasets flag string
eval_datasets_flag=""
if [ -n "$eval_datasets" ]; then
    eval_datasets_flag="-d $eval_datasets"
fi

echo
echo "Training ${#models[@]} models x ${#training_datasets[@]} datasets"
echo "  Decoding profile: $decoding_profile"
echo "  Seed: $seed"
echo "  Reward bits: ${reward_bits_joined:-none}"
echo "  Post-training eval datasets: ${eval_datasets:-all} [$eval_decoding_profile]"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"

    for training_dataset in "${training_datasets[@]}"; do
        dataset_name=$(basename "$training_dataset")

        # call the actual trained_model_path function (rather than reimplementing its
        # naming convention here in bash) so this can never drift out of sync with
        # src/run_train.py - this also means the venv must be active before running
        # this script, since it needs `python` and the `src` package on the path
        trained_model_path=$(python -c "
from src.response.reward import parse_reward_bits
from src.run_train import trained_model_path
print(trained_model_path(
    model='$model',
    dataset='$training_dataset',
    decoding_profile='$decoding_profile',
    reward_bits=parse_reward_bits('$reward_bits_joined'),
    seed=$seed,
))
")

        echo "  Submitting: $dataset_name [$decoding_profile]"

        # submit train job and capture the job id so eval can depend on it
        train_job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=train-${model}-${dataset_name}
#SBATCH --output=logs/train-${model}/${dataset_name}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

train \
    -m $model \
    -d $training_dataset \
    --decoding-profile $decoding_profile \
    -s $seed \
    $reward_bits_flag

echo "RL training complete for model: $model, dataset: $dataset_name"
EOF
)

        # submit eval job to run after training finishes successfully
        sbatch --dependency=afterok:${train_job_id} <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${dataset_name}
#SBATCH --output=logs/eval-${model}/${dataset_name}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

# "command" forces bash to use the installed eval script, not its eval builtin
command eval \
    -m $trained_model_path \
    $eval_datasets_flag \
    --decoding-profile $eval_decoding_profile

echo "Eval complete for model: $trained_model_path"
EOF

        echo "  Submitted train job $train_job_id (eval will follow on completion)"

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
