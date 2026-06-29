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
    # "qwen3-8b"
)

# training dataset directories (each containing train/val/test.jsonl) — one set of
# jobs is submitted per directory
training_datasets=(
    "data/skywork"
)

# grpo training recipe - hyperparameters, including the decoding settings used to
# generate responses during rollout (see config/train/demo.yaml)
recipe_config="config/train/demo.yaml"

# random seed for training
seed=42

# set true to run a tiny smoke test instead of full training - truncates the dataset
# to a handful of samples, to quickly check the training code runs correctly
# end-to-end (see the --debug flag in src/run_grpo.py) before submitting a real job.
# if `models` below is left empty, debug runs default to the tiny "qwen3-debug" model
# (see DEBUG_MODEL_KEY in src/run_grpo.py) - set an explicit model there to smoke-test
# a bigger one instead
debug=false

# reward components switched on for this run (see RewardBit in src/grpo/rewards.py)
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

# join the enabled reward bit names into the comma-separated form `run_grpo` expects
reward_bits_joined=$(IFS=,; echo "${reward_bits[*]}")
reward_bits_flag="--reward-bits $reward_bits_joined"

# build eval-datasets flag string
eval_datasets_flag=""
if [ -n "$eval_datasets" ]; then
    eval_datasets_flag="-d $eval_datasets"
fi

# build debug flag string
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# default to the tiny debug model if smoke-testing and no model was set, mirroring
# run_grpo.py's own --debug default (see DEBUG_MODEL_KEY in src/run_grpo.py) - without
# this, debug=true with an empty `models` array would silently submit nothing
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if the arrays above are empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi
if [ ${#training_datasets[@]} -eq 0 ]; then
    echo "No training datasets configured - uncomment/add at least one entry in the training_datasets array."
    exit 1
fi

echo
echo "Training ${#models[@]} models x ${#training_datasets[@]} datasets"
echo "  Recipe: $recipe_config"
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
        # src/run_grpo.py - this also means the venv must be active before running
        # this script, since it needs `python` and the `src` package on the path
        trained_model_path=$(python -c "
from src.grpo.rewards import parse_reward_bits
from src.run_grpo import trained_model_path
print(trained_model_path(
    model='$model',
    dataset='$training_dataset',
    recipe_config='$recipe_config',
    reward_bits=parse_reward_bits('$reward_bits_joined'),
    seed=$seed,
))
")

        echo "  Submitting: $dataset_name [$recipe_config]"

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

run_grpo \
    -m $model \
    -d $training_dataset \
    --recipe-config $recipe_config \
    -s $seed \
    $reward_bits_flag \
    $debug_flag

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

run_eval \
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
