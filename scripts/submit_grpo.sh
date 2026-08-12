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

# checkpoints are large - use scratch storage, not the repo's own filesystem
OUTPUT_BASE="/scratch/users/$USER/library-reuse-rl"

###############################################################################
#                          rl training configuration                          #
###############################################################################

# the base models to train (see config/models.yaml)
models=(
    # "qwen3-debug"
    "qwen3-4b"
    "qwen3-8b"
)

# training jobs are dataset|reward-profile|prompt-style. keeping these together avoids
# silently applying a reward or a python-specific system prompt to the wrong task.
training_jobs=(
    # "data/train/skywork|full|lib"
    "data/train/library_usage|lu-v1|lib"
)

# grpo training recipe - hyperparameters, including the decoding settings used to
# generate responses during rollout (see config/training.yaml)
recipe_config="config/training.yaml"

# random seed for training
seed=42

# true = tiny smoke test instead of full training (see --debug in src/run_grpo.py)
debug=false

###############################################################################
#                            submit jobs                                      #
###############################################################################

# build debug flag string
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# default to the tiny debug model if smoke-testing with no model set
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if the arrays above are empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi
if [ ${#training_jobs[@]} -eq 0 ]; then
    echo "No training jobs configured - add at least one entry to the training_jobs array."
    exit 1
fi
for training_job in "${training_jobs[@]}"; do
    IFS="|" read -r training_dataset _reward_profile _prompt_style <<< "$training_job"
    if [ ! -f "$training_dataset/train.jsonl" ] || \
        [ ! -f "$training_dataset/val.jsonl" ]; then
        echo "Training dataset is missing train.jsonl or val.jsonl: $training_dataset"
        exit 1
    fi
done

echo
echo "Training ${#models[@]} models x ${#training_jobs[@]} dataset configurations"
echo "  Recipe: $recipe_config"
echo "  Seed: $seed"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"
    mkdir -p "logs/train-$model"

    for training_job in "${training_jobs[@]}"; do
        IFS="|" read -r training_dataset reward_profile library_prompt_style <<< "$training_job"
        dataset_name=$(basename "$training_dataset")

        # registry key this run's trained model gets registered under (config/models.yaml)
        run_id="${model}-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-4)"

        echo "  Submitting: $dataset_name [$reward_profile] [$library_prompt_style]"

        train_job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=train-${model}-${dataset_name}
#SBATCH --output=logs/train-${model}/${dataset_name}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

source ./scripts/setup_job.sh

# requeue on the time-limit warning signal - training resumes from its last checkpoint
trap 'echo "time limit approaching - requeuing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

run_grpo \
    -m $model \
    -d $training_dataset \
    --recipe-config $recipe_config \
    -s $seed \
    --library-prompt-style $library_prompt_style \
    --output-root ${OUTPUT_BASE} \
    --run-id $run_id \
    --reward-profile $reward_profile \
    $debug_flag &
wait $!

echo "RL training complete for model: $model, dataset: $dataset_name"
EOF
)

        echo "  Submitted train job $train_job_id"

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
