#!/usr/bin/env bash

echo
echo ">>> Submitting train jobs to slurm >>>"

###############################################################################
#                         cluster settings (isambard)                         #
###############################################################################

ACCOUNT="brics.u6yb"
PARTITION="workq"
GPUS=4                           # one node's gpus; the model is sharded across them
TIME="24:00:00"                  # the workq qos limit; the job requeues and resumes if hit

# checkpoints are large and must outlive the run, so they go to project storage rather
# than $HOME (100GiB) or scratch (purged after 60 days of inactivity)
OUTPUT_BASE="${PROJECTDIR:?PROJECTDIR is not set - are you on isambard?}/$USER/library-reuse-rl"

###############################################################################
#                          rl training configuration                          #
###############################################################################

# the base models to train (see config/models.yaml)
models=(
    # "qwen3-debug"
    "qwen3-8b"
)

# training jobs are dataset|reward-profile|prompt-style. keeping these together avoids
# silently applying a reward or a python-specific system prompt to the wrong task.
training_jobs=(
    "data/train/library_usage|lu-v1|base"
    # "data/train/skywork|full|lib"
)

# grpo training recipe - hyperparameters, including the decoding settings used to
# generate responses during rollout (see config/training.yaml)
recipe_config="config/training.yaml"

# accelerate config that shards the model across the job's gpus
accelerate_config="config/accelerate_configs/fsdp2.yaml"

# random seed for training
seed=42

# every epoch leaves a checkpoint (see save_strategy in the recipe); each is evaluated by
# a dependent job once training finishes. read from the recipe so the two cannot drift.
epochs=$(awk '/^num_train_epochs:/ {print $2}' "$recipe_config")

# true = tiny smoke test instead of full training (see --debug in src/run_grpo.py).
# for the larger debug runs, use scripts/submit_debug.sh instead.
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
echo "  Sharding: $accelerate_config (${GPUS} gpus)"
echo "  Seed: $seed"
echo "  Checkpoints: $OUTPUT_BASE"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"
    mkdir -p "logs/train-$model"

    for training_job in "${training_jobs[@]}"; do
        IFS="|" read -r training_dataset reward_profile library_prompt_style <<< "$training_job"
        dataset_name=$(basename "$training_dataset")

        # prefix the epoch checkpoints are registered under (config/models.yaml), fixed
        # here so the dependent eval jobs below can name them before training finishes
        run_id="${model}-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-4)"

        echo "  Submitting: $dataset_name [$reward_profile] [$library_prompt_style]"

        train_job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=train-${model}-${dataset_name}
#SBATCH --output=logs/train-${model}/${dataset_name}-%j.out
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPUS}
#SBATCH --time=${TIME}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

source ./scripts/setup_job.sh

# requeue on the time-limit warning signal - training resumes from its last epoch checkpoint
trap 'echo "time limit approaching - requeuing \$SLURM_JOB_ID"; scontrol requeue \$SLURM_JOB_ID; exit 0' USR1

# --module runs src.run_grpo as "python -m" would; the flags after it are the script's
accelerate launch --config_file $accelerate_config --num_processes ${GPUS} \
    --module src.run_grpo \
    --model $model \
    -d $training_dataset \
    --recipe-config $recipe_config \
    -s $seed \
    --library-prompt-style $library_prompt_style \
    --output-root ${OUTPUT_BASE} \
    --run-id $run_id \
    --reward-profile $reward_profile \
    $debug_flag &
wait \$!

echo "RL training complete for model: $model, dataset: $dataset_name"
EOF
)

        echo "  Submitted train job $train_job_id"

        # queue one single-gpu eval job per epoch checkpoint, held until training succeeds.
        # submit_eval.sh owns the eval settings (datasets, samples, decoding, prompt style),
        # so they are configured in one place only.
        if [ "$debug" != true ]; then
            eval_models=""
            for epoch in $(seq 1 "$epochs"); do
                eval_models="$eval_models vllm:${run_id}-epoch${epoch}"
            done
            echo "  Queueing eval jobs for $epochs epoch checkpoints (after $train_job_id)"
            EVAL_MODELS="$eval_models" \
                EVAL_DEPENDENCY="afterok:${train_job_id}" \
                bash scripts/submit_eval.sh
        fi

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
