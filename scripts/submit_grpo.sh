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

# trained models (checkpoints) are large - save them to scratch storage rather than
# the repo's own (quota-limited) filesystem. passed through to run_grpo's
# --output-root flag (see src/run_grpo.py::trained_model_path)
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

# training dataset directories (each containing train/val/test.jsonl) — one set of
# jobs is submitted per directory
training_datasets=(
    "data/skywork"
)

# grpo training recipe - hyperparameters, including the decoding settings used to
# generate responses during rollout (see config/train/demo.yaml)
recipe_config="config/train/demo.yaml"

# wording used for the library system prompt during training, when recipe_config's
# libraries_requirements_file is set (see src/prompts.py::PromptStyle): "lib" just
# lists the available libraries, "use" also pushes the model to prefer them over
# writing its own implementations. passed straight through to run_grpo's
# --library-prompt-style flag (see src/run_grpo.py) - not part of the recipe file,
# so switching this doesn't need a second recipe file
library_prompt_style="lib"
# library_prompt_style="use"

# random seed for training
seed=42

# set true to run a tiny smoke test instead of full training - truncates the dataset
# to a handful of samples, to quickly check the training code runs correctly
# end-to-end (see the --debug flag in src/run_grpo.py) before submitting a real job.
# if `models` below is left empty, debug runs default to the tiny "qwen3-debug" model
# (see DEBUG_MODEL_KEY in src/run_grpo.py) - set an explicit model there to smoke-test
# a bigger one instead
debug=false

# reward profile to use for this run (see config/rewards.yaml)
reward_profile="full"

# comma-separated eval dataset names from config/eval_datasets.yaml, and the decoding
# profile, used for the automatic post-training eval job - leave eval_datasets empty
# to evaluate every dataset in that file
eval_datasets=""
eval_decoding_profile="greedy"

###############################################################################
#                            submit jobs                                      #
###############################################################################

reward_profile_flag="--reward-profile $reward_profile"

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
echo "  Reward profile: $reward_profile"
echo "  Library prompt style: $library_prompt_style"
echo "  Post-training eval datasets: ${eval_datasets:-all} [$eval_decoding_profile]"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"

    for training_dataset in "${training_datasets[@]}"; do
        dataset_name=$(basename "$training_dataset")

        # registry key this run's trained model gets registered under (see
        # config/models.yaml) - generated once here and reused for the dependent eval
        # jobs' --model below, so nothing needs to reconstruct run_grpo's naming
        # convention just to find the model it just trained
        run_id="${model}-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-4)"

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
    --library-prompt-style $library_prompt_style \
    --output-root ${OUTPUT_BASE} \
    --run-id $run_id \
    $reward_profile_flag \
    $debug_flag

echo "RL training complete for model: $model, dataset: $dataset_name"
EOF
)

        # submit follow-up eval jobs to run after training finishes successfully - one
        # per library prompt style ("lib"/"use" - see src/prompts.py::PromptStyle).
        # libraries_requirements_file itself isn't passed here: run_eval always reads
        # it straight from the recipe file, so it automatically matches this training
        # run. results don't clash: run_eval adds a distinct suffix to output file
        # names per style (see src/run_eval.py::eval_results_path)
        for eval_variant in "with-libs" "with-uses"; do
            if [ "$eval_variant" = "with-libs" ]; then
                variant_style_flag="--prompt-style lib"
            else
                variant_style_flag="--prompt-style use"
            fi

            sbatch --dependency=afterok:${train_job_id} <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${dataset_name}-${eval_variant}
#SBATCH --output=logs/eval-${model}/${dataset_name}-${eval_variant}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

run_eval \
    -m $run_id \
    $eval_datasets_flag \
    --decoding-profile $eval_decoding_profile \
    $variant_style_flag

echo "Eval ($eval_variant) complete for model: $run_id"
EOF

        done

        echo "  Submitted train job $train_job_id (with-libs + with-uses evals will follow on completion)"

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
