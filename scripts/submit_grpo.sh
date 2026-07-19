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
    "qwen-debug"
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

# requirements file for the automatic post-training eval jobs - match this to
# recipe_config's libraries_requirements_file (see config/train/demo.yaml) so the
# trained model is evaluated on the same prompt it was trained on. two eval jobs
# are submitted per training run: one with this file in the system prompt, and one
# without any system prompt - leave empty ("") to only submit the no-prompt eval
eval_libraries_requirements_file="setup/requirements/v1.txt"

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

# build the post-training eval job's libraries flag string
eval_libraries_flag=""
if [ -n "$eval_libraries_requirements_file" ]; then
    eval_libraries_flag="--libraries-requirements-file $eval_libraries_requirements_file"
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
echo "  Post-training eval datasets: ${eval_datasets:-all} [$eval_decoding_profile]"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"

    for training_dataset in "${training_datasets[@]}"; do
        dataset_name=$(basename "$training_dataset")

        # mirror src/run_grpo.py::trained_model_path's naming convention here in bash,
        # since python/the venv aren't available on the submission node (only inside
        # the sbatch job, after scripts/setup_job.sh runs) - keep in sync with that
        # function if its naming convention ever changes
        recipe_name="$(basename "$recipe_config")"
        recipe_name="${recipe_name%.*}"

        # only include the seed when it differs from run_grpo.py's DEFAULT_SEED (42),
        # to keep names short
        seed_suffix=""
        if [ "$seed" -ne 42 ]; then
            seed_suffix="-seed${seed}"
        fi

        # append e.g. "-libv1" when a libraries requirements file is set, matching the
        # recipe's own libraries_requirements_file setting (kept in sync manually via
        # eval_libraries_requirements_file above), so runs with and without the
        # library prompt produce distinct output directories
        lib_suffix=""
        if [ -n "$eval_libraries_requirements_file" ]; then
            lib_name="$(basename "$eval_libraries_requirements_file")"
            lib_name="${lib_name%.*}"
            lib_suffix="-lib${lib_name}"
        fi

        trained_model_path="output/${model}/${dataset_name}-${recipe_name}-${reward_profile}${seed_suffix}${lib_suffix}"

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
    $reward_profile_flag \
    $debug_flag

echo "RL training complete for model: $model, dataset: $dataset_name"
EOF
)

        # submit follow-up eval jobs to run after training finishes successfully -
        # one with the library requirements in the system prompt, one without.
        # results don't clash: run_eval adds a "-lib" suffix to output file names
        # when a requirements file is set (see src/run_eval.py::eval_results_path)
        for eval_variant in "with-libs" "no-libs"; do

            # only the "with-libs" variant passes the libraries flag
            variant_libraries_flag=""
            if [ "$eval_variant" = "with-libs" ]; then
                # skip the "with-libs" variant entirely when no requirements file
                # is configured, since it would duplicate the "no-libs" job
                if [ -z "$eval_libraries_requirements_file" ]; then
                    continue
                fi
                variant_libraries_flag="$eval_libraries_flag"
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
    -m $trained_model_path \
    $eval_datasets_flag \
    --decoding-profile $eval_decoding_profile \
    $variant_libraries_flag

echo "Eval ($eval_variant) complete for model: $trained_model_path"
EOF

        done

        echo "  Submitted train job $train_job_id (with-libs + no-libs evals will follow on completion)"

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
