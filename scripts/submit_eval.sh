#!/usr/bin/env bash

echo
echo ">>> Submitting eval jobs to slurm >>>"

###############################################################################
#                         cluster settings (isambard)                         #
###############################################################################

ACCOUNT="brics.u6yb"
PARTITION="workq"
GPUS=1                           # evaluation generates with one vllm engine
TIME="12:00:00"

###############################################################################
#                          evaluation configuration                           #
###############################################################################

# the models to evaluate, prefixed with their provider (matching config/models.yaml's
# "provider" field - "vllm" for local weights, whether a released model or one of a
# training run's epoch checkpoints).
#
# api-served models are not submitted here: isambard has no cpu partition, and they only
# need network access - run them with scripts/run_eval_local.sh instead.
models=(
    # "vllm:qwen3-debug"
    "vllm:qwen3-4b"
    "vllm:qwen3-8b"
    "vllm:qwen3-14b"
    # "vllm:qwen3-32b"  # both ran out of the 12h limit on libreusebench_v2_full
    # "vllm:olmo3-7b"
    "vllm:qwen3.5-9b"
    "vllm:domyn-small"
)

# training runs to evaluate, by run id (e.g. "qwen3-8b-4796"). each is added to the models
# above as every epoch checkpoint that run registered (config/trained_models.yaml, or
# config/debug_models.yaml for a debug run), so a run's epochs needn't be listed one by one
training_runs=(
    "qwen3-8b-4796"
    "qwen3-8b-e650"
)

# EVAL_MODELS replaces both lists above (a space-separated string of models), which is how
# scripts/submit_grpo.sh queues a run's epochs before training has registered them;
# EVAL_DATASETS likewise replaces the datasets below, so a run's epochs are always scored on
# the datasets submit_grpo.sh names, whatever this file is set up for; EVAL_DEPENDENCY holds
# those jobs until training succeeds, and EVAL_LOG_DIR puts their logs in the training run's
# log directory.

# eval dataset names (config/eval_datasets.yaml). leave empty to evaluate every dataset in
# that file. product_choice is gitignored and must have been built locally.
datasets=(
    # "library_usage"
    # "library_usage_val"
    # "library_usage_train"
    # "library_usage_script_pilot"
    # "libreusebench_test"
    # "libreusebench_control"
    # "libreusebench_function_test"
    # "libreusebench_function_control"
    # "libreusebench_v2_pilot"
    # "libreusebench_v2_pilot_control"
    # "libreusebench_v2_pilot2"
    # "libreusebench_v2_pilot3"
    # "libreusebench_v2_pilot3_25"
    # "libreusebench_v2_pilot4"
    # "libreusebench_v2_pilot5"
    # "libreusebench_v2_full"
    # "libreusebench_v2_test"  # the same tasks as libreusebench_v2_full, whose results carry over
    "libreusebench_v2_control"
)

# completions sampled per prompt (see --n-samples in src/run_eval.py); eval-only, unrelated
# to training rollouts. leave blank to use the default of 1.
n_samples=5

# decoding profiles to evaluate with (eval_decoding_profiles in config/training.yaml)
decoding_profiles=(
    # "greedy"
    # "default"
    "default_16k"
    # "qwen3"
)

# system prompt wordings to evaluate with, one job per entry (src/prompts.py::PromptStyle)
prompt_styles=(
    # "lib"
    # "use"
    # "std"
    # "ext"
    "base"
)

# true = tiny smoke test instead of full evaluation (see --debug in src/run_eval.py)
debug=false

# true = regenerate results even if already saved (see --update in src/run_eval.py).
# false skips a model/dataset/decoding-profile/prompt-style combination whose result file
# already exists under output/ (see src/utils/paths.py), so resubmitting only fills in
# what is missing.
update=false

###############################################################################
#                      submit jobs - do not edit below                        #
###############################################################################

# EVAL_DATASETS names the datasets outright (see above)
if [ -n "$EVAL_DATASETS" ]; then
    read -ra datasets <<< "$EVAL_DATASETS"
fi

# build datasets flag string, joining the array into the comma-separated list run_eval
# expects (-d)
datasets_flag=""
if [ ${#datasets[@]} -gt 0 ]; then
    datasets_flag="-d $(IFS=,; echo "${datasets[*]}")"
fi

# build debug flag string
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# build update flag string
update_flag=""
if [ "$update" = true ]; then
    update_flag="--update"
fi

# build n-samples flag string
n_samples_flag=""
if [ -n "$n_samples" ]; then
    n_samples_flag="--n-samples $n_samples"
fi

# add each training run's registered epoch checkpoints to the models to evaluate, in epoch
# order - unless EVAL_MODELS names the models outright (see above)
if [ -n "$EVAL_MODELS" ]; then
    read -ra models <<< "$EVAL_MODELS"
else
    for run_id in "${training_runs[@]}"; do
        # registry keys start their line, as "{run_id}-epoch{n}:"
        epoch_keys=$(cat config/trained_models.yaml config/debug_models.yaml 2>/dev/null \
            | grep -oE "^${run_id}-epoch[0-9]+" | sort -V)
        if [ -z "$epoch_keys" ]; then
            echo "No registered epochs found for training run '$run_id' - check its id in config/trained_models.yaml."
            exit 1
        fi
        for epoch_key in $epoch_keys; do
            models+=("vllm:$epoch_key")
        done
    done
fi

# default to the tiny debug model if smoke-testing with no model set
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("vllm:qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if `models` is empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi

# one timestamped directory per invocation, so logs from different submissions don't mix -
# unless queued by scripts/submit_grpo.sh, which passes its own (EVAL_LOG_DIR) so the eval
# logs sit beside the training log they follow
run_timestamp=$(date +%y%m%dT%H%M%S)
log_dir="${EVAL_LOG_DIR:-logs/${run_timestamp}}"
mkdir -p "$log_dir"
echo "Log directory: $log_dir"

echo
echo "Evaluating ${#models[@]} models x ${#decoding_profiles[@]} decoding profiles x ${#prompt_styles[@]} prompt styles"
echo "  Datasets: ${datasets[*]:-all}"
echo "  Decoding profiles: ${decoding_profiles[*]}"
echo "  Prompt styles: ${prompt_styles[*]}"
echo "  n_samples: ${n_samples:-1 (default)}"
echo

for model_entry in "${models[@]}"; do
    provider="${model_entry%%:*}"
    model="${model_entry#*:}"

    echo "Submitting jobs for $model"

    if [ "$provider" != "vllm" ]; then
        echo "Unknown provider '$provider' for model entry '$model_entry' - expected 'vllm:<model>'."
        echo "API-served models run with scripts/run_eval_local.sh, not on slurm."
        exit 1
    fi

    # held until the training job it depends on succeeds, when submit_grpo.sh sets it
    dependency_directive=""
    if [ -n "$EVAL_DEPENDENCY" ]; then
        dependency_directive="#SBATCH --dependency=${EVAL_DEPENDENCY}"
    fi

    for decoding_profile in "${decoding_profiles[@]}"; do
        for prompt_style in "${prompt_styles[@]}"; do
            echo "  Submitting: [$decoding_profile] [$prompt_style]"

            eval_job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=reuse-eval-${model}-${decoding_profile}-${prompt_style}
#SBATCH --output=${log_dir}/eval-${model}-${decoding_profile}-${prompt_style}-%j.out
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus=${GPUS}
#SBATCH --time=${TIME}
${dependency_directive}

source ./scripts/setup_job.sh

run_eval \
    -m $model \
    $datasets_flag \
    --decoding-profile $decoding_profile \
    --prompt-style $prompt_style \
    $n_samples_flag \
    $debug_flag \
    $update_flag

echo "Eval complete for model: $model, decoding: $decoding_profile, prompt style: $prompt_style [\$(date '+%Y-%m-%d %H:%M:%S')]"
EOF
)
            echo "    Submitted eval job $eval_job_id - follow it with:"
            echo "      tail -f ${log_dir}/eval-${model}-${decoding_profile}-${prompt_style}-${eval_job_id}.out"

            # small delay between submissions to avoid scheduler overload
            sleep 0.5
        done
    done

    echo
done

echo "All jobs submitted!"
