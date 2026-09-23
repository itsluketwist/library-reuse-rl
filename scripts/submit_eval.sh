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
#
# EVAL_MODELS overrides this list (a space-separated string), which is how
# scripts/submit_grpo.sh queues its epoch checkpoints without duplicating the settings
# below; EVAL_DEPENDENCY holds those jobs until training succeeds.
models=(
    # "vllm:qwen3-debug"
    "vllm:qwen3-4b"
    "vllm:qwen3-8b"
    "vllm:qwen3-14b"
    "vllm:qwen3-32b"
    "vllm:olmo3-7b"
    "vllm:ocr-7b"
)
if [ -n "$EVAL_MODELS" ]; then
    read -ra models <<< "$EVAL_MODELS"
fi

# eval dataset names (config/eval_datasets.yaml). leave empty to evaluate every dataset in
# that file. product_choice is gitignored and must have been built locally.
datasets=(
    "library_usage"
    "libreusebench_test"
    "libreusebench_control"
)

# completions sampled per prompt (see --n-samples in src/run_eval.py); eval-only, unrelated
# to training rollouts. leave blank to use the default of 1.
n_samples=5

# decoding profiles to evaluate with (eval_decoding_profiles in config/training.yaml)
decoding_profiles=(
    # "greedy"
    "default"
    # "qwen3"
)

# system prompt wordings to evaluate with, one job per entry (src/prompts.py::PromptStyle)
prompt_styles=(
    # "lib"
    # "use"
    # "std"
    "base"
)

# true = tiny smoke test instead of full evaluation (see --debug in src/run_eval.py)
debug=false

# true = regenerate results even if already saved (see --update in src/run_eval.py).
# false skips a model/dataset/decoding-profile/prompt-style combination whose result file
# already exists under output/eval/, so resubmitting only fills in what is missing.
update=false

###############################################################################
#                      submit jobs - do not edit below                        #
###############################################################################

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

# default to the tiny debug model if smoke-testing with no model set
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("vllm:qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if `models` is empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi

# one timestamped directory per invocation, so logs from different submissions don't mix
run_timestamp=$(date +%y%m%dT%H%M%S)
log_dir="logs/${run_timestamp}"
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
    mkdir -p "${log_dir}/eval-$model"

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

            sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${decoding_profile}-${prompt_style}
#SBATCH --output=${log_dir}/eval-${model}/${decoding_profile}-${prompt_style}-%j.out
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

            # small delay between submissions to avoid scheduler overload
            sleep 0.5
        done
    done

    echo
done

echo "All jobs submitted!"
