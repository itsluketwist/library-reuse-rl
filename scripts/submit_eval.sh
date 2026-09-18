#!/usr/bin/env bash

echo
echo ">>> Submitting eval jobs to slurm >>>"

###############################################################################
#                         hardware constraints                                #
#                                                                             #
# use & for AND, | for OR (e.g., "a100&icelake" or "a100|h200")              #
###############################################################################

GPU_CONSTRAINT="h200"            # e.g., h200, b200, a100, l40s — use | for OR
CPU_MEMORY="96G"                 # cpu memory allocation

# api-served models (config/models.yaml "provider: api" — DeepSeek, GPT, ...) make network
# calls instead of running local inference, so they're submitted to this cpu partition
# instead of the gpu one below, and skip the gpu directives entirely.
CPU_PARTITION="cpu"

###############################################################################
#                          evaluation configuration                           #
###############################################################################

# the models to evaluate, prefixed with their provider (matching config/models.yaml's
# "provider" field - "vllm" for local weights, "api" for api-served models). api-provider
# entries run on the cpu partition instead of the gpu one below - see partition selection.
models=(
    ################################################
    #### local vllm models (require a GPU node) ####
    ################################################
    # "vllm:qwen3-debug"
    "vllm:qwen3-4b"
    "vllm:qwen3-8b"
    "vllm:qwen3-14b"
    "vllm:qwen3-32b"
    "vllm:olmo3-7b"
    "vllm:ocr-7b"

    ####################################################
    #### api-served models (require network access) ####
    ####################################################
    "api:gpt-5.6-sol"
    "api:gpt-5.6-terra"
    "api:gpt-5.6-luna"
    "api:deepseek-flash"
    "api:glm-5.2"
    "api:kimi-k3"
)

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

    if [ "$provider" = "api" ]; then
        partition="$CPU_PARTITION"
        gpu_directives=""
        echo "  Provider: api -> partition=$CPU_PARTITION (no GPU requested)"
    elif [ "$provider" = "vllm" ]; then
        partition="gpu"
        gpu_directives=$'#SBATCH --gres=gpu\n#SBATCH --gpus=1\n#SBATCH --constraint='"${GPU_CONSTRAINT}"
    else
        echo "Unknown provider '$provider' for model entry '$model_entry' - expected 'vllm:<model>' or 'api:<model>'."
        exit 1
    fi

    for decoding_profile in "${decoding_profiles[@]}"; do
        for prompt_style in "${prompt_styles[@]}"; do
            echo "  Submitting: [$decoding_profile] [$prompt_style]"

            sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${decoding_profile}-${prompt_style}
#SBATCH --output=${log_dir}/eval-${model}/${decoding_profile}-${prompt_style}-%j.out
#SBATCH --partition=${partition}
#SBATCH --mem=${CPU_MEMORY}
${gpu_directives}

source ./scripts/setup_job.sh

run_eval \
    -m $model \
    $datasets_flag \
    --decoding-profile $decoding_profile \
    --prompt-style $prompt_style \
    $n_samples_flag \
    $debug_flag \
    $update_flag

echo "Eval complete for model: $model, decoding: $decoding_profile, prompt style: $prompt_style"
EOF

            # small delay between submissions to avoid scheduler overload
            sleep 0.5
        done
    done

    echo
done

echo "All jobs submitted!"
