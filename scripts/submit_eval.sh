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

# the models to evaluate (untrained or pretrained, see config/models.yaml). api-provider
# entries (deepseek-v4-flash, gpt-5.6-luna) run on the cpu partition, see model_provider below.
models=(
    # "qwen3-debug"
    "qwen3-4b"
    "qwen3-8b"
    # "deepseek-v4-flash"
    # "gpt-5.6-luna"
)

# comma-separated eval dataset names (config/eval_datasets.yaml)
# product_choice is gitignored and must have been built locally.
datasets="library_usage,libreusebench"

# decoding profiles to evaluate with (eval_decoding_profiles in config/training.yaml)
decoding_profiles=(
    "greedy"
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

###############################################################################
#                            submit jobs                                      #
###############################################################################

# build datasets flag string
datasets_flag=""
if [ -n "$datasets" ]; then
    datasets_flag="-d $datasets"
fi

# build debug flag string
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# default to the tiny debug model if smoke-testing with no model set
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("qwen3-debug")
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

# look up a model's provider from config/models.yaml, defaulting to "vllm" the same way
# src/run_eval.py does for entries that omit the key (every local/trained model)
model_provider() {
    .venv/bin/python -c '
import sys

import yaml

with open("config/models.yaml") as f:
    config = yaml.safe_load(f)
print(config[sys.argv[1]].get("provider", "vllm"))
' "$1"
}

echo
echo "Evaluating ${#models[@]} models x ${#decoding_profiles[@]} decoding profiles x ${#prompt_styles[@]} prompt styles"
echo "  Datasets: ${datasets:-all}"
echo "  Decoding profiles: ${decoding_profiles[*]}"
echo "  Prompt styles: ${prompt_styles[*]}"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"
    mkdir -p "${log_dir}/eval-$model"

    provider=$(model_provider "$model")
    if [ "$provider" = "api" ]; then
        partition="$CPU_PARTITION"
        gpu_directives=""
        echo "  Provider: api -> partition=$CPU_PARTITION (no GPU requested)"
    else
        partition="gpu"
        gpu_directives=$'#SBATCH --gres=gpu\n#SBATCH --gpus=1\n#SBATCH --constraint='"${GPU_CONSTRAINT}"
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
    --run-timestamp ${run_timestamp} \
    $debug_flag

echo "Eval complete for model: $model, decoding: $decoding_profile, prompt style: $prompt_style"
EOF

            # small delay between submissions to avoid scheduler overload
            sleep 0.5
        done
    done

    echo
done

echo "All jobs submitted!"
