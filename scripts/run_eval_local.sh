#!/usr/bin/env bash

echo
echo ">>> Running eval jobs locally (no slurm) >>>"

###############################################################################
#                          evaluation configuration                           #
###############################################################################

# the models to evaluate (untrained or pretrained, see config/models.yaml). local (vllm)
# entries need a GPU node; api-provider entries just need network access - both run fine
# on this node, one after another.
models=(
    # "qwen3-debug"
    # "qwen3-4b"
    # "qwen3-8b"
    # "qwen3-14b"
    # "qwen3-32b"
    # "olmo3-7b"
    # "ocr-7b"
    "gpt-5.6-sol"
    "gpt-5.6-terra"
    "gpt-5.6-luna"
    "deepseek-flash"
    "glm-5.2"
    "kimi-k3"
)

# comma-separated eval dataset names (config/eval_datasets.yaml)
# product_choice is gitignored and must have been built locally.
datasets="library_usage,libreusebench_test,libreusebench_control"

# completions sampled per prompt (see --n-samples in src/run_eval.py); eval-only, unrelated
# to training rollouts. leave blank to use the default of 1.
n_samples=5

# decoding profiles to evaluate with (eval_decoding_profiles in config/training.yaml)
decoding_profiles=(
    # "greedy"
    "default"
    # "qwen3"
)

# system prompt wordings to evaluate with, one run per entry (src/prompts.py::PromptStyle)
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
# already exists under output/ (see src/utils/paths.py), so rerunning only fills in what
# is missing.
update=false

###############################################################################
#                              run jobs                                       #
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
    models=("qwen3-debug")
fi

# fail loudly instead of silently running nothing if `models` is empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi

# same venv/.env/cuda setup the slurm jobs use (see scripts/submit_eval.sh) - skips its
# gpu diagnostics automatically if this node has no GPU driver
source ./scripts/setup_job.sh

# one timestamped directory per invocation, so logs from different runs don't mix
run_timestamp=$(date +%y%m%dT%H%M%S)
log_dir="logs/${run_timestamp}"
mkdir -p "$log_dir"
echo "Log directory: $log_dir"

echo
echo "Evaluating ${#models[@]} models x ${#decoding_profiles[@]} decoding profiles x ${#prompt_styles[@]} prompt styles, sequentially on this node"
echo "  Datasets: ${datasets:-all}"
echo "  Decoding profiles: ${decoding_profiles[*]}"
echo "  Prompt styles: ${prompt_styles[*]}"
echo "  n_samples: ${n_samples:-1 (default)}"
echo

for model in "${models[@]}"; do
    for decoding_profile in "${decoding_profiles[@]}"; do
        for prompt_style in "${prompt_styles[@]}"; do
            echo
            echo "============================================================"
            echo ">>> $model [$decoding_profile] [$prompt_style] >>>"
            echo "============================================================"

            # tee so progress is visible live and also kept per-run, like the slurm jobs'
            # ${log_dir}/eval-<model>-*.out; run_eval's own exit code still drives $? below.
            run_eval \
                -m "$model" \
                $datasets_flag \
                --decoding-profile "$decoding_profile" \
                --prompt-style "$prompt_style" \
                $n_samples_flag \
                $debug_flag \
                $update_flag \
                2>&1 | tee "${log_dir}/eval-${model}-${decoding_profile}-${prompt_style}-local.out"

            if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                echo "FAILED: $model [$decoding_profile] [$prompt_style] - continuing with the rest"
            fi
        done
    done
done

echo
echo "All local eval runs complete!"
