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

###############################################################################
#                          evaluation configuration                           #
###############################################################################

# the models to evaluate (untrained or pretrained, see config/models.yaml)
models=(
    # "qwen3-debug"
    "qwen3-4b"
    "qwen3-8b"
)

# comma-separated eval dataset names from config/eval_datasets.yaml, all evaluated
# within a single job per model - leave empty to evaluate every dataset in that file
datasets=""

# decoding profiles to evaluate with (from eval_decoding_profiles in the training
# recipe file, config/train/demo.yaml - shared between training and inference)
decoding_profiles=(
    "greedy"
)

# wording used for the library system prompt (see src/prompts.py::PromptStyle) -
# libraries_requirements_file itself always comes from the recipe file's own setting
# (config/train/demo.yaml), not a flag here, so eval always matches what a model was
# trained on. "lib" just lists the available libraries; "use" also pushes the model
# to prefer them over writing its own implementations; "base" skips the system prompt
# entirely, ignoring the recipe file's setting (see src/prompts.py::build_system_prompt)
prompt_style="lib"
prompt_style="use"
prompt_style="base"

# set true to run a tiny smoke test instead of full evaluation - truncates each
# dataset to a handful of samples, to quickly check the eval code runs correctly
# end-to-end (see the --debug flag in src/run_eval.py) before submitting a real job.
# if `models` above is left empty, debug runs default to the tiny "qwen3-debug" model
# (see DEBUG_MODEL_KEY in src/run_eval.py) - set an explicit model there to smoke-test
# a bigger one instead
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

style_flag="--prompt-style $prompt_style"

# default to the tiny debug model if smoke-testing and no model was set, mirroring
# run_eval.py's own --debug default (see DEBUG_MODEL_KEY in src/run_eval.py) - without
# this, debug=true with an empty `models` array would silently submit nothing
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if `models` is empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi

echo
echo "Evaluating ${#models[@]} models x ${#decoding_profiles[@]} decoding profiles"
echo "  Datasets: ${datasets:-all}"
echo "  Decoding profiles: ${decoding_profiles[*]}"
echo "  Prompt style: $prompt_style"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"

    for decoding_profile in "${decoding_profiles[@]}"; do
        echo "  Submitting: [$decoding_profile]"

        sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${decoding_profile}
#SBATCH --output=logs/eval-${model}/${decoding_profile}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

run_eval \
    -m $model \
    $datasets_flag \
    --decoding-profile $decoding_profile \
    $style_flag \
    $debug_flag

echo "Eval complete for model: $model, decoding: $decoding_profile"
EOF

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
