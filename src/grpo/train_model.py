# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GRPO training loop, wired to this project's reward and dataset format."""

import logging
import os
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path

import datasets
import transformers
from transformers import AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from trl import GRPOTrainer, TrlParser

from src.data import load_train_dataset
from src.grpo.checkpoints import epoch_checkpoints, prune_resume_state
from src.grpo.config import GRPOConfig
from src.grpo.rewards import build_reward_functions, infer_reward_kind
from src.grpo.wandb_logging import init_wandb_training
from src.prompts import PromptStyle, build_system_prompt
from src.utils.config import check_config_keys, load_full_yaml
from src.utils.io import save_json
from src.utils.log import log_gpu_memory, log_timer


logger = logging.getLogger(__name__)

# constants used to configure src/run_grpo.py --debug mode
DEBUG_TRAIN_SAMPLES = 16
DEBUG_VAL_SAMPLES = 4
# long enough for qwen3's reasoning to finish, so debug rollouts actually earn rewards
DEBUG_MAX_COMPLETION_LENGTH = 2048
DEBUG_EPOCHS = 2
# 1 completion per gpu x 4 gpus x 4 = 16 completions (2 prompts) per step, so a handful
# of prompts still make whole steps (see config/training.yaml)
DEBUG_GRADIENT_ACCUMULATION_STEPS = 4

# recipe keys this file (and src/run_eval.py) read directly rather than trl - every other
# key must name a real training setting, see check_config_keys below
_EVAL_ONLY_RECIPE_KEYS = frozenset(
    {"eval_decoding_profiles", "rollout_decoding_profile"}
)


class SaveTrainerStateCallback(TrainerCallback):
    """
    Overwrites metrics_dir/trainer_state.json with the training history so far, on the
    checkpointing cadence (`save_steps`), so a crash doesn't lose the metric history
    since the last save. Written to metrics_dir (tracked) not output_dir (scratch).
    """

    def __init__(self, metrics_dir: str | Path) -> None:
        self.metrics_dir = metrics_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            state.save_to_json(os.path.join(self.metrics_dir, "trainer_state.json"))


class PruneResumeStateCallback(TrainerCallback):
    """
    Keeps only the newest checkpoint resumable, as each epoch's checkpoint is saved.

    See prune_resume_state - this runs it on rank zero after every save.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            prune_resume_state(args.output_dir)


class LogGPUMemoryCallback(TrainerCallback):
    """
    Logs gpu memory usage after the first optimizer step - by then the policy and
    reference models, optimizer states, and activation memory have all been
    allocated, giving the truest picture of the run's steady-state memory use.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero and state.global_step == 1:
            log_gpu_memory("after first optimizer step", device=args.device.index)


def _periodic_eval_history(log_history: list[dict]) -> dict[str, dict]:
    """
    Pull the periodic validation rounds out of trainer.state.log_history.

    Returns a dict mapping step (as a string) -> that round's metrics.
    """
    return {
        str(entry["step"]): {k: v for k, v in entry.items() if k != "step"}
        for entry in log_history
        if any(key.startswith("eval_") for key in entry)
    }


def _fix_vllm_kv_cache_size(kv_cache_memory_gib: float) -> None:
    """
    Make trl's colocated vllm engine use a fixed-size kv cache.

    At startup vllm measures the gpu's free memory, runs a dummy forward pass, and
    asserts free memory did not go up in between. In colocate mode the training model
    shares the process, and gpu memory it frees mid-profile (e.g. fsdp's weight
    broadcast buffers) trips that assertion (vllm issue #56830, unfixed in 0.28).
    Giving vllm a fixed cache size skips the check. trl exposes no setting for this, so
    the engine class trl builds is swapped for one with the size filled in.
    """
    from trl.generation import vllm_generation

    vllm_generation.LLM = partial(
        vllm_generation.LLM,
        kv_cache_memory_bytes=int(kv_cache_memory_gib * 1024**3),
    )


def train_model_with_grpo(
    model_name_or_path: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    metrics_dir: str | Path,
    reward_profile: str,
    reward_weights: dict[str, float],
    seed: int,
    recipe_config: str = "config/training.yaml",
    library_prompt_style: PromptStyle = "lib",
    debug: bool = False,
    epochs: int | None = None,
    max_steps: int | None = None,
) -> dict[int, Path]:
    """
    Run full-weight GRPO training for `model_name_or_path` against the dataset at
    `dataset_dir`, saving a checkpoint per epoch under `output_dir` (large, typically
    project storage) and its metrics to `metrics_dir` (small, typically the repo).

    The model is trained across every gpu in the job, sharded with FSDP2 (see
    config/accelerate_configs/fsdp2.yaml); rollouts are generated by a vllm engine
    colocated on those same gpus.

    `seed` and `library_prompt_style` override the corresponding `recipe_config` fields
    (resolved once by the caller), and `epochs`/`max_steps` override the recipe's run
    length when given (see the debug modes in scripts/submit_debug.sh); every other
    hyperparameter, including libraries_requirements_file and the rollout decoding
    settings, comes from `recipe_config`.

    `reward_profile` and `reward_weights` (see config/rewards.yaml) are recorded into
    train_results.json, so the reward function stays recoverable from the metrics dir.

    If `debug` is set, the train/val splits are truncated to DEBUG_TRAIN_SAMPLES /
    DEBUG_VAL_SAMPLES rows for a quick end-to-end smoke test.

    Returns each epoch number mapped to the checkpoint directory it ended at.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    recipe = load_full_yaml(recipe_config)
    # the recipe is parsed with allow_extra_keys below, since it also carries eval-only
    # keys trl ignores - which also means a misspelled or renamed setting would be
    # dropped without a word, so check every key is one something reads first
    check_config_keys(
        config=recipe,
        known={field.name for field in fields(GRPOConfig)} | _EVAL_ONLY_RECIPE_KEYS,
        source=recipe_config,
    )

    # settings that replace the recipe's for this run. they are merged in before parsing,
    # not set on the parsed config afterwards, because trl derives other settings from
    # them while parsing (e.g. its rollout batch size from gradient_accumulation_steps)
    overrides: dict = {
        "output_dir": str(output_dir),
        # the wandb run name - the run id the checkpoint folder and registry entries share
        "run_name": Path(output_dir).name,
        "seed": seed,
        "library_prompt_style": library_prompt_style,
    }

    # rollouts sample with one of the eval decoding profiles (see config/training.yaml)
    rollout_decoding = recipe["eval_decoding_profiles"][
        recipe["rollout_decoding_profile"]
    ]
    overrides["temperature"] = rollout_decoding["temperature"]
    overrides["top_p"] = rollout_decoding["top_p"]
    overrides["top_k"] = rollout_decoding["top_k"]
    overrides["max_completion_length"] = rollout_decoding["max_new_tokens"]

    if epochs is not None:
        overrides["num_train_epochs"] = epochs
    if max_steps is not None:
        overrides["max_steps"] = max_steps

    if debug:
        # a debug run is only a couple of steps: log every step, cap rollout length, and
        # shrink the optimizer step so a handful of prompts still make whole steps
        overrides["logging_steps"] = 1
        overrides["max_completion_length"] = DEBUG_MAX_COMPLETION_LENGTH
        overrides["gradient_accumulation_steps"] = DEBUG_GRADIENT_ACCUMULATION_STEPS
        if epochs is None:
            overrides["num_train_epochs"] = DEBUG_EPOCHS

    (training_args,) = TrlParser((GRPOConfig,)).parse_dict(
        {**recipe, **overrides},
        allow_extra_keys=True,
    )

    set_seed(training_args.seed)

    # logs, timings and progress bars come from the first gpu's process only - the others
    # are identical copies, and printing them all makes the job log unreadable
    is_main_process = training_args.process_index == 0

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # the other processes log at the recipe's log_level_replica (errors only)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    lib_log_level = max(log_level, logging.WARNING)
    training_args.log_level = logging.getLevelName(lib_log_level).lower()
    datasets.utils.logging.set_verbosity(lib_log_level)
    transformers.utils.logging.set_verbosity(lib_log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if not is_main_process:
        datasets.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()

    logger.info(
        f"Distributed training: {training_args.world_size} process(es), one per gpu"
    )
    logger.debug(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ###############
    # Load dataset
    ###############
    system_prompt = build_system_prompt(
        training_args.libraries_requirements_file,
        style=training_args.library_prompt_style,
    )
    dataset = load_train_dataset(
        dataset_dir=dataset_dir,
        system_prompt=system_prompt,
        train_n=DEBUG_TRAIN_SAMPLES if debug else None,
        val_n=DEBUG_VAL_SAMPLES if debug else None,
    )

    ################
    # select reward
    ################
    # require one task type across train and validation; otherwise one reward callable
    # would score some rows with the wrong semantics.
    reward_kinds: set[str] = set()
    for split in ("train", "val"):
        split_dataset = dataset[split]
        if "task_type" not in split_dataset.column_names:
            reward_kinds.add(infer_reward_kind({}))
        else:
            reward_kinds.update(split_dataset["task_type"])
    if len(reward_kinds) != 1:
        raise ValueError(
            f"Training dataset must contain exactly one reward kind; found "
            f"{sorted(reward_kinds)}."
        )

    # build_reward_functions also rejects profile components this task does not understand.
    reward_kind = reward_kinds.pop()
    logger.info(f"Reward kind: {reward_kind}")

    #################
    # Load tokenizer
    #################
    # the model itself is built by the trainer from the recipe's model_init_kwargs, which
    # lets fsdp shard it as it loads; the tokenizer is ours so the reward can see the eos
    # token and the recipe can override the chat template
    logger.info("*** Loading tokenizer ***")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    # one reward function per component, so each is logged as its own rate during
    # training and validation (see build_reward_functions)
    reward_funcs, component_weights = build_reward_functions(
        kind=reward_kind,
        weights=reward_weights,
        eos_token_id=tokenizer.eos_token_id,
    )
    training_args.reward_weights = component_weights

    #############################
    # Initialize the GRPO trainer
    #############################
    if training_args.use_vllm and training_args.vllm_kv_cache_memory_gib is not None:
        _fix_vllm_kv_cache_size(training_args.vllm_kv_cache_memory_gib)

    with log_timer(
        "trainer init (model loading, vllm startup)", enabled=is_main_process
    ):
        trainer = GRPOTrainer(
            model=model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"]
            if training_args.eval_strategy != "no"
            else None,
            processing_class=tokenizer,
            callbacks=[
                SaveTrainerStateCallback(metrics_dir=metrics_dir),
                PruneResumeStateCallback(),
                LogGPUMemoryCallback(),
            ],
        )
    # the trainer loads the model and the reference model (beta > 0), so this shows what
    # is left for optimizer states and activations before training starts
    if is_main_process:
        log_gpu_memory("after trainer init", device=training_args.device.index)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # trl also logs each optimizer step's duration, as step_time in the per-step metrics
    with log_timer("training", enabled=is_main_process):
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)

    checkpoints = epoch_checkpoints(training_args.output_dir)
    if trainer.is_world_process_zero():
        # added after log_metrics (which only formats scalars); save_json below writes
        # them into train_results.json regardless
        metrics["reward_profile"] = {"name": reward_profile, "weights": reward_weights}
        metrics["reward_kind"] = reward_kind
        metrics["epoch_checkpoints"] = {
            str(epoch): str(path) for epoch, path in checkpoints.items()
        }
        # to metrics_dir (the repo) rather than trainer.save_metrics' output_dir, which
        # is project storage alongside the checkpoints
        save_json(metrics, metrics_dir / "train_results.json")
        trainer.state.save_to_json(metrics_dir / "trainer_state.json")
        logger.info(f"Saved {len(checkpoints)} epoch checkpoint(s) to {output_dir}")

    ##########
    # Evaluate
    ##########
    # the recipe turns in-training validation off (the eval jobs score each epoch's
    # checkpoint instead), so this only writes anything if a recipe re-enables it. with
    # eval_strategy "epoch", the last round already scored the final model and is reused
    # rather than generating the whole split again
    eval_step_history = _periodic_eval_history(trainer.state.log_history)
    if eval_step_history:
        final_metrics = eval_step_history.get(str(trainer.state.global_step))
        if final_metrics is not None:
            trainer.log_metrics("eval", final_metrics)
        if trainer.is_world_process_zero():
            # "final" first so it's the headline number when the json file is read
            eval_results = {"final": final_metrics, "steps": eval_step_history}
            save_json(eval_results, metrics_dir / "eval_results.json")

    return checkpoints
