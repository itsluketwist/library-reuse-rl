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
from pathlib import Path

import datasets
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from src.data import load_train_dataset
from src.models import get_model, get_tokenizer
from src.prompts import PromptStyle, build_system_prompt
from src.utils.config import load_full_yaml
from src.utils.io import load_json, save_json
from src.utils.log import log_gpu_memory

from .config import GRPOConfig
from .rewards import make_library_reuse_reward
from .wandb_logging import init_wandb_training


logger = logging.getLogger(__name__)

# sample counts used by --debug mode (src/run_grpo.py) - just enough to fill a train
# batch and give eval a couple of rows (see config/train/demo.yaml), small enough to
# finish in seconds. GRPOTrainer still takes a real optimizer step on a partial/short
# epoch like this - transformers' Trainer flushes any accumulated gradients at epoch end.
DEBUG_TRAIN_SAMPLES = 16
DEBUG_VAL_SAMPLES = 4


class SaveTrainerStateCallback(TrainerCallback):
    """
    Periodically overwrites metrics_dir/trainer_state.json with the training history
    so far, on the same cadence as checkpointing (`save_steps`) - so if a run crashes,
    the metric history up to the last save isn't lost along with it. Written to
    metrics_dir (tracked in the repo) rather than output_dir (scratch storage, for the
    large model files) since this is small and worth keeping alongside the code.
    """

    def __init__(self, metrics_dir: str | Path) -> None:
        self.metrics_dir = metrics_dir

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero and state.global_step % args.save_steps == 0:
            state.save_to_json(os.path.join(self.metrics_dir, "trainer_state.json"))


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
            log_gpu_memory("after first optimizer step")


def _save_metrics_json(
    metrics_dir: str | Path,
    split: str,
    metrics: dict,
) -> None:
    """
    Write metrics_dir/{split}_results.json, updating metrics_dir/all_results.json too.

    Mirrors what trainer.save_metrics() does, except targeting metrics_dir (tracked in
    the repo) instead of output_dir (scratch storage) - see train_model_with_grpo.
    """
    metrics_dir = Path(metrics_dir)
    save_json(metrics, metrics_dir / f"{split}_results.json")

    all_results_path = metrics_dir / "all_results.json"
    all_metrics = load_json(all_results_path) if all_results_path.exists() else {}
    all_metrics.update(metrics)
    save_json(all_metrics, all_results_path)


def train_model_with_grpo(
    model_name_or_path: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    metrics_dir: str | Path,
    reward_profile: str,
    reward_weights: dict[str, float],
    seed: int,
    recipe_config: str = "config/train/demo.yaml",
    library_prompt_style: PromptStyle = "lib",
    debug: bool = False,
) -> None:
    """
    Run GRPO training for `model_name_or_path` against the skywork-format dataset at
    `dataset_dir` (see src/data.py::load_train_dataset), saving the trained model to
    `output_dir` and its training metrics/logs to `metrics_dir`.

    `output_dir` is expected to be large (model weights, checkpoints) and typically
    points at scratch storage, while `metrics_dir` is small (json metrics files,
    trainer_state.json) and typically stays in the repo - see src/run_grpo.py, which
    computes both.

    `seed` and `library_prompt_style` override the corresponding fields parsed from
    `recipe_config`, so that - along with the model, dataset, and output path -
    they're resolved once by the caller (src/run_grpo.py) rather than duplicated in
    the recipe file. Every other training hyperparameter, including
    `libraries_requirements_file` (used to build the system prompt via
    src/prompts.py::build_system_prompt - see src/grpo/config.py) and the decoding
    settings used during rollout generation (temperature/top_p/top_k), comes straight
    from `recipe_config`.

    `reward_profile` and `reward_weights` (see config/rewards.yaml and
    src/grpo/rewards.py) are recorded into train_results.json, alongside the rest of
    the run's training metrics, so the exact reward function used to produce a
    trained model stays recoverable from its metrics directory.

    If `debug` is set, the train/val splits are truncated to DEBUG_TRAIN_SAMPLES /
    DEBUG_VAL_SAMPLES rows, to quickly smoke-test that training runs end-to-end.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    recipe = load_full_yaml(recipe_config)
    # allow_extra_keys=True since recipe also carries rollout_decoding_profile and
    # eval_decoding_profiles (for run_eval.py) that GRPOConfig/ModelConfig don't
    # recognise - training and inference share one recipe file
    training_args, model_args = TrlParser((GRPOConfig, ModelConfig)).parse_dict(
        recipe, allow_extra_keys=True
    )

    # rollouts and eval sample with the same decoding profile - see
    # rollout_decoding_profile/eval_decoding_profiles in config/train/demo.yaml
    rollout_decoding = recipe["eval_decoding_profiles"][
        recipe["rollout_decoding_profile"]
    ]
    training_args.temperature = rollout_decoding["temperature"]
    training_args.top_p = rollout_decoding["top_p"]
    training_args.top_k = rollout_decoding["top_k"]
    training_args.max_completion_length = rollout_decoding["max_new_tokens"]

    model_args.model_name_or_path = model_name_or_path
    training_args.output_dir = str(output_dir)
    training_args.seed = seed
    training_args.library_prompt_style = library_prompt_style

    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training parameters {training_args}")

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
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, chat_template=training_args.chat_template)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(
        model_args, gradient_checkpointing=training_args.gradient_checkpointing
    )
    # log gpu memory now the model is loaded. note: without quantization the model
    # is loaded on cpu and only moved to the gpu by the trainer, so this mostly
    # shows the baseline free memory on the device before training claims it
    log_gpu_memory("after model load")

    # Get the reward function, configured with this run's reward_weights
    reward_func = make_library_reuse_reward(reward_weights)

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=[
            SaveTrainerStateCallback(metrics_dir=metrics_dir),
            LogGPUMemoryCallback(),
        ],
    )
    # log again after trainer init - the trainer loads the reference model (needed
    # since beta > 0), so this shows the memory left for optimizer states and
    # activations before training actually starts
    log_gpu_memory("after trainer init")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    # added after log_metrics (rather than before) since its formatting only handles
    # scalar metric values - _save_metrics_json, called next, just does a plain
    # json.dump, so this still lands in train_results.json alongside the run's metrics
    metrics["reward_profile"] = {"name": reward_profile, "weights": reward_weights}
    _save_metrics_json(metrics_dir, "train", metrics)
    trainer.state.save_to_json(metrics_dir / "trainer_state.json")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {"tags": ["library-reuse-rl"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset["val"])
        trainer.log_metrics("eval", metrics)
        _save_metrics_json(metrics_dir, "eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
