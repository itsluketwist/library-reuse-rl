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
import math
import os
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path

import datasets
import lure
import torch
import transformers
from accelerate.utils import gather_object
from transformers import AutoTokenizer, Trainer, TrainerCallback, set_seed
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import GRPOTrainer, TrlParser
from trl.chat_template_utils import parse_response as parse_chat_response
from trl.trainer.utils import RepeatSampler

from src.data import load_train_dataset
from src.grpo.checkpoints import (
    epoch_checkpoints,
    last_resumable_checkpoint,
    prune_resume_state,
)
from src.grpo.config import GRPOConfig
from src.grpo.rewards import (
    LibraryReward,
    build_reward,
    build_reward_functions,
    infer_reward_kind,
)
from src.grpo.split_eval import (
    judge_completion,
    split_eval_entries,
    summarise_split_eval,
)
from src.grpo.wandb_logging import init_wandb_training
from src.prompts import PromptStyle, build_system_prompt
from src.utils.config import check_config_keys, load_full_yaml
from src.utils.io import save_json
from src.utils.log import log_gpu_memory, log_timer
from src.utils.vllm_settings import VLLM_ADDITIONAL_CONFIG


logger = logging.getLogger(__name__)

# constants used to configure src/run_grpo.py --debug mode
DEBUG_TRAIN_SAMPLES = 16
DEBUG_VAL_SAMPLES = 4
# long enough for qwen3's reasoning to finish, so debug rollouts actually earn rewards
DEBUG_MAX_COMPLETION_LENGTH = 2048
DEBUG_EPOCHS = 2
# 1 completion per gpu x 4 gpus x 4 = 16 completions (one prompt's rollouts) per step, so a
# handful of prompts still make whole steps (see config/training.yaml)
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
    Keeps only the newest checkpoint resumable, as each checkpoint is saved: older epoch
    checkpoints keep just their weights, and older mid-epoch ones are deleted.

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


class ResumeCheckpointCallback(TrainerCallback):
    """
    Saves an extra checkpoint every `save_steps` optimizer steps, besides each epoch's, so a
    job requeued at the slurm time limit loses at most that many steps of work rather than
    up to a whole epoch. Only the newest of these is kept (see prune_resume_state), and they
    are never evaluated or registered - epoch_checkpoints skips them.
    """

    def __init__(self, save_steps: int) -> None:
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step % self.save_steps == 0:
            control.should_save = True


def _find_repeat_sampler(dataloader: object) -> RepeatSampler:
    """
    Find trl's RepeatSampler (which shuffles the prompts) inside the training dataloader,
    where accelerate wraps it in its own batch samplers.

    Returns the sampler.
    """
    pending = [dataloader]
    while pending:
        current = pending.pop()
        if isinstance(current, RepeatSampler):
            return current
        for name in ("batch_sampler", "sampler"):
            inner = getattr(current, name, None)
            if inner is not None:
                pending.append(inner)
    raise RuntimeError(
        "No RepeatSampler found in the training dataloader - trl's sampling has changed, so "
        "a resumed run can no longer replay its shuffles (see ResumeShuffleCallback)."
    )


class ResumeShuffleCallback(TrainerCallback):
    """
    Makes a resumed run see the prompts in the same order an uninterrupted run would.

    trl's sampler draws one fresh shuffle of the prompts per epoch, from a generator seeded
    once when the job starts. A job resumed from a checkpoint would otherwise start that
    sequence again - repeating the first epoch's order, and, resumed partway through an
    epoch, skipping the wrong prompts. Before training starts, this replays the shuffles of
    the epochs already finished, so the generator is where an uninterrupted run's would be.
    The sampler is looked up on every start, so a change in trl fails the first (smoke) run
    rather than only a resumed one.
    """

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        train_dataloader: object = None,
        **kwargs,
    ) -> None:
        sampler = _find_repeat_sampler(train_dataloader)
        if not sampler.shuffle:
            return
        # the epochs finished before this job started (0 for a fresh run). state.epoch is
        # restored from the checkpoint, and lands on a whole number at an epoch's end
        finished_epochs = math.floor((state.epoch or 0) + 1e-6)
        for _ in range(finished_epochs):
            torch.randperm(sampler.num_samples, generator=sampler.generator)


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


class SplitEvalCallback(TrainerCallback):
    """
    Scores the val and test splits with the reward's components as training goes: once
    before the first step (the base model), then every `eval_steps` optimizer steps.

    Each round, every prompt is answered `n_samples` times by the colocated vllm engine,
    sampling exactly as the rollouts do, and every answer is judged by the run's own
    reward (see src/grpo/split_eval.py). The rates are logged as e.g.
    "eval_test/external_lib_used", so the curves show whether what the reward teaches
    carries over to prompts training never sees.

    trl's own evaluation is not used: it generates a few prompts at a time and runs a loss
    pass over each batch, far too slow to repeat every few steps. This calls the same trl
    methods the rollouts use instead (tokenising, then generating - which first copies the
    latest weights into vllm), so it relies on those methods of trl 1.13's GRPOTrainer.
    """

    def __init__(
        self,
        trainer: GRPOTrainer,
        reward: LibraryReward,
        splits: dict[str, datasets.Dataset],
        eval_steps: int,
        n_samples: int,
    ) -> None:
        self.trainer = trainer
        self.reward = reward
        self.splits = {name: list(split) for name, split in splits.items()}
        self.eval_steps = eval_steps
        self.n_samples = n_samples

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # a run resumed after a requeue already scored its earlier steps
        if state.global_step == 0:
            self._evaluate()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step % self.eval_steps == 0:
            # logging the rates (Trainer.log) clears the flag that tells the trainer to log
            # this step's own training metrics, which come straight after this callback. so
            # it is put back - otherwise every eval step's training log would be dropped,
            # and trl would fold its rewards into the next step's
            should_log = control.should_log
            self._evaluate()
            control.should_log = should_log

    def _decode(
        self,
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
    ) -> list[str]:
        """
        Decode the completions into text exactly as trl's _generate does for rollouts, so
        the reward reads the same text it does in training.

        Returns one completion text per completion.
        """
        tokenizer = self.trainer._tokenizer
        if (
            getattr(tokenizer, "response_template", None) is not None
            or getattr(tokenizer, "response_schema", None) is not None
        ):
            return [
                parse_chat_response(tokenizer, ids, prefix=prompt)["content"]
                for prompt, ids in zip(prompt_ids, completion_ids)
            ]
        return self.trainer.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )

    def _evaluate(self) -> None:
        """
        Run one round: every gpu answers its share of the prompts, the judgements are
        gathered, and every process logs the same rates.
        """
        trainer = self.trainer
        with log_timer(
            f"split eval at step {trainer.state.global_step}",
            enabled=trainer.is_world_process_zero(),
        ):
            entries = split_eval_entries(
                splits=self.splits,
                n_samples=self.n_samples,
                process_index=trainer.accelerator.process_index,
                num_processes=trainer.accelerator.num_processes,
            )
            # every gpu must take part, since copying the weights into vllm is collective
            if not entries:
                raise ValueError(
                    "Split eval has fewer answers than gpus - raise split_eval_samples."
                )

            prompt_ids, _images, _fields = trainer._tokenize_prompts(
                [row["prompt"] for _split, row in entries]
            )
            # text-only prompts: no images or extra multimodal fields to pass on
            completion_ids, _logprobs = trainer._generate_single_turn(
                prompt_ids,
                None,
                {},
            )
            texts = self._decode(prompt_ids, completion_ids)

            judgements = [
                judge_completion(
                    reward=self.reward,
                    split=split,
                    row=row,
                    text=text,
                    completion_ids=list(ids),
                )
                for (split, row), text, ids in zip(entries, texts, completion_ids)
            ]
            metrics = summarise_split_eval(
                judgements=gather_object(judgements),
                components=sorted(self.reward.weights),
            )

        # transformers' own Trainer.log, not GRPOTrainer's: that one also logs (and clears)
        # the metrics trl has gathered for the current training step, which would then be
        # missing from the step's own log entry, written straight after this callback
        Trainer.log(trainer, metrics)


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


def _configure_trl_vllm_engine(kv_cache_memory_gib: float | None) -> None:
    """
    Pass extra settings to trl's colocated vllm engine, which trl exposes no setting for,
    by swapping the engine class trl builds for one with them filled in.

    The engine gets the same `additional_config` as eval's (see src/utils/vllm_settings.py).
    And, when `kv_cache_memory_gib` is set, a fixed-size kv cache: at startup vllm measures
    the gpu's free memory, runs a dummy forward pass, and asserts free memory did not go up
    in between. In colocate mode the training model shares the process, and gpu memory it
    frees mid-profile (e.g. fsdp's weight broadcast buffers) trips that assertion (vllm
    issue #56830, unfixed in 0.28). Giving vllm a fixed cache size skips the check.
    """
    from trl.generation import vllm_generation

    engine_settings: dict = {"additional_config": dict(VLLM_ADDITIONAL_CONFIG)}
    if kv_cache_memory_gib is not None:
        engine_settings["kv_cache_memory_bytes"] = int(kv_cache_memory_gib * 1024**3)

    vllm_generation.LLM = partial(vllm_generation.LLM, **engine_settings)


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
    learning_rate: float | None = None,
) -> dict[int, Path]:
    """
    Run full-weight GRPO training for `model_name_or_path` against the dataset at
    `dataset_dir`, saving a checkpoint per epoch under `output_dir` (large, typically
    project storage) and its metrics to `metrics_dir` (small, typically the repo).

    The model is trained across every gpu in the job, sharded with FSDP2 (see
    config/accelerate_configs/fsdp2.yaml); rollouts are generated by a vllm engine
    colocated on those same gpus.

    `seed` and `library_prompt_style` override the corresponding `recipe_config` fields
    (resolved once by the caller), `epochs`/`max_steps` override the recipe's run length
    when given (see the debug modes in scripts/submit_debug.sh), and `learning_rate` its
    peak learning rate (see training_jobs in scripts/submit_grpo.sh); every other
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
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate

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
    # only a fully saved checkpoint counts - a job killed partway through saving one (e.g. at
    # the time limit, before requeuing) leaves an incomplete one behind, which is skipped
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = last_resumable_checkpoint(training_args.output_dir)
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
        test_n=DEBUG_VAL_SAMPLES if debug else None,
    )

    ################
    # select reward
    ################
    # require one task type across every split; otherwise one reward callable would score
    # some rows with the wrong semantics.
    reward_kinds: set[str] = set()
    for split in dataset:
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
    if training_args.use_vllm:
        _configure_trl_vllm_engine(training_args.vllm_kv_cache_memory_gib)

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
                ResumeShuffleCallback(),
                LogGPUMemoryCallback(),
            ],
        )
    # an extra, resume-only checkpoint every few steps, so a requeue loses little work
    if training_args.resume_save_steps:
        trainer.add_callback(ResumeCheckpointCallback(training_args.resume_save_steps))

    # score the val and test splits as training goes (see SplitEvalCallback), with a
    # reward of its own so judging them never touches the training reward's batch cache
    if training_args.split_eval_steps:
        trainer.add_callback(
            SplitEvalCallback(
                trainer=trainer,
                reward=build_reward(
                    kind=reward_kind,
                    weights=reward_weights,
                    eos_token_id=tokenizer.eos_token_id,
                ),
                splits={
                    split: dataset[split]
                    for split in ("val", "test")
                    if split in dataset
                },
                eval_steps=training_args.split_eval_steps,
                n_samples=training_args.split_eval_samples,
            )
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
        # the peak learning rate used, which a run can override (see --learning-rate)
        metrics["learning_rate"] = training_args.learning_rate
        # the lure version that parsed and scored every completion
        metrics["lure_version"] = lure.__version__
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
    # every round of validation logged during training (SplitEvalCallback's "eval_*"
    # rates, or trl's own if a recipe turns eval_strategy on), by step, so the curves can
    # be drawn from the repo. "final" is the round at the last step, when there was one
    eval_step_history = _periodic_eval_history(trainer.state.log_history)
    if eval_step_history:
        final_metrics = eval_step_history.get(str(trainer.state.global_step))
        if final_metrics is not None:
            trainer.log_metrics("eval", final_metrics)
        if trainer.is_world_process_zero():
            # "final" first so it's the headline number when the json file is read
            eval_results = {"final": final_metrics, "steps": eval_step_history}
            save_json(eval_results, metrics_dir / "eval_results.json")

    # once every gpu's process is done, close the connections between them - otherwise
    # pytorch warns at exit that destroy_process_group() was never called
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()

    return checkpoints
