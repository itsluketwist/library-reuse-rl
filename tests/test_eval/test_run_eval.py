"""Tests for evaluation run setup: output paths and model-provider resolution."""

from pathlib import Path

import pytest

from src.eval import evaluate
from src.eval.generate_api import (
    DEFAULT_MAX_TOKENS_PARAM,
    DEFAULT_MAX_WORKERS,
    ApiSettings,
)
from src.run_eval import apply_environment_root, build_api_settings, eval_results_path


class TestEvalResultsPath:
    """eval_results_path decides where a run's results are saved."""

    def test_saves_in_the_models_folder(self) -> None:
        path = eval_results_path(
            output_dir=Path("output/baselines/qwen3-8b"),
            dataset="libreusebench",
            decoding_profile="greedy",
        )

        assert path.parent == Path("output/baselines/qwen3-8b")

    def test_prompt_styles_never_share_a_path(self) -> None:
        """ "base", and "lib"/"use" with no requirements file, all produce an empty library
        suffix - without the style in the name they would overwrite each other's results,
        which is exactly what concurrent slurm jobs from one submission would do."""
        paths = {
            eval_results_path(
                output_dir=Path("output/baselines/qwen3-8b"),
                dataset="libreusebench",
                decoding_profile="greedy",
                prompt_style=style,
            )
            for style in ("lib", "use", "std", "ext", "base")
        }

        assert len(paths) == 5

    def test_requirements_file_still_versions_the_suffix(self) -> None:
        """A requirements file keeps its versioned suffix, so runs under different library
        prompts stay distinguishable."""
        path = eval_results_path(
            output_dir=Path("output/baselines/qwen3-8b"),
            dataset="libreusebench",
            decoding_profile="greedy",
            libraries_requirements_file="setup/requirements/v1.txt",
            prompt_style="lib",
        )

        assert path.name == "libreusebench-greedy-libv1.json"


class TestApplyEnvironmentRoot:
    """apply_environment_root relocates the disposable execution venvs."""

    def test_moves_datasets_that_build_an_environment(self) -> None:
        """On the HPC these venvs belong on scratch, not in the repo."""
        configs = {
            "libreusebench_test": {
                "path": "x.jsonl",
                "evaluator": "libreusebench",
                "environment_root": ".libreuseeval-venvs",
            }
        }

        moved = apply_environment_root(configs, "/scratch/me/libreuseeval-venvs")

        assert moved["libreusebench_test"]["environment_root"] == (
            "/scratch/me/libreuseeval-venvs"
        )
        # the original config is not mutated, only copied
        assert (
            configs["libreusebench_test"]["environment_root"] == ".libreuseeval-venvs"
        )

    def test_leaves_datasets_that_never_build_one(self) -> None:
        """library_usage is scored without executing anything, so it has no venv."""
        configs = {"library_usage": {"path": "x.jsonl", "evaluator": "library_usage"}}

        moved = apply_environment_root(configs, "/scratch/me/libreuseeval-venvs")

        assert "environment_root" not in moved["library_usage"]

    def test_unset_override_changes_nothing(self) -> None:
        """Off the HPC the configured local default is used."""
        configs = {
            "libreusebench_test": {
                "path": "x.jsonl",
                "evaluator": "libreusebench",
                "environment_root": ".libreuseeval-venvs",
            }
        }

        assert apply_environment_root(configs, None) == configs


class TestDebugEnvironment:
    """Debug runs reuse one venv instead of building a disposable one each time."""

    def test_debug_reuses_a_named_environment(self, monkeypatch, tmp_path) -> None:
        """Repeated smoke tests are never concurrent, so they can share a venv rather
        than reinstalling every package - and it survives for inspection afterwards."""
        seen = {}

        def fake_score_dataset(**kwargs):
            seen.update(kwargs)
            return {"task": "libreusebench", "n_total": 0}, []

        monkeypatch.setattr(evaluate, "score_dataset", fake_score_dataset)
        monkeypatch.setattr(evaluate, "load_eval_dataset", lambda **_: {"prompt": []})
        monkeypatch.setattr(evaluate, "generate_api_responses", lambda **_: [])

        evaluate.evaluate_model(
            base="deepseek-flash",
            weights_path=None,
            dataset_configs={
                "libreusebench": {
                    "path": "x.jsonl",
                    "evaluator": "libreusebench",
                    "environment_root": ".libreuseeval-venvs",
                }
            },
            decoding_settings={"max_new_tokens": 16},
            # a temporary folder, so the result files it writes never land in the repo
            output_paths={"libreusebench": tmp_path / "out.json"},
            debug=True,
            api=ApiSettings(api_key_env="FAKE_API_KEY"),
        )

        assert seen["environment_root"] == ".libreuseeval-venvs"
        assert seen["environment_cache_dir"] == str(
            Path(".libreuseeval-venvs") / evaluate.DEBUG_ENVIRONMENT_NAME
        )


class TestBuildApiSettings:
    """build_api_settings reads a model entry's API configuration."""

    def test_local_entry_has_no_api_settings(self) -> None:
        assert build_api_settings({"base": "Qwen/Qwen3-8B", "path": None}) is None

    def test_api_entry_is_read_with_defaults(self) -> None:
        api = build_api_settings(
            {
                "provider": "api",
                "base": "deepseek-flash",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        )

        assert api is not None
        assert api.api_key_env == "DEEPSEEK_API_KEY"
        assert api.base_url == "https://api.deepseek.com"
        assert api.max_tokens_param == DEFAULT_MAX_TOKENS_PARAM
        assert api.extra_params is None
        assert api.max_workers == DEFAULT_MAX_WORKERS

    def test_api_entry_can_cap_its_concurrent_requests(self) -> None:
        """A provider with a low concurrency limit sets max_workers below it."""
        api = build_api_settings(
            {
                "provider": "api",
                "base": "deepseek-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
                "max_workers": 4,
            }
        )

        assert api is not None
        assert api.max_workers == 4

    def test_unknown_provider_raises(self) -> None:
        """A typo is caught here rather than silently falling back to vllm, which would
        try to load an api model id as local weights."""
        with pytest.raises(ValueError, match="unknown provider"):
            build_api_settings({"provider": "apo", "base": "deepseek-flash"})

    def test_api_entry_without_a_key_variable_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key_env"):
            build_api_settings({"provider": "api", "base": "deepseek-flash"})


def test_registered_api_models_resolve() -> None:
    """Every "provider: api" entry in the real registry is complete enough to run."""
    from src.utils.config import load_full_yaml

    registry = load_full_yaml("config/models.yaml")
    api_entries = {
        key: entry for key, entry in registry.items() if entry.get("provider") == "api"
    }

    assert api_entries
    for key, entry in api_entries.items():
        api = build_api_settings(entry)
        assert api is not None, key
        # a hosted model has no local weights, so "path" must stay blank
        assert not entry.get("path"), key


def test_a_hung_shutdown_is_forced_to_exit(monkeypatch):
    """After an error, the process exits even if something keeps it alive."""
    import threading
    import time

    from src import run_eval as run_eval_module

    exits = []
    monkeypatch.setattr(run_eval_module.os, "_exit", exits.append)

    run_eval_module._exit_if_shutdown_hangs(seconds=0.01)
    deadline = time.time() + 2
    while not exits and time.time() < deadline:
        time.sleep(0.01)

    assert exits == [1]
    assert not any(
        isinstance(thread, threading.Timer) and thread.is_alive()
        for thread in threading.enumerate()
    )


class TestScoringFailures:
    """A scoring failure keeps the generations and doesn't stop the other datasets."""

    _CONFIGS = {
        "first": {"path": "first.jsonl", "evaluator": "library_usage"},
        "second": {"path": "second.jsonl", "evaluator": "library_usage"},
    }

    def _evaluate(self, monkeypatch, tmp_path, generate, score) -> dict[str, Path]:
        """Run evaluate_model on two one-prompt datasets, with fake generation and scoring.

        Returns each dataset's result path.
        """
        monkeypatch.setattr(
            evaluate, "load_eval_dataset", lambda **_: {"prompt": ["p"]}
        )
        monkeypatch.setattr(evaluate, "generate_api_responses", generate)
        monkeypatch.setattr(evaluate, "score_dataset", score)
        output_paths = {name: tmp_path / f"{name}.json" for name in self._CONFIGS}
        evaluate.evaluate_model(
            base="deepseek-flash",
            weights_path=None,
            dataset_configs=self._CONFIGS,
            decoding_settings={"max_new_tokens": 16},
            output_paths=output_paths,
            api=ApiSettings(api_key_env="FAKE_API_KEY"),
        )
        return output_paths

    @staticmethod
    def _score(**kwargs):
        """A fake scorer that fails on the first dataset's response."""
        if kwargs["responses"] == [[("first response", "eos", 2)]]:
            raise RuntimeError("could not install a package")
        return {"task": "library_usage", "n_total": 1}, [{"idx": 0}]

    def test_failure_keeps_generations_and_carries_on(self, monkeypatch, tmp_path):
        """The failed dataset's generations are saved, the next dataset still runs, and
        the run then reports the failure."""
        # each dataset's response is named after it, in the order they're generated
        names = iter(["first", "second"])

        def generate(**kwargs):
            return [[(f"{next(names)} response", "eos", 2)]]

        with pytest.raises(RuntimeError, match=r"\['first'\]"):
            self._evaluate(monkeypatch, tmp_path, generate, self._score)

        # the failed dataset kept its generations for later, with no results
        assert (tmp_path / "first.unscored.jsonl").exists()
        assert not (tmp_path / "first.json").exists()
        # the other dataset was scored as normal, replacing its unscored copy
        assert (tmp_path / "second.json").exists()
        assert not (tmp_path / "second.unscored.jsonl").exists()

    def test_a_later_run_scores_the_saved_generations(self, monkeypatch, tmp_path):
        """Saved generations are scored without generating them again."""
        evaluate.save_unscored_generations(
            [[("saved response", "eos", 2)]],
            tmp_path / "first.unscored.jsonl",
        )
        scored = []

        def generate(**kwargs):
            return [[("second response", "eos", 2)]]

        def score(**kwargs):
            scored.append(kwargs["responses"])
            return {"task": "library_usage", "n_total": 1}, [{"idx": 0}]

        self._evaluate(monkeypatch, tmp_path, generate, score)

        # the saved sample comes back whole, its token count included
        assert scored[0] == [[("saved response", "eos", 2)]]
        assert (tmp_path / "first.json").exists()
        assert not (tmp_path / "first.unscored.jsonl").exists()


def test_mismatched_saved_generations_are_ignored(tmp_path):
    """Saved generations are only reused when they match the run's prompts and samples."""
    path = tmp_path / "x.unscored.jsonl"
    evaluate.save_unscored_generations([[("a", "eos", 1)], [("b", "eos", 1)]], path)

    assert evaluate.load_unscored_generations(path, n_prompts=2, n_samples=1) == [
        [("a", "eos", 1)],
        [("b", "eos", 1)],
    ]
    assert evaluate.load_unscored_generations(path, n_prompts=3, n_samples=1) is None
    assert evaluate.load_unscored_generations(path, n_prompts=2, n_samples=5) is None
    assert evaluate.load_unscored_generations(tmp_path / "none.jsonl", 2, 1) is None


def test_saved_generations_from_before_token_counts_still_load(tmp_path):
    """A file saved before token counts were kept loads, with no count for its samples."""
    path = tmp_path / "old.unscored.jsonl"
    path.write_text('{"samples": [{"response": "a", "stop_reason": "eos"}]}\n')

    assert evaluate.load_unscored_generations(path, n_prompts=1, n_samples=1) == [
        [("a", "eos", None)],
    ]


class TestFailedApiRequests:
    """A failed api request is asked for again, and nothing is scored while one still fails."""

    def _evaluate(
        self,
        monkeypatch,
        tmp_path: Path,
        replies: list[list[list[tuple[str, str, int | None]]]],
        n_samples: int = 1,
    ) -> tuple[list[dict], list[list[list[tuple[str, str, int | None]]]]]:
        """Run evaluate_model on one one-prompt dataset; each generation call returns the
        next of `replies`.

        Returns the generation calls made (their prompts and settings) and what was scored.
        """
        calls: list[dict] = []
        scored: list[list[list[tuple[str, str, int | None]]]] = []

        def generate(**kwargs):
            calls.append(kwargs)
            return replies[len(calls) - 1]

        def score(**kwargs):
            scored.append(kwargs["responses"])
            return {"task": "library_usage", "n_total": 1}, [{"idx": 0}]

        monkeypatch.setattr(
            evaluate, "load_eval_dataset", lambda **_: {"prompt": ["p"]}
        )
        monkeypatch.setattr(evaluate, "generate_api_responses", generate)
        monkeypatch.setattr(evaluate, "score_dataset", score)
        evaluate.evaluate_model(
            base="deepseek-flash",
            weights_path=None,
            dataset_configs={
                "only": {"path": "only.jsonl", "evaluator": "library_usage"}
            },
            decoding_settings={"max_new_tokens": 16, "n_samples": n_samples},
            output_paths={"only": tmp_path / "only.json"},
            api=ApiSettings(api_key_env="FAKE_API_KEY"),
        )
        return calls, scored

    def test_a_failed_request_is_retried_then_scored(self, monkeypatch, tmp_path):
        calls, scored = self._evaluate(
            monkeypatch,
            tmp_path,
            replies=[
                [[("answer", "eos", 3), ("API request failed: 429", "error", None)]],
                [[("retried answer", "eos", 4)]],  # the retry: the failed sample only
            ],
            n_samples=2,
        )

        assert calls[1]["prompts"] == ["p"]
        assert calls[1]["decoding_settings"]["n_samples"] == 1
        assert scored == [[[("answer", "eos", 3), ("retried answer", "eos", 4)]]]
        assert (tmp_path / "only.json").exists()

    def test_a_request_that_keeps_failing_leaves_the_dataset_unscored(
        self,
        monkeypatch,
        tmp_path,
    ):
        failed = [[("API request failed: 402", "error", None)]]
        with pytest.raises(RuntimeError, match=r"\['only'\]"):
            self._evaluate(monkeypatch, tmp_path, replies=[failed, failed])

        # a failed request would count as a wrong answer, so nothing is scored or saved
        assert not (tmp_path / "only.json").exists()
        assert (tmp_path / "only.unscored.jsonl").exists()

    def test_a_later_run_asks_only_for_the_failed_samples(self, monkeypatch, tmp_path):
        evaluate.save_unscored_generations(
            [[("kept", "eos", 5), ("API request failed: 402", "error", None)]],
            tmp_path / "only.unscored.jsonl",
        )
        calls, scored = self._evaluate(
            monkeypatch,
            tmp_path,
            replies=[[[("new", "eos", 6)]]],
            n_samples=2,
        )

        # only one call: the retry of the one failed sample, with nothing regenerated
        assert len(calls) == 1
        assert scored == [[[("kept", "eos", 5), ("new", "eos", 6)]]]
