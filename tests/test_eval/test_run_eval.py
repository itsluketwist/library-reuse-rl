"""Tests for evaluation run setup: output paths and model-provider resolution."""

from pathlib import Path

import pytest

from src.eval import evaluate
from src.eval.generate_api import DEFAULT_MAX_TOKENS_PARAM, ApiSettings
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
            base="deepseek-v4-flash",
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
                "base": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        )

        assert api is not None
        assert api.api_key_env == "DEEPSEEK_API_KEY"
        assert api.base_url == "https://api.deepseek.com"
        assert api.max_tokens_param == DEFAULT_MAX_TOKENS_PARAM
        assert api.extra_params is None

    def test_unknown_provider_raises(self) -> None:
        """A typo is caught here rather than silently falling back to vllm, which would
        try to load an api model id as local weights."""
        with pytest.raises(ValueError, match="unknown provider"):
            build_api_settings({"provider": "apo", "base": "deepseek-v4-flash"})

    def test_api_entry_without_a_key_variable_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key_env"):
            build_api_settings({"provider": "api", "base": "deepseek-v4-flash"})


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
