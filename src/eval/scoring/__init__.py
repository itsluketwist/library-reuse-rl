"""Score generated eval responses per task type (no model runtime, so unit-testable).

The eval loop (src/eval/evaluate.py) generates one response per prompt; this package turns
those responses into per-row results and an analysis summary using the evaluator explicitly
configured beside the dataset. Tested-implementation datasets execute code, product-choice
and library-usage use the deterministic checks their rewards use, and LibReuseBench
delegates to its official scorer. One file per task type, mirroring src/grpo/rewards/.
"""

from typing import Iterable

from src.eval.scoring.library_usage import score_library_usage
from src.eval.scoring.libreusebench import score_libreusebench
from src.eval.scoring.libreusebench_function import score_libreusebench_function
from src.eval.scoring.product_choice import score_product_choice
from src.eval.scoring.tested import score_tested


# split by responses shape so each dict stays internally homogeneous (a type checker can't
# verify a single dict mixing both signatures against a single call site)
_SINGLE_SAMPLE_SCORERS = {
    "implement_tested": score_tested,
    "product_choice": score_product_choice,
}
_MULTI_SAMPLE_SCORERS = {
    "library_usage": score_library_usage,
    "libreusebench": score_libreusebench,
    "libreusebench_function": score_libreusebench_function,
}
_SCORERS = {**_SINGLE_SAMPLE_SCORERS, **_MULTI_SAMPLE_SCORERS}

# evaluators that score every sampled response per row (see eval_decoding_profiles'
# n_samples); the rest only ever score the first sample - score_dataset takes care of
# that split itself, so callers never need to know which evaluators are multi-sample
MULTI_SAMPLE_EVALUATORS = set(_MULTI_SAMPLE_SCORERS)


def score_dataset(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str]]],
    evaluator: str,
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score a whole eval dataset with its configured evaluator.

    `responses` holds every sampled response per row (see generate_vllm_responses'
    n_samples support). MULTI_SAMPLE_EVALUATORS score every sample; the rest only ever
    look at the first, which this takes care of internally.

    Datasets that execute code build a venv to run it in: `environment_root` is where a
    disposable one goes, and `environment_cache_dir` names one to reuse across runs
    instead (see libreusebench.environment.Environment).

    Returns the (analysis, results) pair for the dataset's task.
    """
    rows = list(dataset)
    if not rows:
        return {"task": evaluator, "n_total": 0}, []

    if evaluator in _MULTI_SAMPLE_SCORERS:
        return _MULTI_SAMPLE_SCORERS[evaluator](
            rows, responses, python_executable, environment_root, environment_cache_dir
        )
    if evaluator in _SINGLE_SAMPLE_SCORERS:
        single_responses = [samples[0] for samples in responses]
        return _SINGLE_SAMPLE_SCORERS[evaluator](
            rows,
            single_responses,
            python_executable,
            environment_root,
            environment_cache_dir,
        )
    raise ValueError(f"Unknown evaluator '{evaluator}'. Available: {sorted(_SCORERS)}")


__all__ = [
    "MULTI_SAMPLE_EVALUATORS",
    "score_dataset",
]
