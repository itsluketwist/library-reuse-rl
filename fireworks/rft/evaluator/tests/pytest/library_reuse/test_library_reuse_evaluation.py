"""Eval Protocol entry point for live local testing and Fireworks RFT upload."""

import os
from pathlib import Path

import pytest


_EVALUATOR_ROOT = Path(__file__).resolve().parents[3]

# ordinary repository tests run from the repository root and must not make api calls.
# eval protocol runs from the evaluator root both locally and in fireworks, so those
# deliberate evaluator runs must remain enabled even when the opt-in variable is absent.
_RUNNING_FROM_EVALUATOR_ROOT = Path.cwd().resolve() == _EVALUATOR_ROOT
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FIREWORKS_EVAL") != "1" and not _RUNNING_FROM_EVALUATOR_ROOT,
    reason=(
        "run from the evaluator directory or set RUN_FIREWORKS_EVAL=1 for "
        "API-backed Eval Protocol rollouts"
    ),
)
pytest.importorskip("eval_protocol")

from eval_protocol.models import (  # noqa: E402
    EvaluateResult,
    EvaluationRow,
    MetricResult,
)
from eval_protocol.pytest import (  # noqa: E402
    SingleTurnRolloutProcessor,
    evaluation_test,
)
from library_reuse_evaluator import score_response  # noqa: E402


_SAMPLE_DATASET = _EVALUATOR_ROOT / "development" / "sample.jsonl"
_INPUT_DATASET = os.environ.get(
    "FIREWORKS_RFT_DATASET",
    str(_SAMPLE_DATASET),
)
_LOCAL_MODEL = os.environ.get(
    "FIREWORKS_EVAL_MODEL",
    "fireworks_ai/accounts/fireworks/models/qwen3-8b",
)


@evaluation_test(
    input_dataset=[_INPUT_DATASET],
    completion_params=[
        {
            "model": _LOCAL_MODEL,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=0.0,
    mode="pointwise",
)
def test_library_reuse_evaluation(row: EvaluationRow) -> EvaluationRow:
    """Score the last assistant message using only its final-answer content.

    Returns the row with its structural reward and diagnostics attached.
    """
    assistant_messages = row.get_assistant_messages()
    content = assistant_messages[-1].content if assistant_messages else None
    ground_truth = row.ground_truth if isinstance(row.ground_truth, dict) else {}
    result = score_response(content, ground_truth)
    metrics = {
        name: MetricResult(
            score=score,
            reason=f"{name}={bool(score)}",
        )
        for name, score in result.binary_metrics().items()
    }
    row.evaluation_result = EvaluateResult(
        score=result.score,
        is_score_valid=True,
        reason=result.reason,
        metrics=metrics,
    )
    return row
