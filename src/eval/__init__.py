"""Evaluation utilities: generating responses from a model and scoring them."""

from typing import Any


__all__ = [
    "evaluate_model",
    "generate_responses",
]


# lazily imported so submodules that don't need the model runtime (e.g. src.eval.scoring)
# can be imported without pulling in vllm, which is only installed on the training/eval host
def __getattr__(name: str) -> Any:
    if name == "evaluate_model":
        from .evaluate import evaluate_model

        return evaluate_model
    if name == "generate_responses":
        from .generate import generate_responses

        return generate_responses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
