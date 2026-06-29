"""Evaluation utilities: generating responses from a model and scoring them."""

from .evaluate import evaluate_model
from .generate import generate_responses


__all__ = [
    "evaluate_model",
    "generate_responses",
]
