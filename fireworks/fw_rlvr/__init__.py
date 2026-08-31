"""Neutral package for Fireworks-specific dataset, inference, and scoring glue."""

from .dataset import prepare_records, representative_sample
from .inference import GenerationConfig, generate_dataset


__all__ = [
    "GenerationConfig",
    "generate_dataset",
    "prepare_records",
    "representative_sample",
]
