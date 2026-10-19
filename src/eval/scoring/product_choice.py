"""Score product-choice eval responses: the prose pick (external tool vs DIY vs none)."""

from typing import Iterable

from lure import split_reasoning

from src.eval.scoring.base import _rate
from src.response.pick import extract_primary_pick


def score_product_choice(
    dataset: Iterable[dict],
    responses: list[tuple[str, str, int | None]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score product-choice rows by the prose pick (external tool vs DIY vs none).

    Returns the (analysis, results) pair.
    """
    del python_executable, environment_root, environment_cache_dir
    results = []
    for row, (response, stop_reason, completion_tokens) in zip(dataset, responses):
        # only the answer is judged, never the reasoning
        _, answer = split_reasoning(response)
        pick = extract_primary_pick(
            response=answer,
            allowlist=row.get("allowlist") or [],
            aliases=row.get("aliases") or {},
            diy_terms=row.get("diy_terms") or [],
        )
        results.append(
            {
                "idx": row["idx"],
                "task_id": row.get("task_id"),
                "stop_reason": stop_reason,
                "completion_tokens": completion_tokens,
                "response": response,
                "category_id": row.get("category_id"),
                "primary": pick.primary,
                "external_pick": pick.is_external,
                "diy_pick": pick.is_diy,
                "mentioned_external": pick.mentioned_external,
            }
        )

    n_total = len(results)
    n_external = sum(1 for r in results if r["external_pick"])
    n_diy = sum(1 for r in results if r["diy_pick"])
    analysis = {
        "task": "product_choice",
        "n_total": n_total,
        "external_pick_rate": _rate(n_external, n_total),
        "diy_rate": _rate(n_diy, n_total),
        "no_pick_rate": _rate(n_total - n_external - n_diy, n_total),
    }
    return analysis, results
