"""Tests for scoring the product-choice split."""

from src.eval.scoring import score_dataset


def _responses(*texts: str) -> list[list[tuple[str, str]]]:
    """One sampled response per row, each paired with a 'stop' stop-reason."""
    return [[(text, "stop")] for text in texts]


class TestScoreProductChoice:
    """Tests for scoring the product-choice split."""

    def test_external_pick_and_diy_rates(self):
        """External picks and DIY picks are counted into their overall rates."""
        dataset = [
            {
                "idx": 0,
                "task_type": "product_choice",
                "category_id": "databases",
                "allowlist": ["postgresql"],
                "diy_terms": ["json file"],
            },
            {
                "idx": 1,
                "task_type": "product_choice",
                "category_id": "databases",
                "allowlist": ["postgresql"],
                "diy_terms": ["json file"],
            },
        ]
        responses = _responses(
            "reasoning</think>\nUse PostgreSQL.",
            "reasoning</think>\nJust use a JSON file.",
        )

        analysis, results = score_dataset(
            dataset,
            responses,
            evaluator="product_choice",
        )

        assert analysis["task"] == "product_choice"
        assert analysis["external_pick_rate"] == 0.5
        assert analysis["diy_rate"] == 0.5
        assert results[0]["external_pick"] and results[0]["primary"] == "postgresql"
        assert results[1]["diy_pick"]
