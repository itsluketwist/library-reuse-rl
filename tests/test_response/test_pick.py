"""Tests for the deterministic primary-pick extractor."""

from src.response.pick import extract_primary_pick


_ALLOWLIST = ["postgresql", "mongodb", "next.js"]
_ALIASES = {"postgresql": ["postgres", "pg"]}
_DIY = ["json file", "in-memory dict"]


def _pick(response: str):
    """Run the extractor with the shared databases-style reference data."""
    return extract_primary_pick(
        response=response,
        allowlist=_ALLOWLIST,
        aliases=_ALIASES,
        diy_terms=_DIY,
    )


class TestExtractPrimaryPick:
    """Tests for extract_primary_pick."""

    def test_leading_tool_is_the_primary_pick(self):
        """The earliest un-negated tool mention becomes the primary pick."""
        pick = _pick("I'd use PostgreSQL, though MongoDB also works.")

        assert pick.primary == "postgresql"
        assert pick.is_external and not pick.is_diy
        assert pick.mentioned_external == ["postgresql", "mongodb"]

    def test_alias_matches_the_canonical_tool(self):
        """A pick named by an alias resolves to the canonical allowlist name."""
        pick = _pick("just use postgres for this")

        assert pick.primary == "postgresql" and pick.is_external

    def test_diy_before_any_tool_is_a_diy_pick(self):
        """When a DIY phrase leads, the pick is flagged as roll-your-own."""
        pick = _pick("honestly just write to a JSON file; Postgres is overkill")

        assert pick.is_diy and not pick.is_external
        # the trailing tool is still recorded as mentioned
        assert pick.mentioned_external == ["postgresql"]

    def test_negated_tool_is_ignored(self):
        """A negated mention ('don't use X') is not treated as a pick."""
        pick = _pick("don't use MongoDB here")

        assert pick.primary is None
        assert pick.mentioned_external == []

    def test_word_boundaries_prevent_substring_matches(self):
        """A short alias like 'pg' does not match inside another word."""
        pick = _pick("the pgadmin tool is unrelated")

        # 'pg' only appears inside 'pgadmin', so nothing is picked
        assert pick.primary is None

    def test_no_recognised_tool_or_diy(self):
        """A response naming nothing on the lists yields an empty pick."""
        pick = _pick("it depends on your needs")

        assert pick.primary is None
        assert not pick.is_external and not pick.is_diy
