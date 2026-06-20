"""Tests for reward computation."""

from src.response.parser import ParsedResponse
from src.response.reward import (
    REWARD_WEIGHTS,
    RewardBit,
    compute_reward,
    compute_reward_bits,
)


def _dummy_parsed(
    code: str = "print(1)",
    valid: bool = True,
    std_libs: list[str] | None = None,
    ext_libs: list[str] | None = None,
    lib_usage: dict[str, list[dict]] | None = None,
) -> ParsedResponse:
    """Build a ParsedResponse for reward tests, with sensible defaults."""
    return ParsedResponse(
        reasoning="",
        answer=code,
        code=code,
        valid=valid,
        error=None,
        std_libs=std_libs or [],
        ext_libs=ext_libs or [],
        lib_usage=lib_usage or {},
        logical_lines=1,
        maintainability_index=100.0,
    )


class TestComputeRewardBits:
    """Tests for compute_reward_bits."""

    def test_correct_format_and_passed(self):
        """Valid, non-empty code that passes sets the format and passed bits only."""
        bits = compute_reward_bits(parsed=_dummy_parsed(), passed=True)

        assert bits == RewardBit.CORRECT_FORMAT | RewardBit.PASSED_TESTS

    def test_failed_still_credits_format(self):
        """Valid, non-empty code that fails evaluation still gets the format bit."""
        bits = compute_reward_bits(parsed=_dummy_parsed(), passed=False)

        assert bits == RewardBit.CORRECT_FORMAT

    def test_invalid_code_has_no_format_bit(self):
        """Code that didn't parse as valid python doesn't get the format bit."""
        bits = compute_reward_bits(parsed=_dummy_parsed(valid=False), passed=False)

        assert RewardBit.CORRECT_FORMAT not in bits

    def test_empty_code_has_no_format_bit(self):
        """An empty (but technically 'valid') code fragment doesn't get the format bit."""
        bits = compute_reward_bits(parsed=_dummy_parsed(code=""), passed=False)

        assert RewardBit.CORRECT_FORMAT not in bits

    def test_os_import_alone_is_not_an_internal_lib(self):
        """Importing only 'os' (stdin/stdout boilerplate) doesn't set the internal-lib bits."""
        bits = compute_reward_bits(
            parsed=_dummy_parsed(std_libs=["os"], lib_usage={"os": [{"call": "read"}]}),
            passed=False,
        )

        assert RewardBit.INTERNAL_LIB_IMPORTED not in bits
        assert RewardBit.INTERNAL_LIB_USED not in bits

    def test_internal_lib_imported_but_unused(self):
        """A stdlib import with no recorded usage sets imported but not used."""
        bits = compute_reward_bits(
            parsed=_dummy_parsed(std_libs=["math"]), passed=False
        )

        assert RewardBit.INTERNAL_LIB_IMPORTED in bits
        assert RewardBit.INTERNAL_LIB_USED not in bits

    def test_internal_lib_imported_and_used(self):
        """A stdlib import with recorded usage sets both the imported and used bits."""
        bits = compute_reward_bits(
            parsed=_dummy_parsed(
                std_libs=["math"],
                lib_usage={"math": [{"call": "sqrt"}]},
            ),
            passed=False,
        )

        assert RewardBit.INTERNAL_LIB_IMPORTED in bits
        assert RewardBit.INTERNAL_LIB_USED in bits

    def test_external_lib_imported_and_used(self):
        """A third-party import with recorded usage sets both the imported and used bits."""
        bits = compute_reward_bits(
            parsed=_dummy_parsed(
                ext_libs=["numpy"],
                lib_usage={"numpy": [{"call": "array"}]},
            ),
            passed=False,
        )

        assert RewardBit.EXTERNAL_LIB_IMPORTED in bits
        assert RewardBit.EXTERNAL_LIB_USED in bits


class TestComputeReward:
    """Tests for compute_reward."""

    def test_passed_gets_format_and_passed_weights(self):
        """A passing, well-formatted response gets the sum of those two bits' weights."""
        reward = compute_reward(parsed=_dummy_parsed(), passed=True)

        assert reward == (
            REWARD_WEIGHTS[RewardBit.CORRECT_FORMAT]
            + REWARD_WEIGHTS[RewardBit.PASSED_TESTS]
        )

    def test_failed_gets_no_reward_for_passed_bit(self):
        """A response whose code failed evaluation doesn't get the passed-tests weight."""
        reward = compute_reward(parsed=_dummy_parsed(), passed=False)

        assert reward == REWARD_WEIGHTS[RewardBit.CORRECT_FORMAT]
