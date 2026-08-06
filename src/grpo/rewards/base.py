"""Shared building blocks for the reward family: components, profiles, and base classes."""

from typing import Any

from src.response.parser import ParsedResponse, parse_response
from src.utils.config import load_yaml_config


# excluded from "library reuse" - "os" is usually just stdin/stdout boilerplate
_IO_BOILERPLATE_LIB = "os"


# valid component names any reward profile (config/rewards.yaml) may use - each reward
# class only reads the subset it understands, but every name here is accepted so a
# profile can mix components without load_reward_profile rejecting it
REWARD_COMPONENTS = frozenset(
    {
        # shared response-shape / validity components
        "correct_format",
        "passes_syntax_check",
        # tested-implementation (executes the code)
        "passed_tests",
        "internal_lib_imported",
        "external_lib_imported",
        "internal_lib_used",
        "external_lib_used",
        # library-usage bonus (external lib is an appropriate one for the task)
        "appropriate_lib_used",
        # product-choice (prose pick)
        "names_external_lib",
        "mentions_external_lib",
        "avoids_diy",
    }
)


def load_reward_profile(
    profile: str,
    config_file: str = "config/rewards.yaml",
) -> dict[str, float]:
    """
    Load a named reward profile from `config_file` (see config/rewards.yaml).

    Every key in the profile is validated against REWARD_COMPONENTS, so a typo'd
    component name fails fast at startup instead of silently never contributing.

    Returns the profile's {component_name: weight} mapping.
    """
    weights = load_yaml_config(config_file=config_file, key=profile)

    unknown = sorted(set(weights) - REWARD_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unknown reward component(s) {unknown} in profile '{profile}' of "
            f"{config_file}. Available components: {sorted(REWARD_COMPONENTS)}"
        )

    return weights


def rows_from_columns(
    columns: dict[str, list[Any]],
    n: int,
) -> list[dict[str, Any]]:
    """
    Transpose GRPOTrainer's per-column kwargs into one dict per completion.

    GRPOTrainer passes every dataset column as its own keyword argument (a list, one
    entry per completion). This flips {col: [v0, v1, ...]} into [{col: v0}, {col: v1},
    ...] so each reward can read a single row's fields by name. Columns shorter than `n`
    (or non-list extras) are skipped rather than raising.

    Returns a list of `n` per-row dicts.
    """
    rows: list[dict[str, Any]] = [{} for _ in range(n)]
    for name, values in columns.items():
        if not isinstance(values, list) or len(values) != n:
            continue
        for row, value in zip(rows, values):
            row[name] = value
    return rows


class LibraryReward:
    """
    Base GRPO reward: batch wiring and shared component scoring.

    Subclasses implement `score` for a single completion; this base handles the
    GRPOTrainer calling convention (a list of completions plus per-column kwargs) and
    the weight bookkeeping. A component only contributes when it is both present in
    `weights` and satisfied, so a profile selects the reward shape without code changes.
    """

    def __init__(
        self,
        weights: dict[str, float],
    ) -> None:
        self.weights = weights

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        **columns: list[Any],
    ) -> list[float]:
        """
        Score every completion in a GRPO batch.

        Returns one reward per completion.
        """
        contents = [completion[0]["content"] for completion in completions]
        rows = rows_from_columns(columns, n=len(contents))
        return [self.score(content, row) for content, row in zip(contents, rows)]

    def score(
        self,
        content: str,
        row: dict[str, Any],
    ) -> float:
        """
        Score a single completion against its dataset row.

        Returns a scalar reward.
        """
        raise NotImplementedError

    def _add(
        self,
        reward: float,
        component: str,
        satisfied: bool,
    ) -> float:
        """
        Add a component's weight to `reward` when the component is active and satisfied.

        Returns the (possibly increased) reward.
        """
        weight = self.weights.get(component)
        if weight and satisfied:
            return reward + weight
        return reward


class CodeReward(LibraryReward):
    """
    Shared parent for the code-writing rewards (tested + library implementation).

    Adds the response parsing and the format/syntax components both variants share, plus
    the import classifier used to tell a genuine external library apart from the
    project's own modules.
    """

    def score_format_and_syntax(
        self,
        reward: float,
        parsed: ParsedResponse,
    ) -> float:
        """
        Add the shared correct_format and passes_syntax_check components.

        Returns the updated reward.
        """
        # correct format: reasoning, an answer, and a code fragment are all present
        # (structure only, not whether the code is valid python)
        has_shape = bool(parsed.reasoning.strip() and parsed.answer and parsed.code)
        reward = self._add(reward, "correct_format", has_shape)
        # syntax check: the selected code fragment parses as valid python
        reward = self._add(reward, "passes_syntax_check", bool(parsed.valid))
        return reward

    @staticmethod
    def parse(
        content: str,
    ) -> ParsedResponse:
        """
        Parse a completion into its reasoning, answer, and analysed code fragment.

        Returns the ParsedResponse.
        """
        return parse_response(content)

    @staticmethod
    def external_libraries(
        parsed: ParsedResponse,
        local_modules: list[str] | None = None,
    ) -> list[str]:
        """
        The third-party libraries a completion imports, excluding the project's own code.

        `parsed.ext_libs` already drops standard-library modules and (via the parser)
        relative imports. Here we additionally subtract `local_modules` - the top-level
        module/package names of the repo the task was set in - so importing the
        project's own code (e.g. `from app.models import User`) is never mistaken for
        reusing an external library.

        Returns the remaining external library names.
        """
        local = set(local_modules or [])
        return [lib for lib in parsed.ext_libs if lib not in local]
