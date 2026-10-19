"""Library-usage reward functions: score a response for reusing external libraries."""

from collections.abc import Hashable, Iterable, Mapping

from lure.extraction import (
    DEFAULT_REASONING_TAGS,
    ReasoningTags,
    normalise_reasoning_tags,
)
from lure.packages import pip_package
from lure.parsing import ParsedResponse, parse_response


# library or module names, given either as one list shared by every response, or as a dict
# mapping each dataset item's id to that item's own list
NamesOption = Iterable[str] | Mapping[Hashable, Iterable[str]]


# every reward component, in the order they are checked
COMPONENTS: tuple[str, ...] = (
    "correct_format",
    "passes_syntax_check",
    "external_lib_imported",
    "external_lib_used",
    "appropriate_lib_used",
)


def is_appropriate_library(
    library: str,
    appropriate_libraries: Iterable[str],
) -> bool:
    """
    Check whether a library is one of the appropriate libraries. The check ignores case
    and accepts either the import name or the PyPI name (so `PIL` matches "pillow").

    Returns True when the library matches one of the appropriate libraries.
    """
    wanted = {name.casefold() for name in appropriate_libraries}
    return library.casefold() in wanted or pip_package(library).casefold() in wanted


def components_from_parsed(
    parsed: ParsedResponse,
    appropriate_libraries: Iterable[str] = (),
    require_reasoning: bool = True,
) -> dict[str, bool]:
    """
    Check each reward component against an already-parsed response.

    Returns each component name mapped to whether the response satisfies it.
    """
    # a well-formed response has an answer (and reasoning, for thinking models) and was
    # not cut off by the token limit
    has_reasoning = bool(parsed.reasoning) or not require_reasoning
    correct_format = has_reasoning and bool(parsed.answer) and not parsed.truncated

    appropriate = list(appropriate_libraries)
    return {
        "correct_format": correct_format,
        "passes_syntax_check": parsed.valid,
        "external_lib_imported": bool(parsed.external_imports),
        "external_lib_used": bool(parsed.external_used),
        "appropriate_lib_used": any(
            is_appropriate_library(library, appropriate)
            for library in parsed.external_used
        ),
    }


def library_components(
    response: str | None,
    stop_reason: str | None = None,
    local_modules: Iterable[str] = (),
    appropriate_libraries: Iterable[str] = (),
    require_reasoning: bool = True,
    reasoning_tags: ReasoningTags = DEFAULT_REASONING_TAGS,
    ignore_placeholders: bool = True,
) -> dict[str, bool]:
    """
    Parse a response and check each reward component, for logging or custom rewards.

    With `ignore_placeholders` (the default), placeholder modules such as `your_module`
    never count as libraries (see analysis.is_placeholder_module).

    Returns each component name mapped to whether the response satisfies it.
    """
    parsed = parse_response(
        response,
        stop_reason=stop_reason,
        local_modules=local_modules,
        reasoning_tags=reasoning_tags,
        require_reasoning=require_reasoning,
        ignore_placeholders=ignore_placeholders,
    )
    return components_from_parsed(
        parsed,
        appropriate_libraries=appropriate_libraries,
        require_reasoning=require_reasoning,
    )


def _freeze(
    names: NamesOption,
) -> tuple[str, ...] | dict[Hashable, tuple[str, ...]]:
    """
    Copy a names option into tuples, so later changes to the caller's lists have no effect.

    Returns a tuple of names, or a dict of tuples when names were given per item id.
    """
    # check for a dict first, since iterating a dict would silently give just its keys
    if isinstance(names, Mapping):
        return {item_id: tuple(item_names) for item_id, item_names in names.items()}
    return tuple(names)


def _choose_names(
    option: str,
    given: Iterable[str] | None,
    default: tuple[str, ...] | dict[Hashable, tuple[str, ...]],
    item_id: Hashable | None,
) -> Iterable[str]:
    """
    Choose the names to use for one response: names passed with the response win, then
    the item's entry when the defaults are a dict keyed by item id, then the default list.

    A missing item id raises an error rather than falling back to no names, since silently
    dropping `local_modules` would count the project's own code as external libraries.

    Returns the chosen names.
    """
    if given is not None:
        return given
    if not isinstance(default, dict):
        return default
    if item_id is None:
        raise ValueError(
            f"{option} was given per item id, so pass item_id when scoring a response.",
        )
    if item_id not in default:
        raise ValueError(f"No {option} given for item_id {item_id!r}.")
    return default[item_id]


class LibraryReward:
    """
    A reward function that scores an LLM response for reusing external libraries.

    Calling it with a response returns the total weight of the components the response
    satisfies. The `local_modules` and `appropriate_libraries` given when it is created are
    either one list for every response, or a dict of lists keyed by dataset item id (then
    each call passes its `item_id`). A list passed when calling overrides both.
    """

    def __init__(
        self,
        weights: dict[str, float],
        local_modules: NamesOption = (),
        appropriate_libraries: NamesOption = (),
        require_reasoning: bool = True,
        reasoning_tags: ReasoningTags = DEFAULT_REASONING_TAGS,
        ignore_placeholders: bool = True,
    ) -> None:
        unknown = set(weights) - set(COMPONENTS)
        if unknown:
            raise ValueError(
                f"Unknown reward components: {sorted(unknown)}. Choose from {COMPONENTS}.",
            )

        self.weights = dict(weights)
        self.local_modules = _freeze(local_modules)
        self.appropriate_libraries = _freeze(appropriate_libraries)
        self.require_reasoning = require_reasoning
        self.ignore_placeholders = ignore_placeholders
        # checked now, so bad tags fail when the reward is created rather than mid-training
        self.reasoning_tags = normalise_reasoning_tags(reasoning_tags)
        # training frameworks (e.g. TRL) use a reward function's __name__ when logging
        self.__name__ = "library_reward"

    @property
    def max_reward(
        self,
    ) -> float:
        """
        The highest reward possible, useful for normalising rewards to [0, 1].

        Returns the sum of the positive component weights.
        """
        return sum(weight for weight in self.weights.values() if weight > 0)

    def components(
        self,
        response: str | None,
        stop_reason: str | None = None,
        item_id: Hashable | None = None,
        local_modules: Iterable[str] | None = None,
        appropriate_libraries: Iterable[str] | None = None,
    ) -> dict[str, bool]:
        """
        Check each reward component for one response.

        Returns each component name mapped to whether the response satisfies it.
        """
        return library_components(
            response,
            stop_reason=stop_reason,
            local_modules=_choose_names(
                "local_modules",
                given=local_modules,
                default=self.local_modules,
                item_id=item_id,
            ),
            appropriate_libraries=_choose_names(
                "appropriate_libraries",
                given=appropriate_libraries,
                default=self.appropriate_libraries,
                item_id=item_id,
            ),
            require_reasoning=self.require_reasoning,
            reasoning_tags=self.reasoning_tags,
            ignore_placeholders=self.ignore_placeholders,
        )

    def __call__(
        self,
        response: str | None,
        stop_reason: str | None = None,
        item_id: Hashable | None = None,
        local_modules: Iterable[str] | None = None,
        appropriate_libraries: Iterable[str] | None = None,
    ) -> float:
        """
        Score one response.

        Returns the total weight of the components the response satisfies.
        """
        components = self.components(
            response,
            stop_reason=stop_reason,
            item_id=item_id,
            local_modules=local_modules,
            appropriate_libraries=appropriate_libraries,
        )
        return float(
            sum(
                self.weights.get(component, 0.0)
                for component, satisfied in components.items()
                if satisfied
            ),
        )


def create_library_reward(
    *,
    correct_format: float = 0.25,
    passes_syntax_check: float = 0.25,
    external_lib_imported: float = 0.5,
    external_lib_used: float = 1.0,
    appropriate_lib_used: float = 0.0,
    local_modules: NamesOption = (),
    appropriate_libraries: NamesOption = (),
    require_reasoning: bool = True,
    reasoning_tags: ReasoningTags = DEFAULT_REASONING_TAGS,
    ignore_placeholders: bool = True,
) -> LibraryReward:
    """
    Create a reward function that scores LLM responses for reusing external libraries.

    Each component adds its weight to the reward when the response satisfies it:
    - correct_format: the response has reasoning (unless `require_reasoning` is False) and
      an answer, and was not cut off by the token limit. With `require_reasoning`, a
      response with no reasoning tags is unfinished reasoning, so it has no answer (and no
      code) at all.
    - passes_syntax_check: the answer has python code, and all of it is valid.
    - external_lib_imported: the code imports an external (non-standard-library) library.
    - external_lib_used: the code imports an external library and then actually uses it.
    - appropriate_lib_used: the code uses one of `appropriate_libraries`.

    `local_modules` are the project's own modules, which are never counted as libraries.
    With `ignore_placeholders` (the default), neither are placeholder modules such as
    `your_module` or `utils`, which stand for code the response doesn't have (see
    analysis.is_placeholder_module) - set it to False to count every import as before.
    Both `local_modules` and `appropriate_libraries` can be one list for every response, or
    a dict mapping each dataset item's id to its own list - then pass the response's
    `item_id` when scoring it. `reasoning_tags` are the (opening, closing) pairs that wrap
    the model's reasoning, e.g. [("[THINK]", "[/THINK]")], defaulting to <think>-style tags.
    Set a weight to 0 to switch that component off.

    Returns a LibraryReward, called as `reward_func(response, stop_reason=..., item_id=...)`.
    """
    return LibraryReward(
        weights={
            "correct_format": correct_format,
            "passes_syntax_check": passes_syntax_check,
            "external_lib_imported": external_lib_imported,
            "external_lib_used": external_lib_used,
            "appropriate_lib_used": appropriate_lib_used,
        },
        local_modules=local_modules,
        appropriate_libraries=appropriate_libraries,
        require_reasoning=require_reasoning,
        reasoning_tags=reasoning_tags,
        ignore_placeholders=ignore_placeholders,
    )
