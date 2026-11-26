"""Restore the opening reasoning tag a chat template put in the prompt, onto the response."""

from lure import DEFAULT_REASONING_TAGS


def prefilled_reasoning_tag(
    prompt_end: str,
) -> str:
    """
    Find the opening reasoning tag (e.g. "<think>\\n") that a chat template put at the end
    of the prompt, so the model's response starts part way through its reasoning.

    Some templates (e.g. olmo 3's) do this, so a response cut off before
    </think> has no reasoning tags at all, and would be mistaken for an answer.

    Returns the tag with any whitespace after it, or "" if the prompt doesn't end with one.
    """
    stripped = prompt_end.rstrip()
    for opening, _ in DEFAULT_REASONING_TAGS:
        if stripped.lower().endswith(opening.lower()):
            return prompt_end[len(stripped) - len(opening) :]
    return ""
