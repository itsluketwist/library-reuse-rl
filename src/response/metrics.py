"""Code size and maintainability metrics for a model's code, measured with radon."""

from radon.metrics import mi_visit
from radon.raw import analyze


def code_metrics(
    code: str,
) -> tuple[int | None, float | None]:
    """
    Compute the logical lines of code and maintainability index for a code fragment.

    The fragment came from an LLM response rather than trusted source, so any parse
    failure (radon parses with both ast and tokenize internally) is caught and
    reported as missing metrics rather than raising.

    Returns a (logical_lines, maintainability_index) tuple, with None values on parse failure.
    """
    try:
        logical_lines = analyze(code).lloc
        maintainability_index = mi_visit(code, multi=True)
    except Exception:
        return None, None

    return logical_lines, maintainability_index
