"""Extract final-answer Python without inspecting model reasoning."""

import ast
import re
from dataclasses import dataclass


_REASONING_OPEN = re.compile(
    r"^\s*<(?:think|thinking|reasoning|thought)>\s*",
    re.IGNORECASE,
)
_REASONING_CLOSE = re.compile(
    r"</(?:think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)
_FENCE = re.compile(
    r"```[ \t]*(?P<language>[^\n`]*)\r?\n(?P<code>.*?)```",
    re.DOTALL,
)
_PYTHON_LANGUAGES = {"", "py", "python", "python3"}


@dataclass(frozen=True)
class CodeCandidate:
    """One Python fragment extracted from a final answer."""

    code: str
    source: str
    language: str


@dataclass(frozen=True)
class ExtractedAnswer:
    """A reasoning-free final answer and its Python candidates."""

    answer: str | None
    candidates: list[CodeCandidate]


def strip_embedded_reasoning(content: str | None) -> str | None:
    """Remove a leading tagged reasoning trace from message content.

    Returns final-answer text, or None when no complete final answer exists.
    """
    if content is None:
        return None
    if not content.strip():
        return None

    opening = _REASONING_OPEN.search(content)
    if opening is None:
        return content.strip()
    closing = _REASONING_CLOSE.search(content, opening.end())
    if closing is None:
        return None
    answer = content[closing.end() :].strip()
    return answer or None


def extract_python_candidates(answer: str | None) -> list[CodeCandidate]:
    """Extract Python fences, falling back to a wholly parseable raw answer.

    Returns candidates in response order; prose is never treated as executable code.
    """
    if answer is None:
        return []

    matches = list(_FENCE.finditer(answer))
    if matches:
        candidates = []
        for match in matches:
            language = match.group("language").strip().lower()
            if language not in _PYTHON_LANGUAGES:
                continue
            code = match.group("code").strip()
            if code:
                candidates.append(
                    CodeCandidate(
                        code=code,
                        source="fence",
                        language=language or "python",
                    )
                )
        return candidates

    # an unterminated fence is malformed output, not a raw-python response.
    if "```" in answer:
        return []
    try:
        ast.parse(answer)
    except SyntaxError:
        return []
    return [
        CodeCandidate(
            code=answer.strip(),
            source="raw",
            language="python",
        )
    ]


def extract_answer(content: str | None) -> ExtractedAnswer:
    """Strip embedded reasoning and extract Python from final content only.

    Returns the final answer and extracted Python candidates.
    """
    answer = strip_embedded_reasoning(content)
    return ExtractedAnswer(
        answer=answer,
        candidates=extract_python_candidates(answer),
    )
