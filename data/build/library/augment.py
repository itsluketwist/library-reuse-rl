"""LLM seam that diversifies the wording of library-usage tasks (the hybrid layer).

The deterministic build (build.py) already turns each task into several rows via fixed
templates. This module adds extra *natural* wordings per task with an offline LLM, cached to
phrasings.json so the build stays reproducible and needs no model at build time. Each function
only builds the prompt and parses the reply, so it is decoupled from which model runs it - pass
any object satisfying `LLMClient`.

The build reads phrasings.json if present; regenerating it is a deliberate, one-off offline
step. Until a client is wired in this module is inert: importing it costs nothing and the build
produces the template-only dataset without it.
"""

import json
from pathlib import Path
from typing import Protocol


# the cache the build folds in - {task text: [extra bare-mode wordings]}
PHRASINGS_FILE = Path(__file__).parent / "phrasings.json"


class LLMClient(Protocol):
    """A minimal text-in/text-out client, so generators don't depend on a specific model."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
        ...


# asks the model to reword the request only - never naming a library, so the reuse signal
# stays organic (the model under training must choose the library itself, unprompted)
_PHRASING_PROMPT = """\
A developer wants to implement this small feature in Python:
{task}

Write {n} short, natural ways a developer might ask for this to be implemented.

Rules:
- do NOT name any specific library, package, or tool - keep every phrasing tool-agnostic
- vary the phrasing (questions, imperatives, "how do I ...")
- one phrasing per line, no numbering
"""


def build_phrasing_prompt(
    task: str,
    n: int,
) -> str:
    """Build the LLM prompt asking for `n` tool-free rewordings of a task.

    Returns the prompt string.
    """
    return _PHRASING_PROMPT.format(task=task, n=n)


def parse_phrasings(
    reply: str,
    banned_terms: list[str],
) -> list[str]:
    """Parse generated phrasings, dropping blanks and any that leak a banned library name.

    `banned_terms` is the task's category libraries: a phrasing that names one defeats the
    organic-choice design, so it is rejected here rather than trusted.

    Returns the accepted phrasings.
    """
    banned = {term.lower() for term in banned_terms}
    accepted = []
    for line in reply.splitlines():
        text = line.strip().lstrip("-*0123456789. ").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(term in lowered for term in banned):
            continue
        accepted.append(text)
    return accepted


def generate_task_phrasings(
    client: LLMClient,
    task: str,
    banned_terms: list[str],
    n: int = 5,
) -> list[str]:
    """Generate `n` extra tool-free wordings for a task via `client`.

    Returns the accepted phrasings (deduplicated).
    """
    reply = client.generate(build_phrasing_prompt(task, n))
    phrasings = parse_phrasings(reply, banned_terms)
    return list(dict.fromkeys(phrasings))


def load_phrasings_cache() -> dict[str, list[str]]:
    """Load the committed phrasings cache, or an empty mapping if it is absent.

    Returns the {task text: [wordings]} mapping.
    """
    if PHRASINGS_FILE.exists():
        return json.loads(PHRASINGS_FILE.read_text())
    return {}


def write_phrasings_cache(
    phrasings: dict[str, list[str]],
) -> None:
    """Write the phrasings cache to phrasings.json (the offline regeneration step)."""
    PHRASINGS_FILE.write_text(
        json.dumps(phrasings, indent=2, ensure_ascii=False) + "\n"
    )
