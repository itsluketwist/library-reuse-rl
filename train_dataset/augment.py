"""Scaffolding to scale the seed datasets up with an offline LLM (see train_dataset/PLAN.md).

The seed build (pipeline/03_build_dataset.py) turns the four amplify repos and their prompts
into a real but small dataset. This module holds the augmentation seam that scales it:
generating extra natural phrasings per category, synthesising new repo-context skeletons across
stacks, and (later) ingesting real OSS repos. Each generator only builds the LLM prompt and
parses the reply, so it is decoupled from which model runs it - pass any object satisfying
`LLMClient`.

The generation model/budget is deliberately not hard-coded here (it is the main open decision
for scale-up). Wire a client - e.g. a vLLM offline runner, matching src/eval/generate.py - and
call these from a build stage. Until then this module is inert: importing it costs nothing and
the pipeline produces the seed without it.
"""

from typing import Protocol

from train_dataset.contexts import Context


class LLMClient(Protocol):
    """A minimal text-in/text-out client, so generators don't depend on a specific model."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
        ...


# instructs the model to produce open-ended questions that never name a tool - the same
# design rule the amplify prompts follow, so results reflect organic choices
_PHRASING_PROMPT = """\
You are writing benchmark prompts for the "{name}" category ({description}).
Write {n} short, natural questions a developer might ask about this in the project below.

Rules:
- do NOT name any specific tool, library, or service - keep every question open-ended
- vary the phrasing (commands, questions, context-aware asks)
- one question per line, no numbering

Project context:
{context}
"""


def build_phrasing_prompt(
    name: str,
    description: str,
    context: Context,
    n: int,
) -> str:
    """Build the LLM prompt that asks for `n` tool-free phrasings for a category.

    Returns the prompt string.
    """
    return _PHRASING_PROMPT.format(
        name=name,
        description=description,
        n=n,
        context=context.summary,
    )


def parse_phrasings(
    reply: str,
    banned_terms: list[str],
) -> list[str]:
    """Parse generated phrasings, dropping blanks and any that leak a banned tool name.

    `banned_terms` is the category's allowlist (plus aliases): a phrasing that names one of
    them defeats the open-ended design, so it is rejected here rather than trusted.

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


def generate_phrasings(
    client: LLMClient,
    name: str,
    description: str,
    context: Context,
    banned_terms: list[str],
    n: int = 10,
) -> list[str]:
    """Generate `n` extra open-ended phrasings for a category in a context, via `client`.

    Returns the accepted phrasings (tool-free, deduplicated).
    """
    reply = client.generate(build_phrasing_prompt(name, description, context, n))
    phrasings = parse_phrasings(reply, banned_terms)
    # exact-dedup here; the embedding near-dup pass (PLAN.md) plugs in at the build step
    return list(dict.fromkeys(phrasings))


# --- not yet wired (need the generation-model decision) -------------------------------
# - synthesise_context(client, stack, domain) -> Context: generate a greenfield repo
#   skeleton (README + manifest + file tree) for a (stack, domain) pair, then summarise it
#   with train_dataset/contexts.py, deriving local_modules the same way.
# - ingest_oss_repo(url) -> Context: clone a permissively-licensed repo, summarise it, and
#   decontaminate against the eval/test splits before use.
