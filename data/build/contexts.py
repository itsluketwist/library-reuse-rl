"""Turn a vendored test repo into the compact context string a prompt is set in.

Each context is a short stack summary (README head + dependency manifest + file tree) plus,
for python repos, the `local_modules` the repo defines - the top-level import names the
library-usage reward subtracts so importing the project's own code is never mistaken for
reusing an external library (see src/grpo/rewards/library_usage.py). Shared by both the
amplify and library build pipelines.
"""

from dataclasses import dataclass
from pathlib import Path


# known repo directory name -> the stack/language tag used on generated rows. repos not
# listed here have their stack inferred (see _infer_stack), so committed python repos we add
# for the library pipeline are picked up without needing an entry.
REPO_STACK = {
    "nextjs-saas": "nextjs",
    "python-api": "python",
    "react-spa": "react",
    "node-cli": "node",
}

# how much of a repo's README to include (its intent/TODOs, not the whole thing)
_README_CHARS = 1000

# manifests worth showing (dependencies reveal the stack), longest-useful first
_MANIFESTS = ("pyproject.toml", "package.json")


@dataclass
class Context:
    """A repo turned into prompt context, with its language and local module names."""

    context_id: str
    stack: str
    language: str
    summary: str
    local_modules: list[str]


def _file_tree(
    repo_dir: Path,
) -> list[str]:
    """List the repo's source files, relative to the repo root, for the summary.

    Returns the sorted relative paths.
    """
    files = [
        str(path.relative_to(repo_dir))
        for path in repo_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return sorted(files)


def _local_modules(
    repo_dir: Path,
) -> list[str]:
    """The top-level python import names the repo defines (packages and modules).

    A top-level directory containing any .py file is an importable package; a top-level
    .py file is an importable module. These are what a solution would import to reach the
    project's own code, so the reward subtracts them from "external library" detection.

    Returns the sorted module names.
    """
    modules: set[str] = set()
    for entry in repo_dir.iterdir():
        if entry.is_dir() and any(entry.rglob("*.py")):
            modules.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            modules.add(entry.stem)
    return sorted(modules)


def _infer_stack(
    repo_dir: Path,
) -> str:
    """The stack tag for a repo: its known mapping, else inferred from its files.

    A repo with a pyproject.toml or any .py file is treated as python, so committed python
    repos added for the library pipeline work without a REPO_STACK entry.

    Returns the stack tag.
    """
    if repo_dir.name in REPO_STACK:
        return REPO_STACK[repo_dir.name]
    if (repo_dir / "pyproject.toml").exists() or any(repo_dir.rglob("*.py")):
        return "python"
    return "unknown"


def build_context(
    repo_dir: Path,
) -> Context:
    """Summarise a repo into a Context.

    Returns the Context.
    """
    stack = _infer_stack(repo_dir)

    parts = [f"# Project: {repo_dir.name} ({stack})"]

    readme = repo_dir / "README.md"
    if readme.exists():
        parts.append(readme.read_text()[:_README_CHARS].strip())

    for manifest in _MANIFESTS:
        manifest_path = repo_dir / manifest
        if manifest_path.exists():
            parts.append(f"## {manifest}\n{manifest_path.read_text().strip()}")

    tree = _file_tree(repo_dir)
    parts.append("## Files\n" + "\n".join(tree))

    return Context(
        context_id=repo_dir.name,
        stack=stack,
        language="python" if stack == "python" else stack,
        summary="\n\n".join(parts),
        local_modules=_local_modules(repo_dir) if stack == "python" else [],
    )


def build_contexts(
    repos_dir: Path,
) -> dict[str, Context]:
    """Build a Context for every vendored repo.

    Returns a {repo_name: Context} mapping.
    """
    return {
        repo_dir.name: build_context(repo_dir)
        for repo_dir in sorted(repos_dir.iterdir())
        if repo_dir.is_dir()
    }
