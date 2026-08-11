"""Download the amplifying.ai "claude-code-picks" seed the amplify pipeline is built from.

Source: https://github.com/amplifying-ai/claude-code-picks (the dataset behind
https://amplifying.ai/research/claude-code-picks).

The seed provides the amplify pipeline's raw material: the open-ended benchmark prompts, the
per-category tool-frequency extractions (used to seed the allowlists), and four greenfield
test repos used as prompt context. The library pipeline also reuses the vendored python-api
repo as one of its repo contexts, so this stage is a shared prerequisite - run it first. The
upstream repo ships no licence file, so we do not commit its content: this stage fetches it at
build time into data/build/vendor/ (gitignored), pinned to a commit for reproducibility.

Usage: python data/build/amplify/vendor_seed.py
"""

import io
import tarfile
import urllib.request

from data.build.paths import VENDOR_DIR


# commit pinned for reproducibility (dated 2026-02-17)
REVISION = "60c74f9725f5347c9d628edbbf7d89d695b6c65d"

REPO = "amplifying-ai/claude-code-picks"

# only these top-level paths from the archive are kept - the prompts, the combined
# extraction analysis, and the test repos we turn into prompt contexts
KEEP_PREFIXES = (
    "prompts/",
    "results/combined-extracted.json",
    "custom-repos/",
)


def _archive_url() -> str:
    """The github codeload tarball URL for the pinned revision.

    Returns the URL.
    """
    return f"https://codeload.github.com/{REPO}/tar.gz/{REVISION}"


def _wanted(
    member_path: str,
) -> str | None:
    """Map an archive member path to its destination under VENDOR_DIR, or None to skip.

    The tarball wraps everything in a top-level "<repo>-<sha>/" directory, which we strip
    before matching against KEEP_PREFIXES.

    Returns the repo-relative path to keep, or None if the member isn't wanted.
    """
    # drop the leading "<repo>-<sha>/" component
    _, _, relative = member_path.partition("/")
    if relative and any(relative.startswith(prefix) for prefix in KEEP_PREFIXES):
        return relative
    return None


def download_seed() -> None:
    """Fetch and extract the wanted seed files into data/build/vendor/."""
    url = _archive_url()
    print(f"downloading {REPO}@{REVISION[:8]} from {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - trusted github host
        raw = response.read()

    kept = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = _wanted(member.name)
            if relative is None:
                continue

            destination = VENDOR_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            destination.write_bytes(extracted.read())
            kept += 1

    print(f"wrote {kept} files to {VENDOR_DIR}/")


def main() -> None:
    """Download the amplify seed, then write a short provenance NOTICE."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    download_seed()

    # record where the data came from, since it isn't committed with a licence
    notice = (
        f"Seed data vendored from https://github.com/{REPO}\n"
        f"Pinned revision: {REVISION}\n"
        "Fetched by data/build/amplify/vendor_seed.py; not committed to this repo.\n"
        "See https://amplifying.ai/research/claude-code-picks for the accompanying study.\n"
    )
    (VENDOR_DIR / "NOTICE").write_text(notice)
    print(f"wrote provenance NOTICE to {VENDOR_DIR / 'NOTICE'}")


if __name__ == "__main__":
    main()
