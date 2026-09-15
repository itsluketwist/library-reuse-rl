# Prompt generation comparison: 3 sample tasks, all 6 versions

Generated via `data.build.library.augment`'s generators directly (one call per task, picked from 3 different categories so the comparison isn't dominated by one domain). Each row below is exactly what `build.py` would emit for that (mode, source) pair - no before/after, this is the current state of `augment.py`'s prompts.

## Task: `Cluster a set of points by density given a radius and a minimum group size.`  _(category: `clustering`)_

### mode=`repo` source=`template` context=`python-api`

````
# Project: python-api (python)

# DataPipeline API

A REST API for managing data processing jobs. Built for a client who needs to:
- Upload CSV/Excel files
- Process and transform data
- Schedule recurring jobs
- Get notified when jobs complete

## Current Status
- Basic FastAPI structure
- No database yet (storing nothing)
- No authentication
- No file storage
- No job queue

## Planned Features
- User authentication (API keys? OAuth?)
- File upload and storage
- Background job processing
- Email notifications on completion
- Usage tracking for billing
- Rate limiting

## Tech Stack
- Python 3.11
- FastAPI
- Uvicorn

## Running

```bash
pip install -e .
uvicorn src.main:app --reload
```

## API Endpoints (Planned)

```
POST /auth/register    - Create account
POST /auth/login       - Get token
POST /jobs             - Create new processing job
GET  /jobs             - List user's jobs
GET  /jobs/{id}        - Get job status
POST /jobs/{id}/upload - Upload file for job
GET  /jobs/{id}/result - Download processed resul

## pyproject.toml
[project]
name = "python-api"
version = "0.1.0"
description = "A FastAPI starter project"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

## Files
README.md
pyproject.toml
src/__init__.py
src/main.py
src/models.py

Task: Cluster a set of points by density given a radius and a minimum group size.

Implement this in Python.
````

### mode=`repo` source=`llm` context=`python-api`

````
In `src/models.py`, which currently contains:

```python
"""
Database models for DataPipeline API

TODO: Pick an ORM and set this up properly
Options considered:
- SQLAlchemy (most popular, but verbose)
- SQLModel (by FastAPI creator, nice integration)
- Tortoise ORM (async-first)
- Raw SQL with asyncpg?

For now, just defining the schema we'll need.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class User:
    id: str
    email: str
    hashed_password: str
    api_key: Optional[str]
    created_at: datetime


@dataclass
class Job:
    id: str
    user_id: str
    name: str
    description: Optional[str]
    status: JobStatus
    schedule: Optional[str]  # cron expression
    input_file_url: Optional[str]  # where we stored their upload
    output_file_url: Optional[str]  # where to get the result
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


@dataclass
class UsageRecord:
    """Track API usage for billing"""
    id: str
    user_id: str
    job_id: str
    bytes_processed: int
    processing_time_ms: int
    created_at: datetime


# TODO: Set up actual database tables
# Need to decide on:
# 1. Which database? (Postgres seems right for this)
# 2. Which ORM?
# 3. Migrations strategy

```

Add a function to this codebase that groups a set of 2D points into clusters based on density, given a radius and a minimum number of points per group, and put it alongside our existing models.
````

### mode=`simple` source=`template`

````
Write a short Python program to cluster a set of points by density given a radius and a minimum group size.
````

### mode=`simple` source=`llm`

````
Can you write me a function that groups a bunch of 2D points into clusters based on how close they are to each other, where I specify a distance radius and the minimum number of points needed to count as a cluster? Points that don't fit into any dense group should just be marked as noise. Something like cluster_points(points, radius, min_points) returning a list of cluster labels would be great.
````

### mode=`script` source=`template`

````
Write a Python program that reads the necessary input from standard input and prints the result to standard output, to cluster a set of points by density given a radius and a minimum group size.
````

### mode=`script` source=`llm`

````
Write a program that reads a set of 2-D points and groups them into density-based clusters, then reports for each point (in its original order) the cluster it belongs to, or -1 if it is noise.

A point's neighborhood radius eps and a minimum neighborhood size minPts are given. For a given point, count how many points (including itself) lie within distance eps of it (distance is the ordinary Euclidean distance, and equality with eps counts as "within"). A point whose neighborhood size is at least minPts is called a core point. Two core points that are within eps of each other belong to the same cluster, and clusters are formed by connecting core points transitively through such direct links. A non-core point that lies within eps of at least one core point is a border point and joins the cluster of that core point (if it is within eps of core points from more than one cluster, it joins the cluster of the lowest-indexed such core point, using 0-based input order). Any point that is neither a core point nor a border point is noise.

Clusters must be numbered starting from 0, with numbers assigned in the order the clusters are first discovered while scanning points in their original input order and expanding each newly found cluster before moving on (this is the standard order in which such a clustering procedure discovers clusters).

Input format:
- The first line contains three values: an integer n (the number of points), a real number eps, and an integer minPts, separated by spaces.
- Each of the next n lines contains two real numbers x and y, the coordinates of one point.

Output format:
- Print n lines. The i-th line must contain the cluster number (a non-negative integer) assigned to the i-th input point, or -1 if that point is noise.

Example:
Input:
6 1.5 3
0 0
0 1
1 0
10 10
10 11
11 10

Output:
0
0
0
1
1
1
````

---

## Task: `Count the most common words in a document, ignoring stop words.`  _(category: `nlp-tokenize`)_

### mode=`repo` source=`template` context=`pytoolbox`

````
# Project: pytoolbox (python)

# pytoolbox

A small internal Python toolbox and CLI where the team drops the little helper commands that
don't belong anywhere else - one place for the day-to-day data, file, text, media and web
chores that come up across our projects.

It is deliberately broad: if you find yourself writing the same one-off script twice, add it
here as a `toolbox` subcommand so everyone can reuse it.

## Layout

- `toolbox/cli.py` — the command-line entry point and subcommand dispatch.
- `toolbox/tasks.py` — the individual helper commands.
- `toolbox/utils.py` — shared helpers (io, formatting) used across tasks.

## Running

```
toolbox <command> [options]
```

## TODO

Plenty still to add - roughly in priority order:

- more file and data helpers (parsing, converting, validating, summarising)
- text and document utilities (extracting, rendering, matching, cleaning)
- media helpers (images, audio, codes) and simple charts/reports
- web and network chores (fetching, validating inputs, talking to servic

## pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "pytoolbox"
version = "0.1.0"
description = "Internal Python toolbox CLI for small data, file, text, media and web chores."
requires-python = ">=3.11"
dependencies = []

[project.scripts]
toolbox = "toolbox.cli:main"

## Files
README.md
pyproject.toml
toolbox/__init__.py
toolbox/cli.py
toolbox/tasks.py
toolbox/utils.py

We need a new helper in this project: count the most common words in a document, ignoring stop words. Write it in Python.
````

### mode=`repo` source=`llm` context=`pytoolbox`

````
In `toolbox/tasks.py`, which currently contains:

```python
"""The individual toolbox commands - small, focused helpers the team reuses.

New commands go here; keep each one small and register it in cli.py. These are intentionally
minimal stubs - real behaviour is filled in as the helpers are needed.
"""

from pathlib import Path

from toolbox.utils import read_text, write_text


def count_lines(
    path: str | Path,
) -> int:
    """Count the lines in a text file.

    Returns the line count.
    """
    return len(read_text(path).splitlines())


def slugify_title(
    title: str,
) -> str:
    """Turn a title into a rough url-safe slug (placeholder - improve as needed).

    Returns the slug.
    """
    return "-".join(title.lower().split())


def summarise_file(
    path: str | Path,
    out_path: str | Path,
) -> None:
    """Write a tiny summary (line count) of a file to another file."""
    summary = f"{Path(path).name}: {count_lines(path)} lines\n"
    write_text(out_path, summary)

```

Add a new subcommand to this project that reads a document and reports the most frequent words, filtering out common stop words before counting.
````

### mode=`simple` source=`template`

````
How do I count the most common words in a document, ignoring stop words in Python?
````

### mode=`simple` source=`llm`

````
Can you write a function called top_words that takes in a string of text and returns a list of the most frequent words, but skips common stop words like 'the', 'and', 'is'? I'd like it to return the top N words along with their counts.
````

### mode=`script` source=`template`

````
Write a script that takes its input on standard input and prints its output to standard output: count the most common words in a document, ignoring stop words.
````

### mode=`script` source=`llm`

````
Write a program that reads a document's text along with a list of stop words and prints the most frequent remaining words.

Input format:
- The first line contains two integers n and k separated by a space: n is the number of top words to report, and k is the number of stop words.
- The second line contains the k stop words, separated by single spaces (all lowercase).
- All remaining lines (until end of input) contain the document text.

Processing rules:
- Treat the document as case-insensitive: convert everything to lowercase before counting.
- A word is a maximal run of letters and digits; punctuation and whitespace separate words.
- Discard any word that appears in the stop-word list.
- Count the occurrences of each remaining word across the whole document.

Output format:
- Print exactly n lines, each containing a word and its count separated by a single space, listing the n most frequent words in descending order of count.
- Break ties between words with equal counts by alphabetical order (ascending).
- Assume there are always at least n distinct qualifying words.

Example:
Input:
3 2
the a
the quick brown fox jumps over the lazy dog the fox runs a quick race a fox wins

Output:
fox 3
quick 2
brown 1
````

---

## Task: `Sort a list of filenames like file2, file10 in natural order.`  _(category: `natural-sort`)_

### mode=`repo` source=`template` context=`pytoolbox`

````
# Project: pytoolbox (python)

# pytoolbox

A small internal Python toolbox and CLI where the team drops the little helper commands that
don't belong anywhere else - one place for the day-to-day data, file, text, media and web
chores that come up across our projects.

It is deliberately broad: if you find yourself writing the same one-off script twice, add it
here as a `toolbox` subcommand so everyone can reuse it.

## Layout

- `toolbox/cli.py` — the command-line entry point and subcommand dispatch.
- `toolbox/tasks.py` — the individual helper commands.
- `toolbox/utils.py` — shared helpers (io, formatting) used across tasks.

## Running

```
toolbox <command> [options]
```

## TODO

Plenty still to add - roughly in priority order:

- more file and data helpers (parsing, converting, validating, summarising)
- text and document utilities (extracting, rendering, matching, cleaning)
- media helpers (images, audio, codes) and simple charts/reports
- web and network chores (fetching, validating inputs, talking to servic

## pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "pytoolbox"
version = "0.1.0"
description = "Internal Python toolbox CLI for small data, file, text, media and web chores."
requires-python = ">=3.11"
dependencies = []

[project.scripts]
toolbox = "toolbox.cli:main"

## Files
README.md
pyproject.toml
toolbox/__init__.py
toolbox/cli.py
toolbox/tasks.py
toolbox/utils.py

Task: Sort a list of filenames like file2, file10 in natural order.

Implement this in Python.
````

### mode=`repo` source=`llm` context=`pytoolbox`

````
In `toolbox/utils.py`, which currently contains:

```python
"""Shared helpers used across the toolbox commands."""

from pathlib import Path


def read_text(
    path: str | Path,
) -> str:
    """Read a text file's contents.

    Returns the file text.
    """
    return Path(path).read_text(encoding="utf-8")


def write_text(
    path: str | Path,
    content: str,
) -> None:
    """Write text to a file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def print_rows(
    rows: list[dict],
) -> None:
    """Print a list of record dicts, one per line, for quick inspection."""
    for row in rows:
        print(row)

```

Add a helper to this codebase that sorts filenames in natural order, so file2 comes before file10 instead of after, and wire it into the relevant task.
````

### mode=`simple` source=`template`

````
How do I sort a list of filenames like file2, file10 in natural order in Python?
````

### mode=`simple` source=`llm`

````
Can you write a function that sorts a list of filenames so that file2 comes before file10, instead of the usual string ordering putting file10 first?
````

### mode=`script` source=`template`

````
Write a script that takes its input on standard input and prints its output to standard output: sort a list of filenames like file2, file10 in natural order.
````

### mode=`script` source=`llm`

````
Write a program that reads a list of filenames from standard input and prints them sorted in natural order, meaning that embedded numeric substrings are compared by their numeric value rather than character by character, so that "file2" comes before "file10". Comparison of the non-numeric parts should be done in ordinary lexicographic order, and when two filenames are identical after this comparison rule, their original relative order must be preserved.

Input format:
- The first line contains a single integer n, the number of filenames.
- Each of the next n lines contains one filename (a string that may mix letters, digits, dots, and underscores, with no internal spaces).

Output format:
- Print the n filenames, one per line, sorted according to the natural order rule described above.

Example:
Input:
4
file10
file2
file1
file20

Output:
file1
file2
file10
file20
````

---

## Notes

- **`repo`**: now grounded in the actual current content of a real file, not just its path. The
  generator asks the LLM to choose one existing file from the project layout; `augment.py` reads
  that file's real content off disk and embeds it as a fenced code block before the request, so
  the row a model trains on shows genuine surrounding code, not just an abstract "this project".
  If the chosen file doesn't exist, or its real content itself leaks a library name, the whole
  attempt is discarded rather than rendered ungrounded or leaked (see the `uuid-slugs` finding
  below for a case where this now reliably fires).
- **`simple`**: a short chat-style request that may optionally reference a plausible function
  name/signature in prose, restricted to plain built-in types so an invented signature can't
  itself hint at a specific library. If two attempts with a signature both leak, the generator
  now automatically retries without asking for a signature at all, up to 3 more times - see the
  `uuid-slugs` finding below for why that fallback exists.
- **`script`**: anchored to a real embedded LibReuseBench example (`LRB/T050`) plus an explicit
  "don't over-enumerate edge cases" instruction, so it reads as a decisive exercise statement
  rather than a hedge-everything spec.
- **The `uuid-slugs` collision, revisited**: task "Turn an article title into a url-safe slug."
  (seed library `slugify`) is still excluded from the table above, but for a new reason. `simple`
  now succeeds reliably via the no-signature fallback (confirmed: `"Can you write something that
  takes an article title and converts it into a clean, url-safe slug, like lowercasing it and
  replacing spaces with dashes?"`). `repo`, however, now fails *permanently* rather than
  occasionally: pytoolbox's `toolbox/tasks.py` already contains a placeholder function named
  `slugify_title` (a pre-existing stub, apparently anticipating this exact category), and once
  file content is leak-checked, any attempt that lands on that file - which is genuinely the
  right file for this feature - is discarded every time, deterministically, not just on a bad
  roll. The fix is correct (better an empty cache entry than a leaked one), but the practical
  effect is that this one task can't get a `repo`+`llm` row until either the model is steered to
  a different file (worse fit) or `slugify_title` is renamed in the fixture repo itself (e.g. to
  `title_to_slug`) - flagged here rather than changed, since it touches an authored fixture file
  outside this comparison's scope.
