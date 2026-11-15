# Prompt generation comparison: 3 sample tasks, 7 wordings each

Every sample below is copied word for word from the built dataset (`data/train/library_usage/`,
v5): for 3 tasks from 3 different categories, each row `build.py` emits for the task, except the
project-framed `repo` row (a fixed template). The `function` rows (dataset v4) are the three
staged requests from `function_stage_prompts.json`, plainest first; they replace v3's four
`script` rows. After a rebuild, copy the rows again so the samples stay the dataset's own.

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

Add a function to this codebase that groups a list of points into clusters based on density, given a radius and minimum group size, so we can use it when processing uploaded job data.
````

### mode=`simple` source=`template`

````
How do I cluster a set of points by density given a radius and a minimum group size in Python?
````

### mode=`simple` source=`llm`

````
Can you write a function that groups a list of 2D points into clusters based on how close they are to each other, where a group only counts if it has at least a certain number of points within a given radius? I'd like to pass in the points, the radius, and the minimum group size, and get back a list of cluster labels for each point.
````

### mode=`function` source=`llm` stage=`name`

````
Write a function called `cluster_points_by_density` that groups a set of 2D points into density-based clusters and marks isolated points as noise.
````

### mode=`function` source=`llm` stage=`signature`

````
Write a function `cluster_points_by_density(points, radius, min_points)` that takes a list of (x, y) points, a neighbourhood radius and the minimum number of points needed to form a dense region, and clusters the points by density (DBSCAN-style). It returns a list with one integer label per point, in input order, using -1 for noise.
````

### mode=`function` source=`llm` stage=`example`

````
Write a function `cluster_points_by_density(points, radius, min_points)` that takes a list of (x, y) points, a neighbourhood radius and the minimum number of points needed to form a dense region (counting the point itself), and clusters the points by density (DBSCAN-style). It returns a list with one integer label per point, in input order. Clusters are numbered 0, 1, 2, ... in the order they are first reached when going through the points, and noise points get -1. For example, `cluster_points_by_density([(0, 0), (0, 1), (1, 0), (10, 10), (10, 11), (11, 10), (50, 50)], 1.5, 3)` returns `[0, 0, 0, 1, 1, 1, -1]`.
````

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

Task: Count the most common words in a document, ignoring stop words.

Implement this in Python.
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

Add a new subcommand to this project that reads a text file and prints the most frequent words, filtering out common stop words like "the" and "and".
````

### mode=`simple` source=`template`

````
Write a short Python program to count the most common words in a document, ignoring stop words.
````

### mode=`simple` source=`llm`

````
Can you write a function called top_words that takes a chunk of text and a count n, strips out common stop words like 'the', 'and', 'is', and returns the n most frequent remaining words along with how many times each appears?
````

### mode=`function` source=`llm` stage=`name`

````
Write a function called `most_common_words` that finds the most frequently used words in a document, ignoring stop words.
````

### mode=`function` source=`llm` stage=`signature`

````
Write a function `most_common_words(text, n)` that takes a document as a string and an integer n, and returns the n most common words in it, ignoring English stop words and case. The result is a list of (word, count) tuples, most frequent first.
````

### mode=`function` source=`llm` stage=`example`

````
Write a function `most_common_words(text, n)` that takes a document as a string and an integer n, and returns the n most common words in it, ignoring English stop words and case. The result is a list of (word, count) tuples with lowercase words, most frequent first. For example, `most_common_words('The cat chased the dog, and the cat caught the dog. The cat is happy.', 2)` returns `[('cat', 3), ('dog', 2)]`.
````

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

We need a new helper in this project: sort a list of filenames like file2, file10 in natural order. Write it in Python.
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

Add a helper to this codebase that sorts filenames in natural order, so file2 comes before file10 instead of alphabetical order putting file10 first.
````

### mode=`simple` source=`template`

````
How do I sort a list of filenames like file2, file10 in natural order in Python?
````

### mode=`simple` source=`llm`

````
Can you write a function that sorts a list of filenames so file2 comes before file10 instead of alphabetical order putting file10 first?
````

### mode=`function` source=`llm` stage=`name`

````
Write a function called `sort_filenames_naturally` that sorts filenames so that numbered names like file2 and file10 appear in the order a person would expect.
````

### mode=`function` source=`llm` stage=`signature`

````
Write a function `sort_filenames_naturally(filenames)` that takes a list of filename strings and returns a new list with them sorted in natural order, so that numbers inside the names are compared by value (file2 comes before file10).
````

### mode=`function` source=`llm` stage=`example`

````
Write a function `sort_filenames_naturally(filenames)` that takes a list of filename strings and returns a new list with them sorted in natural order, so that numbers inside the names are compared by value (file2 comes before file10). For example, `sort_filenames_naturally(['file10.txt', 'file2.txt', 'file1.txt'])` returns `['file1.txt', 'file2.txt', 'file10.txt']`.
````

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
- **`script`** (dataset v3): one exercise per task in LibReuseBench's template, at three levels of
  detail (`outline` / `precise` / `benchmark`). The generator mirrors the rewrite prompt of the
  benchmark's archived stdin/stdout design (`benchmark/v1/`): it first writes a private solution
  that uses a library, then describes that program's behaviour without naming the library - so a
  library's natural output is a correct answer, but the task can be solved either way. Style
  examples are two of that design's dev-split statements (`LRB/D007` and `LRB/D009` in
  `benchmark/v1/data/dev.jsonl`), never test. The v2 wording (one statement against that design's
  `LRB/T050`) pre-digested inputs
  until no library helped, and gave almost no library-use signal. Since dataset v4 the script
  rows are no longer in the dataset, so they are no longer shown above.
- **`function`** (dataset v4): one function per task in the style of LibReuseBench's
  function tasks, at three levels of detail (`name` / `signature` / `example`), taken from
  `function_stage_prompts.json`. The request explains what makes a wording *contested* (as
  reasonable to write by hand as with a library) and why each rule keeps the choice open. Like
  the benchmark's, the signatures carry no type hints, and the return type is given in words.
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
