# Prompt generation comparison: 3 sample tasks, all 8 versions

Generated via `data.build.library.augment`'s generators directly (one call per task, picked from 3 different categories so the comparison isn't dominated by one domain). Each row below is exactly what `build.py` would emit for that (mode, source) pair - no before/after, this is the current state of `augment.py`'s prompts. The `script` claude rows (dataset v3) are the three staged wordings from `script_stage_prompts.json`, loosest first.

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

### mode=`script` source=`llm` stage=`outline`

````
Write a program that reads clustering parameters and a list of points from standard input and prints the density-based (DBSCAN) cluster label of each point.

Input format:
- First line: the radius (a real number) and the minimum group size (an integer), separated by whitespace.
- Each following line: the coordinates of one point, separated by whitespace. All points have the same dimension and distances are Euclidean.

Output format:
- One line per point, in input order: its cluster label, 0, 1, 2, ... for clustered points and -1 for noise. Clusters are numbered from 0 in increasing order of the smallest input position of their core points.

Example:
Input:
1.5 2
0 0
0 1
1 0
10 10
10 11
50 50

Output:
0
0
0
1
1
-1
````

### mode=`script` source=`llm` stage=`precise`

````
Write a program that reads clustering parameters and a list of points from standard input and prints the density-based (DBSCAN) cluster label of each point.

Input format:
- The first non-empty line holds two whitespace-separated values: the radius (a positive real number) and the minimum group size (a positive integer); these are the usual DBSCAN parameters, the neighbourhood count including the point itself.
- Every following non-empty line holds the coordinates of one point as whitespace-separated real numbers. All points have the same dimension and distances are Euclidean. Blank lines are ignored.

Output format:
- Exactly one line per point, in input order, containing its cluster label as an integer: 0, 1, 2, ... for clustered points, -1 for noise.
- Clusters are numbered consecutively from 0, ordered by the smallest input position among their core points.
- A point lying within the radius of core points of more than one cluster belongs to the lowest-numbered of those clusters.
- If the input is empty, or has the parameter line but no points, print nothing.

Example:
Input:
1.5 2
0 0
0 1
1 0
10 10
10 11
50 50

Output:
0
0
0
1
1
-1
````

### mode=`script` source=`llm` stage=`benchmark`

````
Write a program that reads clustering parameters and a set of points from standard input and prints, for each point, the label of the density-based cluster (DBSCAN) it belongs to.

Input format:
- The first non-empty line contains two whitespace-separated values: the radius (a real number greater than 0) and the minimum group size (an integer of at least 1). These are the standard DBSCAN parameters, with the neighbourhood count including the point itself.
- Each subsequent non-empty line contains the coordinates of one point as whitespace-separated real numbers, possibly negative and possibly in exponent notation. Every point has the same number of coordinates (at least one), and distances between points are Euclidean.
- Blank or whitespace-only lines anywhere in the input are ignored. Points may be repeated.

Output format:
- Exactly one line per point, in the order the points were given, containing that point's cluster label as an integer with no extra text or spaces.
- Labels are 0, 1, 2, ... for points belonging to a cluster and -1 for noise points.
- Clusters are numbered consecutively starting at 0, in increasing order of the smallest input position among their core points.
- If a point lies within the radius of core points belonging to more than one cluster, it is assigned to the lowest-numbered of those clusters.
- If the input is completely empty, or contains the parameter line but no point lines, print nothing at all.

The output is a deterministic function of the input only; no randomness, files, network or clock are involved.

Example:
Input:
1.5 2
0 0
0 1
1 0
10 10
10 11
50 50

Output:
0
0
0
1
1
-1
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

### mode=`script` source=`llm` stage=`outline`

````
Write a program that reads a stop-word list and a document from standard input and prints the most common words of the document, ignoring the stop words.

Input format:
- Line 1: an integer N, how many words to report.
- Line 2: the stop words, separated by spaces (this line may be empty).
- All remaining lines: the document text.

Words are maximal runs of ASCII letters and digits, compared case-insensitively.

Output format:
- Up to N lines, each holding a word, a single space, and its count, most frequent first; ties broken alphabetically.

Example:
Input:
3
the a of and is to in
The quick brown fox jumps over the lazy dog. The dog barks, and the fox runs!

Output:
dog 2
fox 2
barks 1
````

### mode=`script` source=`llm` stage=`precise`

````
Write a program that reads a stop-word list and a document from standard input and prints the most common words of the document, ignoring the stop words.

Input format:
- Line 1: an integer N, how many words to report.
- Line 2: the stop words, separated by whitespace (this line may be empty).
- All remaining lines: the document text, which may span any number of lines.

Words are maximal runs of ASCII letters and digits; everything else (punctuation, whitespace, underscores, other characters) separates words. Words are compared and reported in lowercase, and stop words are matched case-insensitively.

Output format:
- One line per reported word: the word in lowercase, a single space, then its number of occurrences.
- Lines are ordered by decreasing count; words with equal counts are ordered alphabetically.
- At most N lines are printed; if fewer distinct non-stop words exist, print all of them.
- If N is zero or negative, if the input is empty, or if the document contains no word after removing stop words, print nothing.

Example:
Input:
3
the a of and is to in
The quick brown fox jumps over the lazy dog. The dog barks, and the fox runs!

Output:
dog 2
fox 2
barks 1
````

### mode=`script` source=`llm` stage=`benchmark`

````
Write a program that reads a stop-word list and a document from standard input and prints a frequency ranking of the document's words, ignoring the stop words.

Input format:
- Line 1: an integer N, the maximum number of words to report.
- Line 2: the stop words, separated by whitespace; this line may be empty, meaning no stop words.
- Line 3 onwards: the document text, which may span any number of lines and may be empty.

Tokenisation and matching rules:
- A word is a maximal run of ASCII letters and digits; every other character (whitespace, punctuation, underscores, symbols, non-ASCII characters) acts as a separator.
- Words are case-insensitive: they are counted and printed in lowercase.
- A word is discarded if it equals any stop word when both are lowercased. Stop words are themselves taken as whitespace-separated tokens exactly as given.

Output format:
- Print one line for each reported word: the lowercase word, a single space, and its number of occurrences in the document, with no extra spaces or punctuation.
- Order the lines by strictly decreasing count; words sharing a count are ordered alphabetically (ASCII order).
- Print at most N lines. If the number of distinct remaining words is smaller than N, print them all.
- Print nothing at all if: the input is completely empty, the first line is not a valid integer, N is zero or negative, there is no document text, or every word of the document is a stop word.
- No trailing blank line beyond the newline ending the last printed line.

The output depends only on the input; there is no randomness, file access, or use of the clock.

Example:
Input:
3
the a of and is to in
The quick brown fox jumps over the lazy dog. The dog barks, and the fox runs!

Output:
dog 2
fox 2
barks 1
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

### mode=`script` source=`llm` stage=`outline`

````
Write a program that reads a list of filenames from standard input, one per line, and prints them sorted in natural (also called human or alphanumeric) order to standard output.

Input format:
- Zero or more lines, each line holding one filename.
- Lines that are blank are ignored.

Output format:
- The filenames, one per line, in natural sort order.
- If there are no filenames, print nothing.

Example:
Input:
file10
file2
file1

Output:
file1
file2
file10
````

### mode=`script` source=`llm` stage=`precise`

````
Write a program that reads a list of filenames from standard input, one per line, and prints them sorted in natural (also called human or alphanumeric) order to standard output.

Input format:
- Zero or more lines, each line holding one filename.
- A line is taken exactly as it appears apart from its line terminator; no other whitespace is trimmed.
- Lines that are empty or contain only whitespace are ignored.

Output format:
- The retained filenames, one per line, in natural sort order, each line ending with a newline.
- Digit runs embedded in a name are ordered by numeric value; the remaining text is ordered by ordinary case-sensitive character comparison, so uppercase letters come before lowercase ones.
- Names that are indistinguishable in this ordering (for example differing only by leading zeros in a number, or exact duplicates) are printed in the order they appeared in the input, and every input name is printed once.
- If no filenames remain after ignoring blank lines, print nothing.

Example:
Input:
file10
file2
file1

Output:
file1
file2
file10
````

### mode=`script` source=`llm` stage=`benchmark`

````
Write a program that reads a list of filenames from standard input, one per line, and prints them sorted in natural (also called human or alphanumeric) order to standard output.

Input format:
- Zero or more lines, each line holding one filename. The input may be completely empty.
- A line is used exactly as it appears apart from its line terminator; no other whitespace is stripped, so a name may contain spaces, dots, dashes or any other characters.
- Lines that are empty or consist only of whitespace are ignored and never printed.

Output format:
- The retained filenames, one per line, in natural sort order, each line terminated by a newline.
- Runs of consecutive digits appearing anywhere in a name are compared by the number they denote rather than character by character; the text around them is compared with ordinary case-sensitive character ordering, so uppercase letters sort before lowercase letters and before other characters according to their usual code point order.
- Leading zeros do not change the value of a number, and a plus or minus sign or a decimal point is treated as ordinary text, not as part of a number.
- A name that is a prefix of another (ignoring the differences above) comes first, e.g. "file" before "file1".
- Names that compare as equal under this ordering, including exact duplicates and names differing only by leading zeros, are printed in the relative order in which they appeared in the input. Every retained input line is printed exactly once, so the output has as many lines as there are non-blank input lines.
- If no filenames remain after ignoring blank lines, print nothing at all.

The output depends only on the input text; there is no randomness, file access or use of the clock.

Example:
Input:
file10
file2
file1

Output:
file1
file2
file10
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
- **`script`** (dataset v3): one exercise per task in LibReuseBench's template, at three levels of
  detail (`outline` / `precise` / `benchmark`). The generator mirrors the benchmark's own rewrite
  prompt: it first writes a private solution that uses a library, then describes that program's
  behaviour without naming the library - so a library's natural output is a correct answer, but
  the task can be solved either way. Style examples are dev-split statements (`LRB/D007`,
  `LRB/D009`), never test. The v2 wording (one statement against `LRB/T050`) pre-digested inputs
  until no library helped, and gave almost no library-use signal.
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
