"""Human-review helpers: show each task the agent review kept, and record keep/reject verdicts.

The earlier stages prove a task runs (02) and that an independent model finds it clear and fair
(03). Human review comes last, and decides what ships. A companion notebook (04_review.ipynb)
shows each task one at a time - its text, its tests, both gold solutions, how much code the
library saves and the agent's notes - so the reviewer can record keep / update / reject. The build
stage (05) then refuses to make the final splits until every task has been seen.

Every function here takes a `track` ("reuse" or "control"), defaulting to "reuse", which selects
both the pool of tasks and the verdicts file - the two tracks are reviewed independently. Verdicts
live in benchmark/artefacts/04_reviews.jsonl (04_reviews_control.jsonl for the control track). The
file is append-only and the latest entry per reviewer and seed wins, so review can span sessions
and be revised.

Review is a two-author process, and both authors independently review the whole pool (not a split
of it). Each verdict carries the reviewing author's anonymous id - "author_1", "author_2", never a
real name, since this repo must stay anonymous for submission. Each reviewer's GUI resumes from
their OWN first-unreviewed task and never shows the other author's verdict, note or edited text -
that would bias an independent second look. A task only ships once both authors have weighed in
and neither rejected it: reject wins on any disagreement (see `approved_tasks`).

A verdict also carries the fingerprint of the task it was given for. If a task is regenerated
afterwards, its old verdicts no longer count, and it has to be reviewed again.
"""

import html
import re
from pathlib import Path
from typing import Literal

from build import agent_kept_tasks, agent_reviews, track_artefact_path

from src.utils.io import append_jsonl, load_jsonl


# keep = ship as-is, reject = drop, update = ship with the author's edited task text
Verdict = Literal["keep", "reject", "update"]

# the two reviewers this project is set up for - a task only ships once both have weighed in
REVIEWERS: tuple[str, ...] = ("author_1", "author_2")


def review_path(
    track: str = "reuse",
) -> Path:
    """Path to a track's human-review verdicts artefact.

    Returns the full path.
    """
    return track_artefact_path("04_reviews.jsonl", track)


def verdicts(
    track: str = "reuse",
) -> dict[str, dict[str, dict]]:
    """Every reviewer's latest verdict per seed for a track, nested by author.

    Returns {seed_id: {author: verdict record}} (empty if nothing has been reviewed yet).
    """
    path = review_path(track)
    if not path.exists():
        return {}
    latest: dict[str, dict[str, dict]] = {}
    for row in load_jsonl(path):
        latest.setdefault(row["seed_id"], {})[row["author"]] = row
    return latest


def task_verdicts(
    task: dict,
    all_verdicts: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """The verdicts that count for a task: those given for its current version.

    Returns the {author: verdict record} map.
    """
    return {
        author: row
        for author, row in all_verdicts.get(task["seed_id"], {}).items()
        if row.get("task_hash") == task["task_hash"]
    }


def record_verdict(
    task: dict,
    verdict: Verdict,
    note: str = "",
    prompt: str | None = None,
    track: str = "reuse",
    author: str = "",
) -> dict:
    """Append a verdict for one task (append-only, latest wins).

    Pass `prompt` for an "update" verdict to store the author's edited task text as an override,
    and `author` to attribute the review - with two reviewers sharing a track, this is what tells
    their verdicts apart.

    Returns the record that was written.
    """
    record: dict = {
        "seed_id": task["seed_id"],
        "task_hash": task["task_hash"],
        "verdict": verdict,
        "note": note,
        "author": author,
    }
    if prompt is not None:
        record["prompt"] = prompt
    append_jsonl(record, review_path(track))
    return record


def pending(
    tasks: list[dict] | None = None,
    track: str = "reuse",
    author: str | None = None,
) -> list[dict]:
    """Tasks still missing a verdict, in order.

    With `author`, missing means that reviewer specifically hasn't recorded one yet - the review
    GUI uses this to resume each author's own, independent pass over the pool. Without one, missing
    means at least one of the two REVIEWERS hasn't - the build gate uses this stricter check, since
    a task only ships once both authors have independently weighed in.

    Returns the unreviewed task records.
    """
    tasks = agent_kept_tasks(track) if tasks is None else tasks
    all_verdicts = verdicts(track)
    needed = REVIEWERS if author is None else (author,)
    return [
        task
        for task in tasks
        if not all(reviewer in task_verdicts(task, all_verdicts) for reviewer in needed)
    ]


def approved_tasks(
    track: str = "reuse",
    tasks: list[dict] | None = None,
) -> list[dict]:
    """Tasks both authors accepted, with an edited task text applied if either left one.

    A task ships only once both REVIEWERS have recorded a verdict for its current version, and
    neither rejected it: reject wins on any disagreement. If more than one reviewer edited the
    text, the first reviewer in `REVIEWERS` order whose edit was recorded ships.

    Returns the shippable task records.
    """
    tasks = agent_kept_tasks(track) if tasks is None else tasks
    all_verdicts = verdicts(track)
    approved: list[dict] = []
    for task in tasks:
        by_author = task_verdicts(task, all_verdicts)
        if not all(
            by_author.get(reviewer, {}).get("verdict") in ("keep", "update")
            for reviewer in REVIEWERS
        ):
            continue
        prompt = next(
            (
                by_author[reviewer]["prompt"]
                for reviewer in REVIEWERS
                if by_author[reviewer].get("prompt")
            ),
            None,
        )
        approved.append(task | {"prompt": prompt} if prompt else task)
    return approved


# --- notebook review GUI ---------------------------------------------------------------

# a scrollable, monospace panel for solution text
_BOX_STYLE = (
    "max-height: 320px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; "
    "white-space: pre-wrap; font-family: monospace; font-size: 12px; line-height: 1.4;"
)


def _code_html(
    source: str,
) -> str:
    """Wrap source text in a scrollable, escaped, monospace html panel.

    Returns the html string.
    """
    return f"<div style='{_BOX_STYLE}'>{html.escape(source.strip())}</div>"


def _cases_html(
    task: dict,
) -> str:
    """Render a task's tests as an html list: each call, and the result it expects.

    Returns the html string.
    """
    items = "".join(
        f"<li>{html.escape(task['function_name'])}(*{html.escape(arguments)})<br>"
        f"<b>&rarr;</b> {html.escape(expected)}</li>"
        for arguments, expected in zip(task["test_inputs"], task["expected_outputs"])
    )
    return f"<ul style='font-family: monospace; font-size: 12px;'>{items}</ul>"


def _reuse_html(
    task: dict,
) -> str:
    """Describe how much code the library saves on a task (nothing for a control task).

    Returns the html string.
    """
    if "reuse_value" not in task:
        return "<b>control task</b>: no library should help here"
    return (
        f"<b>reuse value {task['reuse_value']}</b> ({html.escape(task['reuse_tier'])} "
        f"tier): the standard-library solution is {task['stdlib_lloc']} logical lines, "
        f"the {html.escape(task['primary_library'])} solution is {task['extlib_lloc']}"
    )


def _agent_html(
    review: dict | None,
) -> str:
    """Render the agent review of a task: its note, its answers and how the blind solves did.

    Returns the html string.
    """
    if review is None:
        return "no agent review"
    blind = "; ".join(
        f"{kind} solve {'passed' if solve['passed'] else 'FAILED'}"
        + (f" (used {', '.join(solve['libraries'])})" if solve["libraries"] else "")
        for kind, solve in review.get("blind", {}).items()
    )
    failed = [question for question, answer in review["rubric"].items() if not answer]
    return (
        f"<b>verdict</b> {html.escape(review['verdict'])}<br>"
        f"<b>note</b> {html.escape(review['note'] or '(none)')}<br>"
        f"<b>blind solves</b> {html.escape(blind)}<br>"
        f"<b>questions answered no</b> {html.escape(', '.join(failed) or '(none)')}"
    )


_AUTHOR_PATTERN = re.compile(r"^author_[1-9]\d*$")


def review_gui(
    author: str,
    tasks: list[dict] | None = None,
    track: str = "reuse",
) -> object:
    """Launch the keep/reject review interface for the tasks of one track.

    `author` identifies whoever is reviewing in this session (set it in the notebook) - it's
    stored on every verdict this session records, so two reviewers sharing a track can tell their
    reviews apart later. Must be an anonymous id of the form "author_1", "author_2", ... - never a
    real name, since this repo must stay anonymous for submission.

    Shows one task at a time - its text, how much code the library saves, and collapsible panels
    for the agent's review, the tests and the gold solution(s) - with keep / update / reject
    buttons and a notes field. Saves after every click and resumes from the first task THIS author
    hasn't given a verdict yet, so both authors independently review the whole pool and review can
    span sittings. Only this author's own past verdict for a task is ever shown - never the other
    author's, so an independent second look isn't biased. Control-track tasks have no library
    solution, so that panel is only shown when the task has one.

    Returns the widget to display in the notebook.
    """
    import ipywidgets as widgets  # lazy: the module must load without a notebook environment

    if not _AUTHOR_PATTERN.match(author):
        raise ValueError(
            f'author must look like "author_1", "author_2", ... (got {author!r}) - '
            "no real names, this repo must stay anonymous for submission"
        )

    tasks = agent_kept_tasks(track) if tasks is None else tasks
    agent = agent_reviews(track)
    all_verdicts = verdicts(track)
    # this author's own verdict on each task's current version, by seed
    reviewed = {
        task["seed_id"]: task_verdicts(task, all_verdicts)[author]
        for task in tasks
        if author in task_verdicts(task, all_verdicts)
    }

    # start on the first task THIS author has no verdict for yet
    position = next(
        (i for i, task in enumerate(tasks) if task["seed_id"] not in reviewed),
        max(len(tasks) - 1, 0),
    )

    header = widgets.HTML()
    # the task text is editable: tidy it up here and click "update" to ship the edited version
    prompt_editor = widgets.Textarea(
        layout=widgets.Layout(width="100%", height="240px")
    )
    reuse_box = widgets.HTML()
    agent_box = widgets.HTML()
    agent_acc = widgets.Accordion(children=[agent_box], selected_index=None)
    cases_box = widgets.HTML()
    cases_acc = widgets.Accordion(children=[cases_box], selected_index=None)
    stdlib_box = widgets.HTML()
    stdlib_acc = widgets.Accordion(children=[stdlib_box], selected_index=None)
    extlib_box = widgets.HTML()
    extlib_acc = widgets.Accordion(children=[extlib_box], selected_index=None)
    notes = widgets.Text(description="notes:", layout=widgets.Layout(width="80%"))
    status = widgets.HTML()

    button_layout = widgets.Layout(width="auto")
    keep_button = widgets.Button(
        description="keep", button_style="success", layout=button_layout
    )
    update_button = widgets.Button(
        description="update", button_style="info", layout=button_layout
    )
    reject_button = widgets.Button(
        description="reject", button_style="danger", layout=button_layout
    )
    back_button = widgets.Button(
        description="← back", button_style="warning", layout=button_layout
    )

    def _show() -> None:
        """Render the task at the current position."""
        task = tasks[position]
        seed = task["seed_id"]
        existing = reviewed.get(seed)
        header.value = (
            f"<b>reviewing as {html.escape(author)}</b> &nbsp;|&nbsp; "
            f"<b>reviewed {len(reviewed)} / {len(tasks)}</b> &nbsp;|&nbsp; {seed} "
            f"&nbsp;|&nbsp; {html.escape(task.get('domain', '?'))} &nbsp;|&nbsp; "
            f"{html.escape(task.get('category', '?'))}"
        )
        # prefill the editor with THIS author's own edited text if there is one, else the original
        prompt_editor.value = (
            existing["prompt"]
            if existing and existing.get("prompt")
            else task["prompt"]
        )
        reuse_box.value = _reuse_html(task)
        agent_box.value = _agent_html(agent.get(seed))
        agent_acc.set_title(0, "agent review")
        cases_box.value = _cases_html(task)
        cases_acc.set_title(0, f"tests ({len(task['test_inputs'])})")
        stdlib_box.value = _code_html(task["stdlib_solution"])
        stdlib_acc.set_title(0, "gold: standard library")
        # control-track tasks have no library gold - hide that panel entirely for them
        has_extlib = "extlib_solution" in task
        extlib_acc.layout.display = "" if has_extlib else "none"
        if has_extlib:
            extlib_box.value = _code_html(task["extlib_solution"])
            extlib_acc.set_title(
                0,
                f"gold: library {task.get('gold_extlib_packages', [])}",
            )
        notes.value = existing.get("note", "") if existing else ""
        status.value = f"your verdict: <b>{existing['verdict']}</b>" if existing else ""

    def _save(
        verdict: Verdict,
        prompt: str | None = None,
    ) -> None:
        """Record a verdict for the current task and advance to the next unreviewed one.

        For an "update" verdict, `prompt` is the edited task text to ship in place of the original.
        """
        nonlocal position
        task = tasks[position]
        reviewed[task["seed_id"]] = record_verdict(
            task,
            verdict,
            notes.value,
            prompt=prompt,
            track=track,
            author=author,
        )
        remaining = [i for i, t in enumerate(tasks) if t["seed_id"] not in reviewed]
        position = next(
            (i for i in remaining if i > position),
            min(position + 1, len(tasks) - 1),
        )
        _show()

    def _back(
        _button: object,
    ) -> None:
        """Step back to the previous task to revise its verdict."""
        nonlocal position
        position = max(position - 1, 0)
        _show()

    keep_button.on_click(lambda _b: _save("keep"))
    # "update" ships whatever is currently in the editable text box
    update_button.on_click(lambda _b: _save("update", prompt_editor.value))
    reject_button.on_click(lambda _b: _save("reject"))
    back_button.on_click(_back)

    _show()
    return widgets.VBox(
        [
            header,
            widgets.HTML(
                "<b>task text</b> (edit and click <i>update</i> to ship your changes)"
            ),
            prompt_editor,
            reuse_box,
            agent_acc,
            cases_acc,
            stdlib_acc,
            extlib_acc,
            notes,
            widgets.HBox([keep_button, update_button, reject_button, back_button]),
            status,
        ]
    )
