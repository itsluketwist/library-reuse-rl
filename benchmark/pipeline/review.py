"""Author-review helpers: load validated problems and record keep/reject verdicts, resumably.

The executable gates (stages 02-05) prove a problem runs and is genuinely dual-solvable, but not
that it reads coherently. Stage 06 is a light human coherence check - not a deep audit - over the
survivors: the companion notebook (06_review.ipynb) shows each problem one at a time so the author
can confirm the task makes sense and catch anything that slipped through the automated gates,
recording keep / update / reject. The build stage (07) then enforces that every problem was seen.

Verdicts live in a JSONL under the (gitignored) artefacts directory; the keep/update/reject summary
is captured in the audit REPORT. The file is append-only and the latest entry per seed wins, so
review can span sessions and be revised.
"""

import html
from typing import Literal

from build import artefact_path, latest_records

from src.utils.io import append_jsonl, load_jsonl


# keep = ship as-is, reject = drop, update = ship with the author's edited prompt
Verdict = Literal["keep", "reject", "update"]

# the stage-06 artefact (under the gitignored artefacts directory)
REVIEW_PATH = artefact_path("06_reviews.jsonl")


def validated_problems() -> list[dict]:
    """The problems that passed every executable gate (stage 05 kept), latest record per seed.

    Returns the list of validated problem rows.
    """
    rows = latest_records(artefact_path("05_validated.jsonl")).values()
    return [row for row in rows if row.get("kept")]


def verdicts() -> dict[str, dict]:
    """The latest recorded review verdict per seed (later entries override earlier ones).

    Returns the {seed_id: verdict record} map (empty if nothing has been reviewed yet).
    """
    if not REVIEW_PATH.exists():
        return {}
    latest: dict[str, dict] = {}
    for row in load_jsonl(REVIEW_PATH):
        latest[row["seed_id"]] = row
    return latest


def record_verdict(
    seed_id: str,
    verdict: Verdict,
    note: str = "",
    prompt: str | None = None,
) -> None:
    """Append a verdict for one problem (append-only, latest wins).

    Pass `prompt` for an "update" verdict to store the author's edited prompt as an override.
    """
    record: dict = {"seed_id": seed_id, "verdict": verdict, "note": note}
    if prompt is not None:
        record["prompt"] = prompt
    append_jsonl(record, REVIEW_PATH)


def pending(
    problems: list[dict] | None = None,
) -> list[dict]:
    """Validated problems that do not yet have a review verdict, in order.

    Returns the unreviewed problem rows.
    """
    problems = validated_problems() if problems is None else problems
    reviewed = verdicts()
    return [problem for problem in problems if problem["seed_id"] not in reviewed]


def accepted_ids() -> set[str]:
    """Seed ids that will ship - kept as-is or updated with an edited prompt.

    Returns the set of accepted seed ids.
    """
    return {
        seed
        for seed, row in verdicts().items()
        if row.get("verdict") in ("keep", "update")
    }


def approved_problems() -> list[dict]:
    """Validated problems the author accepted, with any edited prompt applied.

    Keeps problems whose verdict is keep or update; for an update, the stored prompt override
    replaces the original prompt so the improved version is what ships.

    Returns the shippable problem rows.
    """
    reviewed = verdicts()
    approved: list[dict] = []
    for problem in validated_problems():
        record = reviewed.get(problem["seed_id"])
        if not record or record.get("verdict") not in ("keep", "update"):
            continue
        if record.get("prompt"):
            problem = problem | {"prompt": record["prompt"]}
        approved.append(problem)
    return approved


# --- notebook review GUI ---------------------------------------------------------------

# a scrollable, monospace panel for prompt / solution text
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
    problem: dict,
) -> str:
    """Render a problem's (stdin, expected stdout) pairs as an html list (reprs, so whitespace shows).

    Returns the html string.
    """
    items = "".join(
        f"<li><b>stdin</b> {html.escape(repr(stdin))}<br>"
        f"<b>stdout</b> {html.escape(repr(expected))}</li>"
        for stdin, expected in zip(problem["inputs"], problem["expected_outputs"])
    )
    return f"<ul style='font-family: monospace; font-size: 12px;'>{items}</ul>"


def review_gui(
    problems: list[dict] | None = None,
) -> object:
    """Launch the keep/reject review interface for the validated problems.

    Shows one problem at a time - the prompt, its test cases, and both gold solutions in
    collapsible panels - with keep / reject buttons and a notes field. Saves after every click and
    resumes from the first unreviewed problem, so review can span sittings.

    Returns the widget to display in the notebook.
    """
    import ipywidgets as widgets  # lazy: the module must load without a notebook environment

    problems = validated_problems() if problems is None else problems
    reviewed = verdicts()

    # start on the first problem without a verdict
    position = next(
        (i for i, p in enumerate(problems) if p["seed_id"] not in reviewed),
        max(len(problems) - 1, 0),
    )

    header = widgets.HTML()
    # the prompt is editable: tidy it up here and click "update" to ship the edited version
    prompt_editor = widgets.Textarea(
        layout=widgets.Layout(width="100%", height="240px")
    )
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
        """Render the problem at the current position."""
        problem = problems[position]
        seed = problem["seed_id"]
        existing = reviewed.get(seed)
        n_done = sum(1 for p in problems if p["seed_id"] in reviewed)
        header.value = (
            f"<b>reviewed {n_done} / {len(problems)}</b> &nbsp;|&nbsp; {seed} "
            f"&nbsp;|&nbsp; {problem.get('category', '?')} &nbsp;|&nbsp; "
            f"seed_libs {problem.get('seed_libs', [])}"
        )
        # prefill the editor with the author's edited prompt if there is one, else the original
        prompt_editor.value = (
            existing["prompt"]
            if existing and existing.get("prompt")
            else problem["prompt"]
        )
        cases_box.value = _cases_html(problem)
        cases_acc.set_title(0, f"test cases ({len(problem['inputs'])})")
        stdlib_box.value = _code_html(problem["gold_stdlib"])
        stdlib_acc.set_title(
            0, f"gold: stdlib [{problem.get('gold_stdlib_model', '?')}]"
        )
        extlib_box.value = _code_html(problem["gold_extlib"])
        extlib_acc.set_title(
            0,
            f"gold: extlib {problem.get('gold_extlib_packages', [])} "
            f"[{problem.get('gold_extlib_model', '?')}]",
        )
        notes.value = existing.get("note", "") if existing else ""
        status.value = (
            f"current verdict: <b>{existing['verdict']}</b>" if existing else ""
        )

    def _save(
        verdict: Verdict,
        prompt: str | None = None,
    ) -> None:
        """Record a verdict for the current problem and advance to the next unreviewed one.

        For an "update" verdict, `prompt` is the edited prompt to ship in place of the original.
        """
        nonlocal position
        problem = problems[position]
        record_verdict(problem["seed_id"], verdict, notes.value, prompt=prompt)
        record: dict = {
            "seed_id": problem["seed_id"],
            "verdict": verdict,
            "note": notes.value,
        }
        if prompt is not None:
            record["prompt"] = prompt
        reviewed[problem["seed_id"]] = record
        remaining = [i for i, p in enumerate(problems) if p["seed_id"] not in reviewed]
        position = next(
            (i for i in remaining if i > position),
            min(position + 1, len(problems) - 1),
        )
        _show()

    def _back(
        _button: object,
    ) -> None:
        """Step back to the previous problem to revise its verdict."""
        nonlocal position
        position = max(position - 1, 0)
        _show()

    keep_button.on_click(lambda _b: _save("keep"))
    # "update" ships whatever is currently in the editable prompt box
    update_button.on_click(lambda _b: _save("update", prompt_editor.value))
    reject_button.on_click(lambda _b: _save("reject"))
    back_button.on_click(_back)

    _show()
    return widgets.VBox(
        [
            header,
            widgets.HTML(
                "<b>prompt</b> (edit and click <i>update</i> to ship your changes)"
            ),
            prompt_editor,
            cases_acc,
            stdlib_acc,
            extlib_acc,
            notes,
            widgets.HBox([keep_button, update_button, reject_button, back_button]),
            status,
        ]
    )
