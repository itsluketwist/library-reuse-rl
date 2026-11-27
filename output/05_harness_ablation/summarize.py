"""Validate the saved cohort and regenerate tables/plots without model calls."""

import argparse
import csv
import hashlib
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = {"ml4": "ML4", "glm53": "GLM-5.3", "deepseek_flash": "DeepSeek Flash"}
ARMS = {"direct": "Direct API", "vibe": "Vibe", "mini_swe": "mini-swe"}


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def mean(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return statistics.mean(values) if values else None


def summarize(rows, audits):
    """Keep splits separate and retain missing telemetry and length denominators."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["model"], row["arm"]].append(row)
    summaries = []
    for split in ["test", "control"]:
        for model in MODELS:
            for arm in ARMS:
                group = groups.get((split, model, arm), [])
                if not group:
                    continue
                audit = (
                    [r for r in audits if r["model"] == model and r["arm"] == arm]
                    if split == "test"
                    else []
                )
                high = [
                    r
                    for r in group
                    if r["reuse_value"] is not None and r["reuse_value"] >= 15
                ]
                words = sum(
                    r["thinking_words"]
                    for r in group
                    if r["thinking_words"] is not None
                )
                repeated = sum(
                    r["repeated_words"]
                    for r in group
                    if r["repeated_words"] is not None
                )
                n = len(group)
                summaries.append(
                    {
                        "model": model,
                        "arm": arm,
                        "split": split,
                        "tasks": n,
                        "passed": sum(r["passed"] for r in group),
                        "pass_at_1": sum(r["passed"] for r in group) / n,
                        "not_run": sum(not r["ran"] for r in group),
                        "imported": sum(bool(r["imported_libraries"]) for r in group),
                        "used_original": sum(
                            bool(r["used_libraries_original"]) for r in group
                        ),
                        "used_corrected": sum(
                            bool(r["used_libraries_corrected"]) for r in group
                        ),
                        "high_reuse_tasks": len(high),
                        "high_reuse_used": sum(
                            bool(r["used_libraries_corrected"]) for r in high
                        ),
                        "mean_thinking_tokens": mean(group, "thinking_tokens"),
                        "mean_output_tokens": mean(group, "output_tokens"),
                        "thinking_observed_tasks": sum(
                            r["thinking_tokens"] is not None for r in group
                        ),
                        "output_observed_tasks": sum(
                            r["output_tokens"] is not None for r in group
                        ),
                        "api_calls": sum(
                            r["api_calls"] for r in group if r["api_calls"] is not None
                        ),
                        "usage_reported_calls": sum(
                            r["usage_reported_calls"]
                            for r in group
                            if r["usage_reported_calls"] is not None
                        ),
                        "mean_solution_lloc": mean(group, "solution_lloc"),
                        "lloc_measured_submissions": sum(
                            r["solution_lloc"] is not None for r in group
                        ),
                        "thinking_words": words,
                        "repeated_words": repeated,
                        "repeated_word_share": repeated / words if words else None,
                        "repeated_intent_tasks": sum(
                            r["repeated_intent"] is True for r in group
                        ),
                        "library_only_exercised": sum(
                            r["qualifies_library_only"] for r in audit
                        )
                        if split == "test"
                        else None,
                        "mixed_or_partial_exercised": sum(
                            r["category"] == "mixed_or_partial_used" for r in audit
                        )
                        if split == "test"
                        else None,
                        "no_library_call_on_tests": sum(
                            r["no_library_calls_on_tests"] for r in audit
                        )
                        if split == "test"
                        else None,
                        "manual_or_mixed_source": sum(
                            r["structural_category"] == "manual_or_mixed" for r in audit
                        )
                        if split == "test"
                        else None,
                        "any_inactive_import": sum(
                            r["some_import_inactive_on_tests"] for r in audit
                        )
                        if split == "test"
                        else None,
                    }
                )
    return summaries


def markdown(summary):
    lines = [
        "# LibReuseBench results",
        "",
        "Generated from the included records by `summarize.py`. One sample per task; all nine test conditions cover 216/216 tasks. Controls are separate.",
        "",
        "## Test split",
        "",
        "| Model | Arm | Passes / tasks | Pass@1 | Mean thinking | Mean output | Imports | Corrected use | Mean LLOC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tests = [s for s in summary if s["split"] == "test"]
    for s in tests:
        lines.append(
            f"| {MODELS[s['model']]} | {ARMS[s['arm']]} | {s['passed']}/{s['tasks']} | {s['pass_at_1']:.1%} | {s['mean_thinking_tokens']:,.0f} | {s['mean_output_tokens']:,.0f} | {s['imported']}/216 | {s['used_corrected']}/216 | {s['mean_solution_lloc']:.1f} |"
        )
    lines += [
        "",
        "Output includes thinking, visible text and tool-call arguments, summed across reported calls. LLOC is logical lines in the final solution; its denominator can be smaller than 216. Static usage is a possible reference, not proof of execution.",
        "",
        "## Analyzer correction, coverage and repetition",
        "",
        "| Model | Arm | Original → corrected use | High-reuse used / 52 | Measurable LLOC | Calls with usage / recorded calls | Repeated thinking words | Repeated intent tasks |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in tests:
        lines.append(
            f"| {MODELS[s['model']]} | {ARMS[s['arm']]} | {s['used_original']} → {s['used_corrected']} | {s['high_reuse_used']}/52 | {s['lloc_measured_submissions']}/216 | {s['usage_reported_calls']}/{s['api_calls']} | {s['repeated_word_share']:.2%} | {s['repeated_intent_tasks']}/216 |"
        )
    lines += [
        "",
        "All tasks have some reported output/thinking usage. Recorded attempts include transport/rate-limit retries, particularly for GLM; attempts without usage are not necessarily generated completions. Partial/deadline call usage is not estimated. Repetition ratios pool word counts. A repeated-intent flag is a heuristic, not proof of infinite generation.",
        "",
        "## Separate source and runtime audit",
        "",
        "| Model | Arm | Imported | Library-only, exercised | Mixed/partial, exercised | No library call on tests | Manual/mixed source | Any inactive import |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in tests:
        lines.append(
            f"| {MODELS[s['model']]} | {ARMS[s['arm']]} | {s['imported']} | {s['library_only_exercised']} | {s['mixed_or_partial_exercised']} | {s['no_library_call_on_tests']} | {s['manual_or_mixed_source']} | {s['any_inactive_import']} |"
        )
    lines += [
        "",
        "The 26 library-only, 50 mixed/partial-with-call and 14 no-call solutions partition the 90 imported programs. The 61 manual/mixed source labels and 15 inactive-import programs are different, overlapping measurements. Audit observations cover 1,022 saved test inputs; correctness scores are not replaced by the audit.",
        "",
        "## Historical ML4 controls",
        "",
        "Only ML4 ran the 84 control tasks. They are not included in the main model comparison or its plots.",
        "",
        "| Arm | Passes / tasks | Pass@1 | Imported | Corrected used | Mean thinking | Mean output |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        if s["split"] == "control":
            lines.append(
                f"| {ARMS[s['arm']]} | {s['passed']}/{s['tasks']} | {s['pass_at_1']:.1%} | {s['imported']} | {s['used_corrected']} | {s['mean_thinking_tokens']:,.0f} | {s['mean_output_tokens']:,.0f} |"
            )
    return "\n".join(lines) + "\n"


def validate(rows, audits):
    keys = [
        (r["model"], r["arm"], r["split"], r["task_id"], r["replica"]) for r in rows
    ]
    assert len(keys) == len(set(keys)) == 2196
    for split, models, count in [("test", MODELS, 216), ("control", ["ml4"], 84)]:
        expected_tasks = None
        for model in models:
            for arm in ARMS:
                selected = [
                    r
                    for r in rows
                    if (r["split"], r["model"], r["arm"]) == (split, model, arm)
                ]
                assert len(selected) == count
                tasks = {r["task_id"] for r in selected}
                if expected_tasks is None:
                    expected_tasks = tasks
                assert tasks == expected_tasks
    for r in rows:
        code = r["solution"]
        assert r["solution_sha256"] == (
            hashlib.sha256(code.encode()).hexdigest() if code is not None else None
        )
        assert isinstance(r["passed"], bool) and isinstance(r["ran"], bool)
        assert set(r["used_libraries_corrected"]) <= set(r["imported_libraries"])
    tests = [r for r in rows if r["split"] == "test"]
    assert sum(bool(r["used_libraries_original"]) for r in tests) == 61
    assert sum(bool(r["used_libraries_corrected"]) for r in tests) == 90
    assert (
        sum(r["used_libraries_original"] != r["used_libraries_corrected"] for r in rows)
        == 32
    )
    expected = {
        (r["model"], r["arm"], r["task_id"]) for r in tests if r["imported_libraries"]
    }
    assert {(r["model"], r["arm"], r["task_id"]) for r in audits} == expected
    assert len(audits) == 90
    assert sum(r["test_inputs_executed"] for r in audits) == 1022
    runtime = read_rows("audit/runtime_observations.jsonl")
    assert len(runtime) == 90 and not any(r["errors"] for r in runtime)


def plot(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    tests = {(s["model"], s["arm"]): s for s in summary if s["split"] == "test"}
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    for name, fields in [
        (
            "tokens",
            [
                ("mean_thinking_tokens", "Thinking tokens"),
                ("mean_output_tokens", "Total output tokens"),
            ],
        ),
        (
            "outcomes",
            [
                ("pass_at_1", "Pass@1 (%)"),
                ("used_corrected", "Corrected static library use (%)"),
                ("mean_solution_lloc", "Solution length (logical lines)"),
            ],
        ),
    ]:
        fig, axes = plt.subplots(
            1, len(fields), figsize=(6 * len(fields), 4.6), layout="constrained"
        )
        for ax, (field, title) in zip(axes, fields):
            for i, arm in enumerate(ARMS):
                values = [tests[model, arm][field] for model in MODELS]
                if field == "pass_at_1":
                    values = [100 * v for v in values]
                if field == "used_corrected":
                    values = [100 * v / 216 for v in values]
                bars = ax.bar(
                    [j + (i - 1) * 0.25 for j in range(3)],
                    values,
                    0.24,
                    label=ARMS[arm],
                    color=colors[i],
                )
                ax.bar_label(
                    bars,
                    labels=[
                        f"{v:,.0f}"
                        if field.startswith("mean_") and field.endswith("tokens")
                        else f"{v:.1f}"
                        for v in values
                    ],
                    fontsize=8,
                    padding=3,
                )
            ax.set_xticks(range(3), MODELS.values())
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=10,
        )
        fig.suptitle("LibReuseBench · 216 test tasks per condition", fontsize=14)
        out = ROOT / "charts"
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=180)
        fig.savefig(
            out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate summary JSON/CSV and RESULTS.md",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Regenerate PNG/PDF figures (requires matplotlib)",
    )
    parser.add_argument(
        "--verify-static",
        action="store_true",
        help="Verify all saved corrected library sets using the installed lure",
    )
    args = parser.parse_args()
    rows = read_rows("submissions.jsonl") + read_rows("ml4_controls.jsonl")
    audits = read_rows("audit/classifications.jsonl")
    validate(rows, audits)
    summary = summarize(rows, audits)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(summary[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    outputs = {
        "summary.json": json.dumps(summary, indent=2) + "\n",
        "summary.csv": buffer.getvalue(),
        "RESULTS.md": markdown(summary),
    }
    for name, content in outputs.items():
        if args.write:
            (ROOT / name).write_text(content)
        else:
            assert (ROOT / name).read_text() == content, (
                f"Stale {name}: rerun with --write"
            )
    if args.verify_static:
        from lure import analyse_code

        for r in rows:
            block = analyse_code(r["solution"]) if r["solution"] is not None else None
            assert (
                sorted(block.external_imports if block else [])
                == r["imported_libraries"]
            ), (r["model"], r["arm"], r["task_id"], "imports")
            assert (
                sorted(block.external_used if block else [])
                == r["used_libraries_corrected"]
            ), (r["model"], r["arm"], r["task_id"], "usage")
    if args.plots:
        plot(summary)
    print(
        f"Validated {len(rows)} submissions, {len(summary)} conditions and {len(audits)} audited programs."
    )


if __name__ == "__main__":
    main()
