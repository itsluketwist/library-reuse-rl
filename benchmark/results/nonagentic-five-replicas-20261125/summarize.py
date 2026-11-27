"""Verify the exported native-format pairs and regenerate tables without model calls."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
        actual,
        expected,
    )


def verify(check_static=False):
    """Reconcile every response with its hashes, scores and complete cohort."""
    protocol = read("protocol.json")
    reference = read("reference_analysis.json")
    files = read("files.json")
    assert len(files) == 15
    keys, task_ids, all_rows = set(), None, []
    replicas = []
    for entry in files:
        pair = read(entry["analysis"])
        metadata, analysis = pair["metadata"], pair["analysis"]
        rows = [
            json.loads(line)
            for line in (ROOT / entry["generations"]).read_text().splitlines()
        ]
        assert len(rows) == 216
        assert metadata["replica"] == entry["replica"]
        assert metadata["system_prompt"] is None and metadata["prompt_style"] == "base"
        assert metadata["decoding_settings"] == {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 32768,
            "enable_thinking": True,
            "reasoning_effort": "high",
            "n_samples": 1,
        }
        ids = {r["task_id"] for r in rows}
        assert len(ids) == 216
        if task_ids is None:
            task_ids = ids
        assert ids == task_ids
        for row in rows:
            assert row["idx"] == 0
            assert (
                row["model_key"] == entry["model"]
                and row["replica"] == entry["replica"]
            )
            key = row["model_key"], row["replica"], row["task_id"]
            assert key not in keys
            keys.add(key)
            assert digest(row["response"]) == row["response_sha256"]
            assert digest(row["submitted_answer"]) == row["submitted_answer_sha256"]
            if row["response"] == row["submitted_answer"]:
                reasoning = ""
            else:
                suffix = "</think>" + row["submitted_answer"]
                assert row["response"].startswith("<think>") and row[
                    "response"
                ].endswith(suffix)
                reasoning = row["response"][7 : -len(suffix)]
            assert digest(reasoning) == row["reasoning_sha256"]
            assert (
                row["completion_tokens"] is not None
                and row["reasoning_tokens"] is not None
            )
            assert 0 <= row["reasoning_tokens"] <= row["completion_tokens"]
            assert row["stop_reason"] in ["eos", "length"]
            assert row["external_lib"] == bool(row["external_libs"])
            assert row["external_lib_used"] == bool(row["used_external_libs"])
            assert set(row["used_external_libs"]) <= set(row["external_libs"])
            if check_static:
                from lure import parse_response

                block = parse_response(row["submitted_answer"]).longest_block
                full = parse_response(row["response"]).longest_block
                code = block.code if block else None
                assert (full.code if full else None) == code
                assert (digest(code) if code is not None else None) == row[
                    "solution_sha256"
                ]
                assert (
                    sorted(block.external_imports if block else [])
                    == row["external_libs"]
                )
                assert (
                    sorted(block.external_used if block else [])
                    == row["used_external_libs"]
                )
        passed = sum(r["passed"] for r in rows)
        assert (
            analysis["n_tasks"] == analysis["n_responses"] == analysis["n_total"] == 216
        )
        assert analysis["k"] == 1
        close(analysis["pass@1"], passed / 216)
        replicas.append(
            {
                "model": entry["model"],
                "replica": entry["replica"],
                "scored": 216,
                "passed": passed,
                "pass_at_1": passed / 216,
                "imported": sum(r["external_lib"] for r in rows),
                "used": sum(r["external_lib_used"] for r in rows),
                "mean_thinking_tokens": statistics.mean(
                    r["reasoning_tokens"] for r in rows
                ),
                "mean_output_tokens": statistics.mean(
                    r["completion_tokens"] for r in rows
                ),
            }
        )
        all_rows.extend(rows)
    assert len(keys) == protocol["samples"] == 3240
    assert task_ids is not None
    metrics = [
        json.loads(line)
        for line in (ROOT / "task_metrics.jsonl").read_text().splitlines()
    ]
    metric_fields = [
        "model_key",
        "replica",
        "task_id",
        "passed",
        "ran",
        "external_lib",
        "external_lib_used",
        "lloc",
        "reasoning_tokens",
        "completion_tokens",
        "output_cap",
    ]
    assert metrics == [
        {field: row[field] for field in metric_fields} for row in all_rows
    ]
    models = []
    for name in protocol["models"]:
        rows = [r for r in all_rows if r["model_key"] == name]
        assert len(rows) == 1080
        groups = {t: [r for r in rows if r["task_id"] == t] for t in task_ids}
        assert all(len(v) == 5 for v in groups.values())
        pooled = read(f"raw/{name}/pooled.json")["analysis"]
        expected = next(m for m in reference["models"] if m["model"] == name)
        passed = sum(r["passed"] for r in rows)
        passed_at_5 = sum(any(r["passed"] for r in group) for group in groups.values())
        lengths = [r["lloc"] for r in rows if r["lloc"] is not None]
        model = {
            "model": name,
            "label": protocol["models"][name]["label"],
            "replicas": 5,
            "tasks": 216,
            "scored": 1080,
            "passed": passed,
            "pass_at_1": pooled["pass@1"],
            "tasks_passed_at_5": passed_at_5,
            "pass_at_5": pooled["pass@k"],
            "passes_per_replica": [r["passed"] for r in replicas if r["model"] == name],
            "imported": sum(r["external_lib"] for r in rows),
            "used": sum(r["external_lib_used"] for r in rows),
            "used_rate": sum(r["external_lib_used"] for r in rows) / 1080,
            "mean_thinking_tokens": statistics.mean(
                r["reasoning_tokens"] for r in rows
            ),
            "mean_output_tokens": statistics.mean(r["completion_tokens"] for r in rows),
            "total_output_tokens": sum(r["completion_tokens"] for r in rows),
            "mean_lloc": statistics.mean(lengths),
            "lloc_measured": len(lengths),
            "api_errors": 0,
            "output_cap_samples": sum(r["output_cap"] for r in rows),
            "empty_answers": sum(not r["submitted_answer"] for r in rows),
            # Native not_run counts installation-blocked responses. No-code/syntax
            # failures can also have ran=False but belong to ordinary failures.
            "not_run_samples": sum(bool(r["install_failed"]) for r in rows),
            "grading_completed_utc": expected["grading_completed_utc"],
        }
        close(pooled["pass@1"], passed / 1080)
        close(pooled["pass@k"], passed_at_5 / 216)
        for field, value in model.items():
            if isinstance(value, float):
                close(value, expected[field])
            else:
                assert value == expected[field], (name, field, value, expected[field])
        for field in ["imported", "used"]:
            count = sum(
                bool(r["external_lib" if field == "imported" else "external_lib_used"])
                for r in rows
            )
            close(
                pooled["library_use"][field]["per_response"]["using_lib_rate"],
                count / 1080,
            )
        models.append(model)
    for row, expected in zip(replicas, reference["replicas"], strict=True):
        for field, value in row.items():
            if isinstance(value, float):
                close(value, expected[field])
            else:
                assert value == expected[field], (field, value, expected[field])
    return models, replicas


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def report(models, replicas):
    lines = [
        "# Five-replica results",
        "",
        "All 3,240 samples are scored: 216 test tasks × five replicas × three models. Direct API calls, high reasoning, temperature 1, top-p 1, 32,768 output tokens per call.",
        "",
        "| Model | Replica passes (each /216) | Passes /1,080 | Pass@1 | Pass@5 | Mean thinking | Mean output | Imported | Static used | Mean LLOC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        lines.append(
            f"| {m['label']} | {', '.join(map(str, m['passes_per_replica']))} | {m['passed']}/1,080 | {m['pass_at_1']:.2%} | {m['pass_at_5']:.2%} | {m['mean_thinking_tokens']:,.0f} | {m['mean_output_tokens']:,.0f} | {m['imported']}/1,080 | {m['used']}/1,080 ({m['used_rate']:.2%}) | {m['mean_lloc']:.1f} |"
        )
    lines += [
        "",
        "Mean output includes thinking and answer tokens from the accepted response. Every accepted response has usage. Partial/retried requests are outside these means. LLOC uses measurable code only: 1,048 ML4, 1,069 GLM and 1,079 DeepSeek submissions.",
        "",
        "No exhausted API errors occurred. Capped/empty responses remain in the denominator, as does one GLM response not run because backports could not install. Static use means a possible imported-name reference, not execution coverage or absence of handwritten fallback code.",
        "",
        "## Individual replicas",
        "",
        "| Model | Replica | Passes /216 | Pass@1 | Imported | Used | Mean thinking | Mean output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {m["model"]: m["label"] for m in models}
    for r in replicas:
        lines.append(
            f"| {labels[r['model']]} | {r['replica']} | {r['passed']}/216 | {r['pass_at_1']:.2%} | {r['imported']} | {r['used']} | {r['mean_thinking_tokens']:,.0f} | {r['mean_output_tokens']:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def plots(models, replicas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fields = [
        ("mean_thinking_tokens", "Thinking tokens", 1),
        ("mean_output_tokens", "Total output tokens", 1),
        ("pass_at_1", "Pass@1 (%)", 100),
        ("used_rate", "Corrected static library use (%)", 100),
    ]
    for ax, (field, title, scale) in zip(axes.flat, fields):
        values = [m[field] * scale for m in models]
        ax.bar(range(3), values, width=0.6, color=colors)
        for index, model in enumerate(models):
            replica_values = []
            for j, replica in enumerate(
                r for r in replicas if r["model"] == model["model"]
            ):
                value = (
                    replica[field] * scale
                    if field != "used_rate"
                    else replica["used"] / 216 * 100
                )
                replica_values.append(value)
                ax.scatter(
                    index + (j - 2) * 0.065,
                    value,
                    color="#111111",
                    s=15,
                    alpha=0.65,
                    zorder=3,
                )
            ax.annotate(
                f"{values[index]:,.0f}" if scale == 1 else f"{values[index]:.2f}",
                (index, max([values[index], *replica_values])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_xticks(range(3), [m["label"] for m in models])
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(ax.get_ylim()[1] * 1.12, 1))
    fig.suptitle(
        "LibReuseBench · five non-agentic replicas · high reasoning · T=1 · top-p=1",
        fontsize=14,
    )
    fig.supxlabel(
        "Bars: pooled results. Dots: the five individual replicas (not confidence intervals).",
        fontsize=10,
    )
    directory = ROOT / "charts"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "comparison.png", dpi=170)
    fig.savefig(
        directory / "comparison.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Regenerate summaries and RESULTS.md"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Regenerate PNG/PDF charts; requires matplotlib",
    )
    parser.add_argument(
        "--verify-static",
        action="store_true",
        help="Recompute code selections/imports/usage with installed lure",
    )
    args = parser.parse_args()
    models, replicas = verify(args.verify_static)
    outputs = {
        "summary.json": json.dumps({"models": models, "replicas": replicas}, indent=2)
        + "\n",
        "final_models.csv": csv_text(models),
        "final_replicas.csv": csv_text(replicas),
        "RESULTS.md": report(models, replicas),
    }
    for name, text in outputs.items():
        if args.write:
            (ROOT / name).write_text(text)
        else:
            assert (ROOT / name).read_text() == text, f"Stale {name}; run with --write"
    if args.plots:
        plots(models, replicas)
    print(
        "Verified 3,240 full generations, 15 native-format file pairs, hashes and frozen scores."
    )


if __name__ == "__main__":
    main()
