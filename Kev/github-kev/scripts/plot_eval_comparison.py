import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_LABELS = {
    "agnews": "AG News",
    "agnews_yn": "AG News · Yes/no",
    "banking77": "Banking77",
    "boolq": "BoolQ",
    "mnli": "MNLI",
    "sst5": "SST-5 · Score",
    "yelp": "Yelp · Score",
    "yelp_yn": "Yelp · Yes/no",
}
CAVEATS = [
    "Preliminary baseline snapshot on the frozen development suite; not a final model-selection claim.",
    "Locked final test has not been evaluated.",
    "Original kev checkpoint was fine-tuned on all six sources; Jev training exposure is unknown.",
    "This is neither an out-of-domain equivalence claim nor a fair architecture ablation.",
    "Accuracy is top-1 exact match, including Score questions; perturbations are excluded from plotted accuracy.",
    "Macro accuracy weights eight tasks equally; micro accuracy weights 720 clean questions equally.",
    "Only the saved paired macro-difference interval is shown; no per-task uncertainty is estimated.",
    "Jev is a hosted alias with an unexposed revision, evaluated through Vercel AI SDK 7.0.105.",
    "NLL is not plotted: Jev returns rounded zeros, making NLL clipping-floor sensitive (EPS 1e-9; saved alternatives 1e-6 and 1e-3).",
]


def load_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.isclose(actual, expected, rtol=0, atol=1e-12), message)


def validate_report(report, rows, name):
    index = {(row["id"], row["question"]): row for row in rows}
    require(len(index) == len(rows), f"{name}: duplicate question IDs")
    require(report["split"] == "development", f"{name}: expected development split")
    coverage = report["coverage"]
    require(len(rows) == coverage["evaluated_questions"] == coverage["requested_questions"],
            f"{name}: question coverage mismatch")
    require(len({row["id"] for row in rows}) == coverage["evaluated_records"] == coverage["requested_records"],
            f"{name}: record coverage mismatch")
    require(coverage["rejected_records"] == coverage["truncated_records"] == 0,
            f"{name}: incomplete coverage")
    clean = [row for row in rows if row["variant"] == "clean"]
    require({row["task"] for row in clean} == set(report["tasks"]),
            f"{name}: task sets differ")
    for task, metrics in [(None, report["clean"]), *report["tasks"].items()]:
        subset = clean if task is None else [row for row in clean if row["task"] == task]
        require(len(subset) == metrics["n"], f"{name}/{task}: question count mismatch")
        accuracy = np.mean([np.argmax(row["p"]) == row["label"] for row in subset])
        close(accuracy, metrics["acc"], f"{name}/{task}: accuracy mismatch")
    return index, len({row["id"] for row in clean})


def build_summary(paths):
    kev, jev, paired = (load_json(paths[key]) for key in ("kev_report", "jev_report", "comparison"))
    kev_rows, jev_rows = (load_json(paths[key]) for key in ("kev_rows", "jev_rows"))
    a, clean_records = validate_report(kev, kev_rows, "kev")
    b, jev_clean_records = validate_report(jev, jev_rows, "Jev")
    require(a.keys() == b.keys(), "Question IDs differ between models")
    for key, row in a.items():
        require(all(row[field] == b[key][field]
                    for field in ("group", "source", "task", "type", "variant", "keys", "label")),
                f"Pair metadata differs: {key}")
    require(clean_records == jev_clean_records, "Clean record counts differ")
    require(kev["suite_sha256"] == jev["suite_sha256"] == paired["suite_sha256"],
            "Suite hashes differ")
    require(set(kev["tasks"]) == set(jev["tasks"]) == set(TASK_LABELS),
            "This snapshot expects the eight decision-v1 tasks")
    tasks = []
    for task, label in TASK_LABELS.items():
        ka, ja = kev["tasks"][task], jev["tasks"][task]
        require(ka["n"] == ja["n"], f"{task}: model question counts differ")
        tasks.append({"task": task, "label": label, "questions": ka["n"],
                      "kev_accuracy": ka["acc"], "jev_accuracy": ja["acc"]})
    macro = {name: float(np.mean([row[f"{name}_accuracy"] for row in tasks]))
             for name in ("kev", "jev")}
    interval = paired["paired"]["acc"]
    close(macro["kev"] - macro["jev"], interval["macro_acc_delta"], "Saved macro delta differs")
    require(interval["ci95"][0] <= interval["macro_acc_delta"] <= interval["ci95"][1],
            "Invalid saved confidence interval")
    for name, report in (("candidate", kev), ("reference", jev)):
        require(report["clean"]["n"] == paired["clean"][name]["n"], "Comparison population differs")
        close(report["clean"]["acc"], paired["clean"][name]["acc"], "Comparison accuracy differs")
    return {
        "status": "preliminary baseline snapshot",
        "suite": "evals/decision-v1/development.jsonl",
        "suite_sha256": kev["suite_sha256"],
        "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for key, path in paths.items()},
        "models": {"kev": "Original released Qwen2.5-0.5B + LoRA checkpoint", "jev": jev["provider"]},
        "coverage_per_model": kev["coverage"],
        "clean_records": clean_records,
        "clean_questions": kev["clean"]["n"],
        "tasks": tasks,
        "macro_accuracy": macro,
        "macro_accuracy_difference": {
            "direction": "kev minus Jev",
            "fraction": interval["macro_acc_delta"],
            "percentage_points": 100 * interval["macro_acc_delta"],
            "ci95_fraction": interval["ci95"],
            "ci95_percentage_points": [100 * value for value in interval["ci95"]],
            "bootstrap_samples": interval["samples"],
            "bootstrap_unit": interval["unit"],
            "interval_source": str(paths["comparison"]),
        },
        "micro_accuracy": {"kev": kev["clean"]["acc"], "jev": jev["clean"]["acc"]},
        "micro_accuracy_difference_percentage_points": 100 * (kev["clean"]["acc"] - jev["clean"]["acc"]),
        "nll_floor_sensitivity_not_plotted": paired["nll_floor_sensitivity"],
        "caveats": CAVEATS,
    }


def plot(summary, output):
    ink, muted, rule = "#202B30", "#5D6970", "#DCE1E4"
    colors = {"kev": "#355C6B", "jev": "#AFBBC1"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": ink,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15, 10.8))
    fig.text(.045, .955, "PRELIMINARY  /  BASELINE SNAPSHOT", fontsize=10, weight="bold", color=muted)
    fig.text(.045, .911, "kev vs. Jev", fontsize=29, weight="bold")
    fig.text(.045, .878, "Original kev: Qwen2.5-0.5B + LoRA     |     Jev: typesafe-ai/jev · Vercel AI SDK 7.0.105",
             fontsize=11.5, color=muted)
    coverage = summary["coverage_per_model"]
    fig.text(.045, .848,
             f"Same frozen development suite · {coverage['evaluated_records']:,} requests / "
             f"{coverage['evaluated_questions']:,} questions including perturbations",
             fontsize=11, color=muted)
    fig.add_artist(Line2D([.045, .96], [.825, .825], transform=fig.transFigure, color=rule, lw=1))
    fig.text(.045, .792, "Per-task accuracy", fontsize=15, weight="bold")
    fig.text(.045, .767,
             f"Clean subset only · {summary['clean_records']:,} records / {summary['clean_questions']:,} questions · top-1 exact match",
             fontsize=10, color=muted)

    ax = fig.add_axes([.185, .232, .49, .505])
    tasks = summary["tasks"]
    y = np.arange(len(tasks))
    for name, offset in (("kev", -.17), ("jev", .17)):
        values = [100 * task[f"{name}_accuracy"] for task in tasks]
        bars = ax.barh(y + offset, values, height=.28, color=colors[name], label="kev" if name == "kev" else "Jev", zorder=3)
        for bar, value in zip(bars, values):
            ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%",
                    va="center", fontsize=9.5, color=ink)
    ax.set_yticks(y, [f"{task['label']}\nn = {task['questions']:,} questions" for task in tasks], fontsize=10)
    ax.set_ylim(len(tasks) - .5, -.5)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Accuracy · higher is better", labelpad=10, fontsize=10)
    ax.grid(axis="x", color=rule, linewidth=.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 1.03), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.5, fontsize=11)

    fig.add_artist(Line2D([.71, .71], [.226, .80], transform=fig.transFigure, color=rule, lw=1))
    left = .744
    fig.text(left, .792, "Macro-task difference", fontsize=15, weight="bold")
    fig.text(left, .766, "kev minus Jev · equal task weight", fontsize=10, color=muted)
    delta = summary["macro_accuracy_difference"]
    value = delta["percentage_points"]
    lo, hi = delta["ci95_percentage_points"]
    fig.text(left, .717, f"{value:+.2f} pp", fontsize=29, weight="bold", color=colors["kev"])
    fig.text(left, .684, f"95% CI  [{lo:+.2f}, {hi:+.2f}] pp", fontsize=12)
    ci_ax = fig.add_axes([left, .612, .21, .050])
    bound = max(8, np.ceil(max(abs(lo), abs(hi)) / 2) * 2)
    ci_ax.axvline(0, color=muted, linestyle=(0, (3, 3)), lw=1)
    ci_ax.errorbar(value, 0, xerr=[[value - lo], [hi - value]], fmt="o", color=colors["kev"],
                   markersize=6, capsize=5, linewidth=2)
    ci_ax.set_xlim(-bound, bound)
    ci_ax.set_ylim(-1, 1)
    ci_ax.set_yticks([])
    ci_ax.set_xticks([-bound, 0, bound], [f"{int(-bound)}", "0", f"+{int(bound)}"])
    ci_ax.tick_params(axis="x", length=0, labelsize=9)
    for spine in ci_ax.spines.values():
        spine.set_visible(False)
    fig.text(left, .579, "← Jev higher          kev higher →", fontsize=9, color=muted)
    fig.text(left, .543,
             f"Saved paired bootstrap · {delta['bootstrap_samples']:,} draws\n"
             "Source-stratified original records;\nsibling questions stay together.",
             fontsize=9.5, color=muted, linespacing=1.5, va="top")
    interval_note = "Interval includes zero." if lo <= 0 <= hi else "Interval excludes zero."
    fig.text(left, .466, interval_note, fontsize=11, weight="bold")
    fig.text(left, .440, "Not evidence of equivalence.", fontsize=10, color=muted)
    fig.add_artist(Line2D([left, .96], [.419, .419], transform=fig.transFigure, color=rule, lw=1))
    fig.text(left, .388, "Aggregate accuracy", fontsize=13, weight="bold")
    fig.text(.865, .358, "kev", color=colors["kev"], weight="bold", ha="right", fontsize=10)
    fig.text(.95, .358, "Jev", color=muted, weight="bold", ha="right", fontsize=10)
    for ypos, label, metric in ((.326, "Macro · 8 tasks", "macro_accuracy"),
                               (.292, f"Micro · {summary['clean_questions']} Qs", "micro_accuracy")):
        fig.text(left, ypos, label, fontsize=9)
        fig.text(.865, ypos, f"{100 * summary[metric]['kev']:.2f}%", ha="right", fontsize=10)
        fig.text(.95, ypos, f"{100 * summary[metric]['jev']:.2f}%", ha="right", fontsize=10)
    fig.text(left, .253, "Micro weights questions, not tasks.\nBoth aggregates use clean questions only.",
             fontsize=9, color=muted, linespacing=1.5, va="top")

    fig.add_artist(Line2D([.045, .96], [.166, .166], transform=fig.transFigure, color=rule, lw=1))
    footnotes = [
        "Scope: original kev was fine-tuned on all six sources; Jev training exposure is unknown. Not an out-of-domain equivalence claim or fair architecture ablation.",
        "Status: the locked final test has not been evaluated. More runs are pending; this development snapshot is not a final model-selection claim.",
        "Probability caveat: NLL is omitted. Jev returns rounded zeros; NLL is clipping-floor sensitive (EPS 1e-9; saved alternatives 1e-6 and 1e-3).",
        f"Source: {summary['suite']} · saved paired comparison: {Path(delta['interval_source']).name} · no per-task error bars are estimated.",
    ]
    for ypos, text in zip((.138, .111, .084, .057), footnotes):
        fig.text(.045, ypos, text, fontsize=9, color=muted)
    fig.savefig(output, dpi=200, metadata={"Title": "kev vs. Jev — preliminary development baseline",
                                          "Description": "Clean per-task accuracy and saved paired macro-accuracy confidence interval."})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a preliminary kev/Jev snapshot using saved development artifacts only.")
    parser.add_argument("--kev-report", type=Path, default=ROOT / "runs/research-kev-v01/report.json")
    parser.add_argument("--jev-report", type=Path, default=ROOT / "runs/research-jev-v1/report.json")
    parser.add_argument("--kev-rows", type=Path, default=ROOT / "runs/research-kev-v01/rows.json")
    parser.add_argument("--jev-rows", type=Path, default=ROOT / "runs/research-jev-v1/rows.json")
    parser.add_argument("--comparison", type=Path, default=ROOT / "runs/kev-vs-jev-v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/kev-vs-jev.png")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/kev-vs-jev-summary.json")
    args = parser.parse_args()
    paths = {key: getattr(args, key).resolve() for key in ("kev_report", "jev_report", "kev_rows", "jev_rows", "comparison")}
    outputs = (args.out.resolve(), args.summary.resolve())
    require(len(set(outputs)) == 2, "Image and summary paths must differ")
    for output in outputs:
        require(output not in paths.values(), "Output must not overwrite an input artifact")
        require(output.parent.is_dir(), f"Output directory does not exist: {output.parent}")
    summary = build_summary(paths)
    plot(summary, args.out)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"image": str(args.out), "summary": str(args.summary),
                      "macro_accuracy": summary["macro_accuracy"],
                      "micro_accuracy": summary["micro_accuracy"],
                      "macro_accuracy_difference": summary["macro_accuracy_difference"]}, indent=2))


if __name__ == "__main__":
    main()
