"""Plot training loss from train logs and held-out accuracy vs baselines from eval.json.

Run: uv run python -m kev.plot --logs runs/train.log:kev-0.5b runs/train_holdout.log:holdout --eval runs/kev/eval.json --out docs/training.png
"""
import argparse, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+)")
ORDER = ["agnews", "agnews_yn", "banking77", "boolq", "mnli", "sst5", "yelp", "yelp_yn"]
LABEL = {"agnews": "AG News\nK=4", "agnews_yn": "AG News\nyes/no", "banking77": "Banking77\nK=77", "boolq": "BoolQ\nyes/no",
         "mnli": "MNLI\nK=3", "sst5": "SST-5\nscore", "yelp": "Yelp\nscore", "yelp_yn": "Yelp\nyes/no"}


def read_log(path):
    steps, losses, epoch_ends, prev_ep = [], [], [], 0
    for m in STEP_RE.finditer(open(path).read()):
        ep, step, loss = int(m[1]), int(m[2]), float(m[4])
        if ep != prev_ep: epoch_ends.append(steps[-1]); prev_ep = ep
        steps.append(step); losses.append(loss)
    return steps, losses, epoch_ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["runs/train.log:kev-0.5b"], help="path:label pairs")
    ap.add_argument("--eval", default="runs/kev/eval.json")
    ap.add_argument("--out", default="docs/training.png")
    a = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=160, gridspec_kw={"width_ratios": [1.1, 1.3]})

    for spec in a.logs:
        path, label = spec.split(":") if ":" in spec else (spec, spec)
        steps, losses, ends = read_log(path)
        ax1.plot(steps, losses, lw=1.6, label=f"{label}  (final {losses[-1]:.2f})")
        for e in ends: ax1.axvline(e, color="0.8", lw=0.8, ls="--")
    ax1.set_yscale("log"); ax1.set_xlabel("optimizer step (8 records each)"); ax1.set_ylabel("train loss (cross-entropy, log scale)")
    ax1.set_title("training loss"); ax1.grid(alpha=0.25); ax1.legend(frameon=False)

    e = json.load(open(a.eval))
    kev, b0, b1 = e["accuracy_calibration"], e.get("baseline_zero_shot_base", {}), e.get("baseline_zero_shot_instruct", {})
    xs = range(len(ORDER)); w = 0.27
    ax2.bar([x - w for x in xs], [b0.get(s, {}).get("acc", 0) for s in ORDER], w, color="0.78", label="zero-shot base (letter logits)")
    ax2.bar([x for x in xs], [b1.get(s, {}).get("acc", 0) for s in ORDER], w, color="0.55", label="zero-shot Instruct (letter logits)")
    ax2.bar([x + w for x in xs], [kev[s]["acc"] for s in ORDER], w, color="#1f77b4", label="kev-0.5b (pointer readout)")
    for x, s in zip(xs, ORDER):
        ax2.text(x + w, kev[s]["acc"] + 0.012, f"ECE\n{kev[s]['ece']:.2f}", ha="center", va="bottom", fontsize=6.5, color="#1f77b4")
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([LABEL[s] for s in ORDER], fontsize=7.5)
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("held-out accuracy"); ax2.set_title(f"held-out accuracy vs zero-shot baselines  (all: {kev['ALL']['acc']:.3f}, ECE {kev['ALL']['ece']:.3f})")
    ax2.grid(axis="y", alpha=0.25); ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for ax in (ax1, ax2): ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); print("wrote", a.out)


if __name__ == "__main__":
    main()
