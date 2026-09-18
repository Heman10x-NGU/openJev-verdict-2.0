import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kev.data import materialize
from kev.evaluate import ece, load, resolve_run
from kev.model import encode
from kev.suite import digest, load_split, record_digest, write_json

EPSILON = 1e-9


def api_request(record):
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()}}


def labels(q):
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return keys, keys.index(q["label"])
    if q["type"] == "noul":
        return ["false", "true"], int(q["label"])
    return [str(i) for i in range(len(q["criteria"]))], int(q["label"])


def validate_distribution(raw, keys):
    if set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    p = np.array([raw[k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    if total <= 0 or abs(total - 1) > max(1e-5, len(keys) * 0.005 + 1e-8):
        raise ValueError(f"invalid probability sum: {total}")
    return p / total, total


def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]
    rows = []
    for qid, q in record["questions"].items():
        keys, y = labels(q)
        p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid,
               "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": y,
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        rows.append(row)
    return rows


def metrics(rows, temperature=1.0):
    if not rows:
        raise ValueError("cannot score an empty population")
    nll, acc, conf, brier, mae, rps = [], [], [], [], [], []
    for row in rows:
        p = np.array(row["p"])
        if temperature != 1:
            z = np.log(np.maximum(p, EPSILON)) / temperature
            p = np.exp(z - z.max()); p /= p.sum()
        y = row["label"]
        target = np.eye(len(p))[y]
        nll.append(-math.log(max(float(p[y]), EPSILON)))
        acc.append(int(p.argmax() == y)); conf.append(float(p.max()))
        brier.append(float(((p - target) ** 2).sum()))
        if row["type"] == "score":
            mae.append(abs(float(p @ np.arange(len(p))) - y))
            rps.append(float(((p.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    result = {"n": len(rows), "nll": float(np.mean(nll)), "acc": float(np.mean(acc)),
              "ece": ece(conf, acc), "brier": float(np.mean(brier)), "mean_conf": float(np.mean(conf))}
    if mae:
        result.update(score_mae=float(np.mean(mae)), ranked_probability_score=float(np.mean(rps)))
    return result


def grouped_metrics(rows, key, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: metrics(group, temperature) for name, group in sorted(groups.items())}


def fit_temperature(rows):
    clean = [row for row in rows if row["variant"] == "clean"]
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4), 81))
    losses = [np.mean([m["nll"] for m in grouped_metrics(clean, "task", float(t)).values()]) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def paired_bootstrap(candidate, reference, samples=1000, seed=0, metric="nll"):
    def index(rows):
        return {(r["id"], r["question"]): r for r in rows if r["variant"] == "clean"}
    a, b = index(candidate), index(reference)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical complete clean examples")
    groups = defaultdict(list)
    for key, row in a.items():
        other = b[key]
        if row["keys"] != other["keys"] or row["label"] != other["label"]:
            raise ValueError("paired comparison labels or option order differ")
        groups[(row["source"], row["group"])].append((row["task"], metrics([row])[metric] - metrics([other])[metric]))
    sources = defaultdict(list)
    for (source, group), pairs in groups.items():
        sources[source].append(pairs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        tasks = defaultdict(list)
        for units in sources.values():
            for i in rng.integers(0, len(units), size=len(units)):
                for task, delta in units[i]:
                    tasks[task].append(delta)
        values.append(float(np.mean([np.mean(v) for v in tasks.values()])))
    observed = np.mean([m[metric] for m in grouped_metrics(list(a.values()), "task").values()]) - np.mean([m[metric] for m in grouped_metrics(list(b.values()), "task").values()])
    return {f"macro_{metric}_delta": float(observed), "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "samples": samples, "unit": "source-stratified original record; sibling questions stay together"}


def summarize(rows, temperature=1.0):
    clean = [r for r in rows if r["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    lookup = {(r["group"], r["question"]): r for r in clean}
    diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["group"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.array(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    return {"objective": -float(np.mean([v["nll"] for v in tasks.values()])),
            "clean": metrics(clean), "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in ("mnli", "sst5")], "task"),
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(clean, temperature),
            "metric_policy": {"nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}


class LocalPredictor:
    def __init__(self, run, device):
        self.run = resolve_run(run)
        self.tok, self.model = load(self.run, device)
        self.device = device

    def __call__(self, record):
        enc = encode(self.tok, materialize(record), strict=True)
        if len(enc["ids"]) > 2048:
            raise ValueError("packed request exceeds frozen 2048-token limit")
        if self.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        ps = self.model.probs(enc)
        if self.device == "mps":
            torch.mps.synchronize()
        return {"probabilities": {qid: dict(zip(labels(q)[0], p.tolist())) for (qid, q), p in zip(record["questions"].items(), ps)},
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"])}


def evaluate_records(records, predictor, directory, temperature=1.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    rows, latencies = [], []
    with (directory / "predictions.jsonl").open("w") as output:
        for record in records:
            try:
                pred = predictor(record)
                new_rows = prediction_rows(record, pred)
            except Exception as error:
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            output.write(json.dumps({"request_sha256": record_digest(api_request(record)), "id": record["_meta"]["id"],
                                     "prediction": pred, "rows": new_rows}, allow_nan=False) + "\n")
            output.flush()
            rows.extend(new_rows)
            latencies.append(pred["latency_ms"])
            coverage["evaluated_records"] += 1
            coverage["evaluated_questions"] += len(new_rows)
            if coverage["evaluated_records"] % 50 == 0:
                print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
    write_json(directory / "rows.json", rows)
    report = summarize(rows, temperature)
    report.update(coverage=coverage, latency_ms={"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, .95))})
    write_json(directory / "report.json", report)
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    split = "test" if a.allow_test else "development"
    records = load_split(a.suite, split, allow_test=a.allow_test)
    report, _ = evaluate_records(records, LocalPredictor(a.run, a.device), a.out)
    report.update(suite_sha256=digest(Path(a.suite) / "manifest.json"), run=a.run, split=split, calibration_applied=False)
    write_json(Path(a.out) / "report.json", report)
    print(json.dumps({"objective": report["objective"], "clean": report["clean"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
