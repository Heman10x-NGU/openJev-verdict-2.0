#!/usr/bin/env python3
"""Compute empirical reference floors on LocalLLaMA/typed-decisions.

Baselines evaluated:
1. Uniform Random: Assigns equal probability (1/K) to all options.
2. Majority Class: Predicts the most frequent label per (workflow, question) in train.
3. TF-IDF + Logistic Regression: Trains an independent classifier per (workflow, question).
4. Gold Distribution Oracle: Uses the true teacher-panel probabilities as predictions.

Metrics match scripts/evaluate_verdict_baseline.py:
- Top-1 Exact Match Accuracy
- Brier Score (on choice and noul questions)
- Expected Calibration Error (ECE, 10 equal-width bins)
- Score MAE (on score questions)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error across confidence bins."""
    if len(confidences) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        mask = (
            (confidences > bin_lower) & (confidences <= bin_upper)
            if i > 0
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
        bin_samples = int(np.sum(mask))

        if bin_samples > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (bin_samples / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def extract_data(split_data: Any) -> List[Dict[str, Any]]:
    """Flatten dataset split into structured records."""
    records = []
    for row in split_data:
        wf = row.get("workflow", "general")
        state = row["state"]
        q_defs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold_defs = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]

        for qid, q in q_defs.items():
            t = q["type"]
            crit = q.get("criteria", {})
            gold = gold_defs.get(qid, {})

            # Candidate labels list
            if t == "choice":
                cand_ids = list(crit.keys())
            elif t == "noul":
                cand_ids = ["false", "true"]
            elif t == "score":
                cand_ids = [str(i) for i in range(len(crit))]
            else:
                cand_ids = []

            gold_score = 0.0
            if t == "score":
                try:
                    gold_score = float(gold.get("score", float(gold.get("label", 0))))
                except (ValueError, TypeError):
                    gold_score = 0.0

            records.append({
                "workflow": wf,
                "qid": qid,
                "type": t,
                "key": f"{wf}::{qid}",
                "state": state,
                "cand_ids": cand_ids,
                "gold_label": str(gold.get("label", "")).lower(),
                "gold_probs": {str(k).lower(): float(v) for k, v in gold.get("probabilities", {}).items()},
                "gold_score": gold_score,
            })
    return records


def run_uniform_random(test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs, confs, briers, maes = [], [], [], []
    for r in test_records:
        k = len(r["cand_ids"])
        p = 1.0 / max(1, k)
        pred = np.random.choice(r["cand_ids"])
        is_corr = 1.0 if pred == r["gold_label"] else 0.0
        accs.append(is_corr)
        confs.append(p)

        if r["type"] in ("choice", "noul"):
            # Uniform vector vs gold vector
            g_vec = np.array([r["gold_probs"].get(c, 0.0) for c in r["cand_ids"]])
            p_vec = np.ones(k) / k
            brier = float(np.sum((p_vec - g_vec) ** 2))
            briers.append(brier)
        elif r["type"] == "score":
            exp_score = sum(float(c) * (1.0 / k) for c in r["cand_ids"])
            maes.append(abs(exp_score - r["gold_score"]))

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "brier": round(float(np.mean(briers)), 4),
        "ece": round(compute_ece(np.array(confs), np.array(accs)), 4),
        "score_mae": round(float(np.mean(maes)), 4),
    }


def run_majority_class(train_records: List[Dict[str, Any]], test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Find majority label per key
    by_key = defaultdict(list)
    for r in train_records:
        by_key[r["key"]].append(r["gold_label"])

    majority_map = {}
    majority_prob = {}
    for k, labels in by_key.items():
        counts = Counter(labels)
        top_lbl, top_cnt = counts.most_common(1)[0]
        majority_map[k] = top_lbl
        majority_prob[k] = top_cnt / len(labels)

    accs, confs = [], []
    for r in test_records:
        pred = majority_map.get(r["key"], r["cand_ids"][0])
        conf = majority_prob.get(r["key"], 0.5)
        is_corr = 1.0 if pred == r["gold_label"] else 0.0
        accs.append(is_corr)
        confs.append(conf)

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "brier": None,
        "ece": round(compute_ece(np.array(confs), np.array(accs)), 4),
    }


def run_gold_oracle(test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs, confs, briers, maes = [], [], [], []
    for r in test_records:
        g_probs = r["gold_probs"]
        cand_ids = r["cand_ids"]

        # Argmax of gold distribution
        top_k = max(g_probs.keys(), key=lambda k: g_probs[k])
        conf = g_probs[top_k]
        is_corr = 1.0 if top_k == r["gold_label"] else 0.0

        accs.append(is_corr)
        confs.append(conf)

        if r["type"] in ("choice", "noul"):
            briers.append(0.0)  # perfect match with itself
        elif r["type"] == "score":
            maes.append(0.0)

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "brier": 0.0000,
        "ece": round(compute_ece(np.array(confs), np.array(accs)), 4),
        "score_mae": 0.0000,
    }


def run_tfidf_baseline(train_records: List[Dict[str, Any]], test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train independent TF-IDF + LogisticRegression for each of the 20 (wf, qid) schemas."""
    # Group train and test by key
    train_by_key = defaultdict(list)
    for r in train_records:
        train_by_key[r["key"]].append(r)

    test_by_key = defaultdict(list)
    for r in test_records:
        test_by_key[r["key"]].append(r)

    all_accs, all_confs, all_briers, all_maes = [], [], [], []

    for key, t_recs in test_by_key.items():
        tr_recs = train_by_key[key]
        qtype = t_recs[0]["type"]
        cand_ids = t_recs[0]["cand_ids"]

        train_texts = [r["state"] for r in tr_recs]
        train_labels = [r["gold_label"] for r in tr_recs]

        test_texts = [r["state"] for r in t_recs]
        test_golds = [r["gold_label"] for r in t_recs]

        # TF-IDF feature extraction (character n-grams 3-6 captures exact domain morphology)
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=2,
            sublinear_tf=True,
        )
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        # Logistic Regression with C=5.0
        clf = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, train_labels)

        # Probabilities
        classes = list(clf.classes_)
        probs = clf.predict_proba(X_test)  # (N, num_classes)

        for i, r in enumerate(t_recs):
            p_row = probs[i]
            top_idx = int(np.argmax(p_row))
            pred = classes[top_idx]
            conf = float(p_row[top_idx])
            gold_lbl = test_golds[i]
            is_corr = 1.0 if pred == gold_lbl else 0.0

            all_accs.append(is_corr)
            all_confs.append(conf)

            p_dict = {c: float(p_row[j]) for j, c in enumerate(classes)}

            if qtype in ("choice", "noul"):
                # Brier score against soft gold distribution
                p_vec = np.array([p_dict.get(c, 0.0) for c in cand_ids])
                g_vec = np.array([r["gold_probs"].get(c, 0.0) for c in cand_ids])
                p_vec = p_vec / max(1e-9, p_vec.sum())
                brier = float(np.sum((p_vec - g_vec) ** 2))
                all_briers.append(brier)
            elif qtype == "score":
                exp_score = sum(float(c) * p_dict.get(c, 0.0) for c in cand_ids)
                all_maes.append(abs(exp_score - r["gold_score"]))

    return {
        "accuracy": round(float(np.mean(all_accs)), 4),
        "brier": round(float(np.mean(all_briers)), 4),
        "ece": round(compute_ece(np.array(all_confs), np.array(all_accs)), 4),
        "score_mae": round(float(np.mean(all_maes)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reference floors on LocalLLaMA/typed-decisions.")
    parser.add_argument("--out", default="reports/reference_floors.json")
    args = parser.parse_args()

    print("Loading LocalLLaMA/typed-decisions train & test splits...")
    ds_train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")

    print(f"Loaded {len(ds_train)} train cases, {len(ds_test)} test cases.")

    train_recs = extract_data(ds_train)
    test_recs = extract_data(ds_test)
    print(f"Extracted {len(train_recs)} train decisions, {len(test_recs)} test decisions.")

    print("\n1. Running Uniform Random Baseline...")
    np.random.seed(42)
    res_uniform = run_uniform_random(test_recs)
    print(f"   Accuracy : {res_uniform['accuracy']*100:.2f}%")
    print(f"   Brier    : {res_uniform['brier']:.4f}")
    print(f"   ECE      : {res_uniform['ece']:.4f}")

    print("\n2. Running Majority Class Baseline...")
    res_majority = run_majority_class(train_recs, test_recs)
    print(f"   Accuracy : {res_majority['accuracy']*100:.2f}%")
    print(f"   ECE      : {res_majority['ece']:.4f}")

    print("\n3. Running TF-IDF + Logistic Regression Baseline...")
    res_tfidf = run_tfidf_baseline(train_recs, test_recs)
    print(f"   Accuracy : {res_tfidf['accuracy']*100:.2f}%")
    print(f"   Brier    : {res_tfidf['brier']:.4f}")
    print(f"   ECE      : {res_tfidf['ece']:.4f}")
    print(f"   Score MAE: {res_tfidf['score_mae']:.4f}")

    print("\n4. Running Gold Distribution Oracle...")
    res_oracle = run_gold_oracle(test_recs)
    print(f"   Accuracy : {res_oracle['accuracy']*100:.2f}%")
    print(f"   Brier    : {res_oracle['brier']:.4f}")
    print(f"   ECE      : {res_oracle['ece']:.4f}")

    # Load Verdict 1.0 baseline metrics for direct comparison
    verdict_1_metrics = {
        "accuracy": 0.2610,
        "brier": 0.5851,
        "ece": 0.4209,
        "score_mae": 1.4417,
        "latency_p50_ms": 313.88,
    }

    out_data = {
        "benchmark": "LocalLLaMA/typed-decisions",
        "total_test_cases": len(ds_test),
        "total_test_decisions": len(test_recs),
        "reference_floors": {
            "uniform_random": res_uniform,
            "majority_label": res_majority,
            "tfidf_logistic_regression": res_tfidf,
            "verdict_1_0_baseline": verdict_1_metrics,
            "gold_distribution_oracle": res_oracle,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved reference floors to: {out_path}")


if __name__ == "__main__":
    main()
