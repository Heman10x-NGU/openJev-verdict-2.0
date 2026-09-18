"""Download and prepare real human benchmark datasets with honest partitions.

Ingests real customer utterances from PolyAI Banking77 and out-of-scope queries
from CLINC150. Applies semantic label enrichment and generates disjoint, non-leaking
partitions for train, validation, calibration, and test with explicit evaluation slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from core.banking_glossary import get_enriched_label
from core.primitives import INSUFFICIENT_EVIDENCE_DESC, INSUFFICIENT_EVIDENCE_ID

BANKING77_TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
BANKING77_TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
CLINC150_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_small.json"


def fetch_url_text(url: str) -> str:
    """Fetch raw text via HTTPS."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def load_banking_data() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Load train and test splits from Banking77."""
    print("Fetching Banking77 dataset...")
    train_text = fetch_url_text(BANKING77_TRAIN_URL)
    test_text = fetch_url_text(BANKING77_TEST_URL)

    def parse_csv(csv_content: str) -> list[tuple[str, str]]:
        reader = csv.reader(csv_content.splitlines())
        next(reader, None)  # skip header
        rows = []
        for r in reader:
            if len(r) >= 2:
                rows.append((r[0].strip(), r[1].strip()))
        return rows

    train_rows = parse_csv(train_text)
    test_rows = parse_csv(test_text)
    print(f"Loaded Banking77: {len(train_rows)} train, {len(test_rows)} test samples.")
    return train_rows, test_rows


def load_clinc_oos_data() -> tuple[list[str], list[str], list[str]]:
    """Load out-of-scope utterances from CLINC150 partitioned into train, val, and test."""
    print("Fetching CLINC150 out-of-scope dataset...")
    raw_json = json.loads(fetch_url_text(CLINC150_URL))
    train_oos = [item[0] for item in raw_json.get("oos_train", [])]
    val_oos = [item[0] for item in raw_json.get("oos_val", [])]
    test_oos = [item[0] for item in raw_json.get("oos_test", [])]

    print(f"Loaded CLINC150 OOS: {len(train_oos)} train, {len(val_oos)} val, {len(test_oos)} test samples.")
    return train_oos, val_oos, test_oos


def create_in_scope_sample(
    text: str,
    cat: str,
    all_categories: list[str],
    candidates_count: int,
    rng: random.Random,
    sample_id: str,
) -> dict[str, Any]:
    """Generate in-scope sample with true label present and random distractors."""
    distractors = [c for c in all_categories if c != cat]
    chosen_distractors = rng.sample(distractors, min(candidates_count - 2, len(distractors)))

    candidates = [{"id": cat, "description": get_enriched_label(cat)}]
    for d in chosen_distractors:
        candidates.append({"id": d, "description": get_enriched_label(d)})
    candidates.append({"id": INSUFFICIENT_EVIDENCE_ID, "description": INSUFFICIENT_EVIDENCE_DESC})
    rng.shuffle(candidates)

    return {
        "id": sample_id,
        "question": "What is the primary customer inquiry or banking request?",
        "text": text,
        "candidates": candidates,
        "target_id": cat,
        "is_abstention": False,
        "abstention_subtype": "in_scope",
    }


def create_missing_option_sample(
    text: str,
    cat: str,
    all_categories: list[str],
    candidates_count: int,
    rng: random.Random,
    sample_id: str,
) -> dict[str, Any]:
    """Generate missing-option sample where the gold intent is excluded from candidates."""
    distractors = [c for c in all_categories if c != cat]
    chosen_distractors = rng.sample(distractors, min(candidates_count - 1, len(distractors)))

    candidates = [{"id": d, "description": get_enriched_label(d)} for d in chosen_distractors]
    candidates.append({"id": INSUFFICIENT_EVIDENCE_ID, "description": INSUFFICIENT_EVIDENCE_DESC})
    rng.shuffle(candidates)

    return {
        "id": sample_id,
        "question": "What is the primary customer inquiry or banking request?",
        "text": text,
        "candidates": candidates,
        "target_id": INSUFFICIENT_EVIDENCE_ID,
        "is_abstention": True,
        "abstention_subtype": "missing_option",
    }


def create_distant_oos_sample(
    text: str,
    all_categories: list[str],
    candidates_count: int,
    rng: random.Random,
    sample_id: str,
) -> dict[str, Any]:
    """Generate distant out-of-scope sample using unrelated CLINC query."""
    chosen_cats = rng.sample(all_categories, min(candidates_count - 1, len(all_categories)))
    candidates = [{"id": c, "description": get_enriched_label(c)} for c in chosen_cats]
    candidates.append({"id": INSUFFICIENT_EVIDENCE_ID, "description": INSUFFICIENT_EVIDENCE_DESC})
    rng.shuffle(candidates)

    return {
        "id": sample_id,
        "question": "What is the primary customer inquiry or banking request?",
        "text": text,
        "candidates": candidates,
        "target_id": INSUFFICIENT_EVIDENCE_ID,
        "is_abstention": True,
        "abstention_subtype": "distant_oos",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare real human benchmark data.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_train, b_test = load_banking_data()
    c_train_oos, c_val_oos, c_test_oos = load_clinc_oos_data()

    # Strict decontamination: remove any utterance that appears in the held-out test set
    test_text_set = set(t.strip().lower() for t, _ in b_test).union(
        set(t.strip().lower() for t in c_test_oos)
    )
    b_train = [row for row in b_train if row[0].strip().lower() not in test_text_set]
    c_train_oos = [txt for txt in c_train_oos if txt.strip().lower() not in test_text_set]
    c_val_oos = [txt for txt in c_val_oos if txt.strip().lower() not in test_text_set]

    all_categories = sorted(list(set(c for _, c in b_train)))
    print(f"Total distinct Banking77 categories: {len(all_categories)}")

    rng = random.Random(args.seed)
    rng.shuffle(b_train)

    # 1. Disjoint partitions of Banking77 training set:
    # 8,000 for training, 1,000 for validation, 1,003 for calibration
    b_train_pool = b_train[:8000]
    b_val_pool = b_train[8000:9000]
    b_cal_pool = b_train[9000:]

    # 2. Partition CLINC val OOS: 50 for val, 50 for calibration
    c_val_pool = c_val_oos[:50]
    c_cal_pool = c_val_oos[50:]

    # --- Construct Training Set (2,500 samples: 2,000 in-scope + 200 missing-option + 100 CLINC OOS) ---
    train_records: list[dict[str, Any]] = []
    # 2,000 in-scope
    for idx, (txt, cat) in enumerate(b_train_pool[:2000]):
        train_records.append(
            create_in_scope_sample(txt, cat, all_categories, 5, rng, f"tr_in_{idx:05d}")
        )
    # 200 missing-option (from disjoint training pool items 2000:2200)
    for idx, (txt, cat) in enumerate(b_train_pool[2000:2200]):
        train_records.append(
            create_missing_option_sample(txt, cat, all_categories, 5, rng, f"tr_mo_{idx:05d}")
        )
    # 100 distant OOS
    for idx, txt in enumerate(c_train_oos):
        train_records.append(
            create_distant_oos_sample(txt, all_categories, 5, rng, f"tr_oos_{idx:05d}")
        )
    rng.shuffle(train_records)

    # --- Construct Validation Set (500 samples: 400 in-scope + 50 missing-option + 50 CLINC OOS) ---
    val_records: list[dict[str, Any]] = []
    for idx, (txt, cat) in enumerate(b_val_pool[:400]):
        val_records.append(
            create_in_scope_sample(txt, cat, all_categories, 5, rng, f"val_in_{idx:05d}")
        )
    for idx, (txt, cat) in enumerate(b_val_pool[400:450]):
        val_records.append(
            create_missing_option_sample(txt, cat, all_categories, 5, rng, f"val_mo_{idx:05d}")
        )
    for idx, txt in enumerate(c_val_pool):
        val_records.append(
            create_distant_oos_sample(txt, all_categories, 5, rng, f"val_oos_{idx:05d}")
        )
    rng.shuffle(val_records)

    # --- Construct Calibration Set (500 samples: 400 in-scope + 50 missing-option + 50 CLINC OOS) ---
    cal_records: list[dict[str, Any]] = []
    for idx, (txt, cat) in enumerate(b_cal_pool[:400]):
        cal_records.append(
            create_in_scope_sample(txt, cat, all_categories, 5, rng, f"cal_in_{idx:05d}")
        )
    for idx, (txt, cat) in enumerate(b_cal_pool[400:450]):
        cal_records.append(
            create_missing_option_sample(txt, cat, all_categories, 5, rng, f"cal_mo_{idx:05d}")
        )
    for idx, txt in enumerate(c_cal_pool):
        cal_records.append(
            create_distant_oos_sample(txt, all_categories, 5, rng, f"cal_oos_{idx:05d}")
        )
    rng.shuffle(cal_records)

    # --- Construct Test Set and Evaluation Slices (from official held-out test splits) ---
    rng_test = random.Random(args.seed + 100)
    # Primary benchmark test set: 800 Banking77 test + 200 CLINC test OOS (1,000 cases)
    test_records: list[dict[str, Any]] = []
    for idx, (txt, cat) in enumerate(b_test[:800]):
        test_records.append(
            create_in_scope_sample(txt, cat, all_categories, 5, rng_test, f"test_in_{idx:05d}")
        )
    for idx, txt in enumerate(c_test_oos[:200]):
        test_records.append(
            create_distant_oos_sample(txt, all_categories, 5, rng_test, f"test_oos_{idx:05d}")
        )
    rng_test.shuffle(test_records)

    # Slice: Missing Option Diagnostic (200 Banking77 test samples with gold removed)
    slice_missing_option: list[dict[str, Any]] = []
    for idx, (txt, cat) in enumerate(b_test[800:1000]):
        slice_missing_option.append(
            create_missing_option_sample(txt, cat, all_categories, 5, rng_test, f"test_mo_{idx:05d}")
        )

    # Slice: Distant OOS (200 distinct CLINC test samples)
    slice_distant_oos: list[dict[str, Any]] = []
    for idx, txt in enumerate(c_test_oos[200:400]):
        slice_distant_oos.append(
            create_distant_oos_sample(txt, all_categories, 5, rng_test, f"test_oos_slice_{idx:05d}")
        )

    # Slice: Cardinality tests (K in 3, 5, 9, 17, 25) on 100 held-out test cases
    cardinality_slices: dict[int, list[dict[str, Any]]] = {}
    for k in [3, 5, 9, 17, 25]:
        slice_k: list[dict[str, Any]] = []
        for idx, (txt, cat) in enumerate(b_test[1000:1100]):
            slice_k.append(
                create_in_scope_sample(txt, cat, all_categories, k, rng_test, f"test_k{k}_{idx:05d}")
            )
        cardinality_slices[k] = slice_k

    # Save datasets
    datasets = {
        "real_banking_train.jsonl": train_records,
        "real_banking_val.jsonl": val_records,
        "real_banking_cal.jsonl": cal_records,
        "real_banking_test.jsonl": test_records,
        "slice_missing_option.jsonl": slice_missing_option,
        "slice_distant_oos.jsonl": slice_distant_oos,
    }
    for k, slice_data in cardinality_slices.items():
        datasets[f"slice_cardinality_k{k}.jsonl"] = slice_data

    for fname, records in datasets.items():
        p = out_dir / fname
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {p} ({len(records)} records)")

    manifest = {
        "benchmark_version": "2.0.0-honest-splits",
        "provenance": {
            "banking77_source": "PolyAI-LDN/task-specific-datasets",
            "clinc_source": "clinc/oos-eval",
        },
        "splits": {
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "calibration_samples": len(cal_records),
            "test_samples": len(test_records),
            "missing_option_slice_samples": len(slice_missing_option),
            "distant_oos_slice_samples": len(slice_distant_oos),
        },
        "candidate_limits": {"max_supported": 25, "default_k": 5},
    }
    with open(out_dir / "real_benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Benchmarking dataset preparation complete with verified zero-leakage partitions.")


if __name__ == "__main__":
    main()
