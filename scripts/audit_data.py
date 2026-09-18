"""Audit dataset partitions to guarantee zero leakage and correct schema invariants."""

import json
from pathlib import Path


def audit_splits(data_dir: str = "data") -> None:
    data_path = Path(data_dir)
    splits = ["real_banking_train.jsonl", "real_banking_val.jsonl", "real_banking_cal.jsonl", "real_banking_test.jsonl"]

    split_texts: dict[str, set[str]] = {}
    total_samples = 0

    for split in splits:
        p = data_path / split
        assert p.exists(), f"Split {p} does not exist"
        texts = set()
        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                assert "id" in rec
                assert "candidates" in rec
                assert len(rec["candidates"]) <= 25, f"Record {rec['id']} has > 25 candidates"
                assert "target_id" in rec
                assert any(c["id"] == rec["target_id"] for c in rec["candidates"]), f"Target {rec['target_id']} not in candidates"
                texts.add(rec["text"].strip().lower())
                count += 1
        split_texts[split] = texts
        total_samples += count
        print(f"[{split}] {count} records, {len(texts)} unique text strings.")

    # Cross-split leakage checks
    split_names = list(split_texts.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_texts[s1].intersection(split_texts[s2])
            assert len(overlap) == 0, f"DATA LEAKAGE DETECTED between {s1} and {s2}: {len(overlap)} overlapping utterances!"

    print("Data leakage audit PASSED! 0 overlapping utterances between train, val, cal, and test splits.")


if __name__ == "__main__":
    audit_splits()
