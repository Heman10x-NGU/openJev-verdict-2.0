"""Synthetic validation data generator with explicit abstention and decontamination.

Generates structured decision pairs for ticket routing, urgency scoring,
and policy verification. Enforces exactly 20% explicit abstention cases
across realistic failure modes and runs lexical decontamination to prevent
template leakage between splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from core.primitives import (
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
)

TEMPLATES = [
    # Billing
    {
        "category": "billing",
        "templates": [
            "Customer {user} was charged ${amount} twice on credit card ending in {card}. Requesting immediate reversal.",
            "Invoice {inv} shows unexpected fee of ${amount}. User disputes the line item and asks for credit.",
            "Recurring subscription for account {user} renewed unexpectedly at ${amount}. Requesting cancellation and refund.",
        ],
        "urgency": ("p2", 2.0, "Moderate billing dispute"),
        "proposition": "User is disputing an unauthorized or duplicate payment.",
    },
    # Tech Support
    {
        "category": "tech_support",
        "templates": [
            "API endpoint /v1/checkout returning HTTP 500 internal server error for user {user}. Cluster degradation suspected.",
            "Database connection pool exhausted on worker node {user}. Latency spiked to 4500ms.",
            "Authentication token validation service throwing TLS handshake timeouts across EU region.",
        ],
        "urgency": ("p4", 4.0, "High severity service degradation"),
        "proposition": "The issue involves system outages or server-side technical failures.",
    },
    # Security / Access
    {
        "category": "security",
        "templates": [
            "User {user} detected suspicious login attempt from unrecognized IP 198.51.100.42. MFA reset requested.",
            "Service account API key leaked in public GitHub commit by developer {user}. Immediate revocation needed.",
            "Unusual privilege escalation alert triggered on IAM role admin-prod for user {user}.",
        ],
        "urgency": ("p5", 5.0, "Critical security threat"),
        "proposition": "A potential security breach or credential compromise has occurred.",
    },
    # General Inquiry
    {
        "category": "general",
        "templates": [
            "User {user} is asking for updated SOC2 compliance documentation and data retention policies.",
            "Inquiry regarding upcoming platform scheduled maintenance window for Q3 from organization {user}.",
            "Partner {user} requesting documentation on webhook payload schemas and retry backoff policies.",
        ],
        "urgency": ("p1", 1.0, "Low priority general inquiry"),
        "proposition": "The user is requesting informational documentation or policies.",
    },
]

ABSTENTION_TEMPLATES = [
    # Missing details / absent information
    "Hello, I need help with my account. Please contact me back as soon as possible.",
    "Issue reported. Need someone to take a look right now.",
    "Not working. Error occurred.",
    # Contradictory evidence
    "The system is running completely normally with zero errors, but also everything is down and crashed.",
    "Please cancel my subscription immediately, and also renew my subscription for 3 years without asking.",
    # Unrelated OOD context
    "The recipe calls for 200g of flour, two eggs, and a pinch of salt. Mix thoroughly.",
    "A total solar eclipse occurs when the Moon passes between the Earth and the Sun.",
    "The local football team won their home game 3-1 after a strong second half performance.",
]


def extract_ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    """Extract word n-grams from text for decontamination checking."""
    words = [w.lower().strip(".,!?:;\"'()[]{}") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def generate_dataset(
    n_samples: int = 500,
    abstention_rate: float = 0.20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate structured synthetic dataset with target abstention rate."""
    rng = random.Random(seed)
    n_abstain = int(n_samples * abstention_rate)
    n_substantive = n_samples - n_abstain

    records: list[dict[str, Any]] = []

    # Substantive samples
    for i in range(n_substantive):
        cat_idx = rng.randint(0, len(TEMPLATES) - 1)
        cat_data = TEMPLATES[cat_idx]
        tmpl = rng.choice(cat_data["templates"])
        text = tmpl.format(
            user=f"usr_{rng.randint(1000, 9999)}",
            amount=f"{rng.randint(10, 500)}.00",
            card=f"{rng.randint(1000, 9999)}",
            inv=f"INV-{rng.randint(10000, 99999)}",
        )

        urgency_id, urgency_val, urgency_desc = cat_data["urgency"]

        records.append(
            {
                "id": f"rec_{i:04d}",
                "context": text,
                "choice_target": cat_data["category"],
                "score_target": urgency_id,
                "score_value": urgency_val,
                "noul_target": "true",
                "is_abstention": False,
                "proposition": cat_data["proposition"],
            }
        )

    # Abstention samples
    for j in range(n_abstain):
        text = rng.choice(ABSTENTION_TEMPLATES)
        records.append(
            {
                "id": f"abstain_{j:04d}",
                "context": text,
                "choice_target": INSUFFICIENT_EVIDENCE_ID,
                "score_target": INSUFFICIENT_EVIDENCE_ID,
                "score_value": None,
                "noul_target": INSUFFICIENT_EVIDENCE_ID,
                "is_abstention": True,
                "proposition": "The user is requesting an immediate payment refund.",
            }
        )

    rng.shuffle(records)
    return records


def decontaminate(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    ngram_n: int = 5,
    max_overlap_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    """Filter evaluation records that share high lexical n-gram overlap with train."""
    train_ngrams: set[tuple[str, ...]] = set()
    for r in train_records:
        train_ngrams.update(extract_ngrams(r["context"], n=ngram_n))

    decontaminated: list[dict[str, Any]] = []
    filtered_count = 0

    for r in eval_records:
        eval_ngrams = extract_ngrams(r["context"], n=ngram_n)
        if not eval_ngrams:
            decontaminated.append(r)
            continue
        overlap = len(eval_ngrams & train_ngrams) / len(eval_ngrams)
        if overlap > max_overlap_ratio and not r["is_abstention"]:
            filtered_count += 1
        else:
            decontaminated.append(r)

    print(f"Decontamination: filtered {filtered_count} overlapping records.")
    return decontaminated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic validation data")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--n_train", type=int, default=600)
    parser.add_argument("--n_calib", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating train dataset...")
    train_data = generate_dataset(args.n_train, abstention_rate=0.20, seed=args.seed)

    print("Generating calibration dataset...")
    calib_data = generate_dataset(args.n_calib, abstention_rate=0.20, seed=args.seed + 1)
    calib_clean = decontaminate(train_data, calib_data)

    print("Generating test dataset...")
    test_data = generate_dataset(args.n_test, abstention_rate=0.20, seed=args.seed + 2)
    test_clean = decontaminate(train_data, test_data)

    for name, data in [
        ("synthetic_train.jsonl", train_data),
        ("synthetic_calibration.jsonl", calib_clean),
        ("synthetic_test.jsonl", test_clean),
    ]:
        dest = out_path / name
        with open(dest, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} records to {dest}")


if __name__ == "__main__":
    main()
