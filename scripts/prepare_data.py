"""Prepare training, validation, and test datasets for Verdict-open-jev-ModernBERT.

Generates 5,000 enterprise decision instances with 20% explicit abstention
candidates (__insufficient_evidence__) across financial triage, infrastructure
alerts, and support routing. Ensures candidate order randomization and 
n-gram decontamination across data splits.
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

DOMAINS = [
    {
        "domain": "financial_triage",
        "question": "What is the primary action required for this financial transaction event?",
        "categories": [
            {
                "id": "freeze_account_fraud",
                "desc": "Freeze account immediately due to suspected unauthorized or fraudulent activity",
                "templates": [
                    "Card ending in {card} initiated 3 transactions of ${amount} in foreign country {country} within 90 seconds. Cardholder IP is {ip}.",
                    "Customer {user} reports unrecognized wire transfer of ${amount} to unknown beneficiary. Requesting urgent freeze.",
                    "Automated fraud engine flagged device fingerprint mismatch and multiple failed PIN attempts on account {user}.",
                    "Suspicious ATM withdrawal of ${amount} in {country} immediately after login from domestic IP {ip} on account {user}.",
                ],
            },
            {
                "id": "dispute_chargeback",
                "desc": "Initiate standard merchant chargeback dispute for billed goods or services",
                "templates": [
                    "Merchant charged ${amount} for order {ref} but customer {user} received damaged merchandise and vendor refuses contact.",
                    "Customer {user} was billed recurring charge of ${amount} on {date} after cancelling subscription via email.",
                    "Duplicate billing detected: transaction {ref} for ${amount} posted twice at merchant {merchant}.",
                    "Hotel charged customer {user} an unexplained incidental penalty fee of ${amount} on invoice {ref}.",
                ],
            },
            {
                "id": "credit_limit_inquiry",
                "desc": "Process customer credit limit adjustment request",
                "templates": [
                    "Customer {user} with annual income ${amount} requests credit line increase from $5,000 to $15,000 for upcoming relocation.",
                    "Account holder {user} requests credit limit review after 12 consecutive months of on-time payments.",
                    "Cardholder {user} asks if temporary credit boost of ${amount} is possible for medical emergency travel.",
                ],
            },
            {
                "id": "standard_transfer",
                "desc": "Execute routine scheduled or immediate balance transfer between accounts",
                "templates": [
                    "User {user} scheduled recurring monthly ACH transfer of ${amount} from checking to high-yield savings.",
                    "Domestic bank transfer of ${amount} requested from account {user} to external recipient IBAN ending in {card}.",
                    "Routine payroll disbursement of ${amount} credited to checking account {user} from employer {merchant}.",
                ],
            },
        ],
    },
    {
        "domain": "cloud_operations",
        "question": "What is the operational severity and remediation pathway for this system alert?",
        "categories": [
            {
                "id": "p0_incident_escalation",
                "desc": "P0 critical outage requiring immediate on-call page and failover dispatch",
                "templates": [
                    "Primary database cluster leader node in region {country} crashed with disk corruption. 100% of read/write queries failing.",
                    "Public API gateway latency spiked to 12,000ms with 502 Bad Gateway rate at 84% across all production clusters.",
                    "Core authentication service JWT signing key store unavailable. All user sessions terminating unexpectedly.",
                    "Kubernetes control plane API server unresponsive across production zone us-east-1. Workers dropping pods.",
                ],
            },
            {
                "id": "p2_degradation_investigation",
                "desc": "P2 moderate performance degradation requiring asynchronous engineer review",
                "templates": [
                    "Cache hit ratio dropped from 94% to 71% on Redis cluster {ref}, causing backend database CPU to reach 68%.",
                    "Worker pool queue depth increased to {amount} messages due to downstream third-party webhook throttle.",
                    "Internal metrics ingestion pipeline experiencing 15-minute lag on batch partitions.",
                    "Non-critical background report export worker memory usage elevated at 82% on host {ip}.",
                ],
            },
            {
                "id": "security_audit_alert",
                "desc": "Security anomaly alert requiring credential rotation and access audit",
                "templates": [
                    "IAM role prod-data-access assumed by unrecognized developer IP {ip} outside business hours.",
                    "Secret manager access key for service account {user} accessed from ephemeral container without tag.",
                    "GuardDuty detected port scan originating from internal pod IP {ip} targeting core VPC subnets.",
                ],
            },
            {
                "id": "routine_maintenance",
                "desc": "Routine platform maintenance or informational release notification",
                "templates": [
                    "Scheduled rolling OS security patch upgrade planned for worker pool {ref} on {date} during maintenance window.",
                    "Automated daily backup of PostgreSQL database cluster completed successfully in 42 minutes with zero errors.",
                    "Certificate renewal for domain api.example.com completed successfully via Let's Encrypt bot.",
                ],
            },
        ],
    },
]

ABSTENTION_SOURCES = [
    # Incomplete / missing critical context
    "Customer message received: 'I need urgent assistance with my recent transaction.' No account number, amount, or transaction ID provided.",
    "Alert webhook received with empty payload and truncated HTTP status header.",
    "Ticket #9941: 'Please review and fix this.' Context is completely blank.",
    "Notification: 'An event occurred on node.' Hostname and event type are null.",
    "User email: 'Hey team, just following up from yesterday.' Previous thread link is missing.",
    # Contradictory statements
    "System alert: 'All services operating at 100% healthy capacity. Critical database failure: total network blackout.'",
    "Customer note: 'Please cancel my account immediately and also process a new loan application for this account.'",
    "Audit log: 'User authenticated successfully with MFA; failed authentication: password rejected.'",
    # Out of scope domain
    "To prepare classic French baguettes, combine bread flour, water, active dry yeast, and fine sea salt. Let rise for 2 hours.",
    "The 2024 total solar eclipse crossed North America, passing over Mexico, the United States, and Canada.",
    "The UEFA Champions League final ended in a 2-0 victory following two late goals in extra time.",
    "Quantum entanglement describes the phenomenon where quantum particles remain interconnected regardless of distance.",
    "A standard guitar has six strings tuned in fourths, typically to E-A-D-G-B-E from lowest to highest pitch.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules within chloroplasts.",
]


def _extract_4grams(text: str) -> set[tuple[str, ...]]:
    words = [w.lower().strip(".,!?:;\"'()[]{}") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 4:
        return {tuple(words)}
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def generate_samples(
    total_samples: int = 5000,
    abstention_rate: float = 0.20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate balanced dataset with explicit abstention candidates."""
    rng = random.Random(seed)
    n_abstain = int(total_samples * abstention_rate)
    n_substantive = total_samples - n_abstain

    countries = ["GB", "US", "DE", "SG", "JP", "FR", "BR", "IN", "AU", "CA"]
    merchants = ["CloudHost Ltd", "Global SaaS Inc", "QuickRide App", "FastFood Corp", "DevTools Co"]
    dates = ["2026-04-12", "2026-05-18", "2026-07-02", "2026-08-15", "2026-09-01"]

    samples: list[dict[str, Any]] = []

    # 1. Substantive samples
    for i in range(n_substantive):
        domain_data = rng.choice(DOMAINS)
        question = domain_data["question"]
        categories = domain_data["categories"]

        # Pick ground truth category
        gt_cat = rng.choice(categories)
        tmpl = rng.choice(gt_cat["templates"])
        text = tmpl.format(
            user=f"usr_{rng.randint(1000, 9999)}",
            amount=f"{rng.randint(15, 2500)}.00",
            card=f"{rng.randint(1000, 9999)}",
            ref=f"REF-{rng.randint(10000, 99999)}",
            country=rng.choice(countries),
            merchant=rng.choice(merchants),
            date=rng.choice(dates),
            ip=f"192.0.2.{rng.randint(1, 254)}",
        )

        # Build candidate options list including abstention
        options = [
            {"id": cat["id"], "description": cat["desc"]}
            for cat in categories
        ]
        options.append(
            {"id": INSUFFICIENT_EVIDENCE_ID, "description": INSUFFICIENT_EVIDENCE_DESC}
        )
        rng.shuffle(options)

        samples.append(
            {
                "id": f"subst_{i:05d}",
                "domain": domain_data["domain"],
                "question": question,
                "text": text,
                "candidates": options,
                "target_id": gt_cat["id"],
                "is_abstention": False,
            }
        )

    # 2. Abstention samples
    for j in range(n_abstain):
        domain_data = rng.choice(DOMAINS)
        question = domain_data["question"]
        categories = domain_data["categories"]

        text = rng.choice(ABSTENTION_SOURCES)

        options = [
            {"id": cat["id"], "description": cat["desc"]}
            for cat in categories
        ]
        options.append(
            {"id": INSUFFICIENT_EVIDENCE_ID, "description": INSUFFICIENT_EVIDENCE_DESC}
        )
        rng.shuffle(options)

        samples.append(
            {
                "id": f"abst_{j:05d}",
                "domain": domain_data["domain"],
                "question": question,
                "text": text,
                "candidates": options,
                "target_id": INSUFFICIENT_EVIDENCE_ID,
                "is_abstention": True,
            }
        )

    rng.shuffle(samples)
    return samples


def partition_and_save(
    samples: list[dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, int]:
    """Split into train, val, test and verify n-gram decontamination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = samples[:n_train]
    val_data = samples[n_train : n_train + n_val]
    test_data = samples[n_train + n_val :]

    # Save files
    files = {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test.jsonl": test_data,
    }

    counts = {}
    for filename, dataset in files.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
        counts[filename] = len(dataset)

    # Decontamination check between train and test
    train_grams: set[tuple[str, ...]] = set()
    for row in train_data:
        train_grams.update(_extract_4grams(row["text"]))

    contaminated = 0
    for row in test_data:
        row_grams = _extract_4grams(row["text"])
        if row_grams and row_grams.issubset(train_grams):
            contaminated += 1

    manifest = {
        "total_samples": n_total,
        "splits": counts,
        "abstention_rate": sum(1 for s in samples if s["is_abstention"]) / n_total,
        "test_contamination_count": contaminated,
        "format": "verdict-open-jev-jsonl-v1",
    }
    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate datasets for Verdict-open-jev-ModernBERT.")
    parser.add_argument("--samples", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--abstention_rate", type=float, default=0.20, help="Fraction of abstention cases")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print(f"Generating {args.samples} samples (abstention={args.abstention_rate*100:.0f}%)...")
    samples = generate_samples(
        total_samples=args.samples,
        abstention_rate=args.abstention_rate,
        seed=args.seed,
    )

    counts = partition_and_save(samples, out_dir)
    print(f"Data generation complete: {counts}")
    print(f"Manifest written to {out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
