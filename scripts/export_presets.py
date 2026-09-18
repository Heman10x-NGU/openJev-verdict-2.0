#!/usr/bin/env python3
"""Generate webgpu-demo/presets.json directly from core.banking_glossary.

Ensures that every candidate description in browser demo presets matches the exact
glossary strings used during model training and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.banking_glossary import BANKING_GLOSSARY, get_enriched_label
from core.primitives import INSUFFICIENT_EVIDENCE_DESC, INSUFFICIENT_EVIDENCE_ID

PRESET_DEFINITIONS = [
    {
        "key": "card_arrival",
        "name": "Card Arrival (In-Scope)",
        "question": "What is the primary customer inquiry or banking request?",
        "text": "I ordered my new card two weeks ago but haven't received it in the mail yet. Can you check delivery?",
        "candidate_ids": [
            "card_arrival",
            "lost_or_stolen_card",
            "compromised_card",
            INSUFFICIENT_EVIDENCE_ID,
        ],
    },
    {
        "key": "lost_or_stolen_card",
        "name": "Lost/Stolen Card (High Urgency)",
        "question": "What is the primary customer inquiry or banking request?",
        "text": "Someone snatched my purse with my debit card inside while I was at the train station!",
        "candidate_ids": [
            "lost_or_stolen_card",
            "card_arrival",
            "pin_blocked",
            INSUFFICIENT_EVIDENCE_ID,
        ],
    },
    {
        "key": "compromised_card",
        "name": "Compromised Card (Fraud Flag)",
        "question": "What is the primary customer inquiry or banking request?",
        "text": "There are two charges for $240 from an electronics shop in another country that I never visited.",
        "candidate_ids": [
            "compromised_card",
            "lost_or_stolen_card",
            "exchange_rate",
            INSUFFICIENT_EVIDENCE_ID,
        ],
    },
    {
        "key": "oos_solar",
        "name": "Out-of-Scope (Solar Eclipse)",
        "question": "What is the primary customer inquiry or banking request?",
        "text": "The 2024 total solar eclipse crossed North America, passing over Mexico, the United States, and Canada.",
        "candidate_ids": [
            "card_arrival",
            "lost_or_stolen_card",
            "compromised_card",
            INSUFFICIENT_EVIDENCE_ID,
        ],
    },
    {
        "key": "custom",
        "name": "Custom Schema",
        "question": "What is the primary customer inquiry or banking request?",
        "text": "Enter your customer request text here...",
        "candidate_ids": [
            "action_a",
            "action_b",
            INSUFFICIENT_EVIDENCE_ID,
        ],
    },
]


def generate_presets(output_path: str = "webgpu-demo/presets.json") -> dict:
    presets = {}
    for p in PRESET_DEFINITIONS:
        candidates = []
        for cid in p["candidate_ids"]:
            if cid == INSUFFICIENT_EVIDENCE_ID:
                desc = INSUFFICIENT_EVIDENCE_DESC
            elif cid in BANKING_GLOSSARY:
                desc = BANKING_GLOSSARY[cid]
            elif cid in ("action_a", "action_b"):
                desc = f"{cid.replace('_', ' ').capitalize()} description"
            else:
                desc = get_enriched_label(cid)
            candidates.append({"id": cid, "desc": desc})

        presets[p["key"]] = {
            "key": p["key"],
            "name": p["name"],
            "question": p["question"],
            "text": p["text"],
            "candidates": candidates,
        }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)
    print(f"Generated presets with exact glossary descriptions in {out_file}")
    return presets


if __name__ == "__main__":
    generate_presets()
