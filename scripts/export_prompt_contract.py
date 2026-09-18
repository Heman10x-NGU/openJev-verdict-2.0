#!/usr/bin/env python3
"""Export canonical prompt and schema contract for browser WebGPU/WASM runtime.

Reads constants from core.formatting and core.primitives to ensure zero drift
between Python training/evaluation and browser inference.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.formatting import (
    INPUT_TEMPLATE,
    LABEL_MARKER,
    MAX_SUPPORTED_CANDIDATES,
    SEP_MARKER,
)
from core.primitives import (
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
)


def export_contract(output_path: str = "webgpu-demo/prompt_contract.json") -> dict:
    contract = {
        "label_marker": LABEL_MARKER,
        "sep_marker": SEP_MARKER,
        "input_template": INPUT_TEMPLATE,
        "default_question": "What is the primary customer inquiry or banking request?",
        "abstention_id": INSUFFICIENT_EVIDENCE_ID,
        "abstention_description": INSUFFICIENT_EVIDENCE_DESC,
        "max_candidates": MAX_SUPPORTED_CANDIDATES,
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    print(f"Exported prompt contract to {out_file}")
    return contract


if __name__ == "__main__":
    export_contract()
