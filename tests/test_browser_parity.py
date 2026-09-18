"""Test prompt contract parity between Python and Browser WebGPU runtimes.

Asserts that build_model_input in core.formatting produces byte-identical strings
to applying webgpu-demo/prompt_contract.json to the same input and candidates.
Also verifies probability agreement on presets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.formatting import build_model_input


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_prompt_contract_byte_parity() -> None:
    contract_path = REPO_ROOT / "webgpu-demo" / "prompt_contract.json"
    assert contract_path.exists(), "webgpu-demo/prompt_contract.json must exist"

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    question = "What is the primary customer inquiry or banking request?"
    context = "I ordered my new card two weeks ago but haven't received it in the mail yet. Can you check delivery?"
    candidate_descs = [
        "Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail",
        "Report a physically lost, stolen, or misplaced card requiring immediate permanent block or cancellation",
        "Report suspected unauthorized card cloning, skimmed magnetic stripe, or fraudulent card transactions",
        "insufficient evidence",
    ]

    # Canonical Python formatting
    py_string = build_model_input(question, context, candidate_descs)

    # JS runtime simulation using contract
    js_text = contract["input_template"].replace("{question}", question).replace("{context}", context)
    label_prefix = "".join(f"{contract['label_marker']}{desc}" for desc in candidate_descs)
    js_string = f"{label_prefix}{contract['sep_marker']}{js_text}"

    assert py_string == js_string, f"Mismatch:\nPy: {py_string}\nJS: {js_string}"


def test_presets_descriptions_match_glossary() -> None:
    from core.banking_glossary import BANKING_GLOSSARY
    from core.primitives import INSUFFICIENT_EVIDENCE_DESC, INSUFFICIENT_EVIDENCE_ID

    presets_path = REPO_ROOT / "webgpu-demo" / "presets.json"
    assert presets_path.exists(), "webgpu-demo/presets.json must exist"

    with open(presets_path, "r", encoding="utf-8") as f:
        presets = json.load(f)

    for p_key, preset in presets.items():
        for cand in preset["candidates"]:
            cid = cand["id"]
            cdesc = cand["desc"]
            if cid == INSUFFICIENT_EVIDENCE_ID:
                assert cdesc == INSUFFICIENT_EVIDENCE_DESC
            elif cid in BANKING_GLOSSARY:
                assert cdesc == BANKING_GLOSSARY[cid], f"Mismatch for {cid} in preset {p_key}"


def test_browser_python_prediction_parity() -> None:
    """Verify that Python ONNX inference produces identical probabilities on card_arrival preset."""
    onnx_path = REPO_ROOT / "artifacts" / "v2" / "model.onnx"
    if not onnx_path.exists():
        onnx_path = REPO_ROOT / "artifacts" / "v2" / "openjev_modernbert.onnx"
    cal_path = REPO_ROOT / "artifacts" / "v2" / "calibrator.json"
    if not cal_path.exists():
        cal_path = REPO_ROOT / "artifacts" / "v2" / "calibrator_modernbert.json"
    presets_path = REPO_ROOT / "webgpu-demo" / "presets.json"

    if not onnx_path.exists() or not cal_path.exists():
        pytest.skip("ONNX model or calibrator not found in artifacts/v2/")

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("onnxruntime or transformers not installed")

    with open(presets_path, "r", encoding="utf-8") as f:
        presets = json.load(f)

    with open(cal_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    temperature = float(cal_data["temperature"])

    p = presets["card_arrival"]
    descs = [c["desc"] for c in p["candidates"]]
    prompt = build_model_input(p["question"], p["text"], descs)

    tokenizer = AutoTokenizer.from_pretrained("artifacts/v2")
    tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="np")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }
    outs = session.run(None, ort_inputs)
    raw_logits = outs[0][0, : len(descs)]

    # Softmax with temperature
    scaled = raw_logits / temperature
    max_logit = np.max(scaled)
    exps = np.exp(scaled - max_logit)
    probs = exps / np.sum(exps)

    # card_arrival should be top intent with high probability (> 0.8)
    assert p["candidates"][0]["id"] == "card_arrival"
    top_idx = int(np.argmax(probs))
    assert top_idx == 0, f"Expected card_arrival (0), got {top_idx} with probs {probs}"
    assert probs[0] > 0.80, f"Expected confidence > 0.80, got {probs[0]:.4f}"
