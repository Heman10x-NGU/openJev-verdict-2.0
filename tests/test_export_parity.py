"""Verify mathematical parity between PyTorch and exported ONNX runtime models.

Ensures that candidate logits produced by PyTorch and ONNX Runtime match
within numerical tolerance across variable sequence lengths and cardinalities.
Parameterised over model bundles and enforces failure when CI_REQUIRE_ARTIFACTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.formatting import build_model_input
from gliclass import GLiClassModel

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


BUNDLE_CONFIGS = [
    (
        "v2",
        Path("artifacts/v2/model.safetensors"),
        Path("artifacts/v2/model.onnx"),
        1e-3,
    ),
    (
        "v2_fp16",
        Path("artifacts/v2/model.safetensors"),
        Path("artifacts/v2/model_fp16.onnx"),
        2e-2,
    ),
    (
        "v1",
        Path("artifacts/openjev_modernbert.safetensors"),
        Path("artifacts/openjev_modernbert.onnx"),
        1e-3,
    ),
]


def check_artifacts_or_skip(pt_path: Path, onnx_path: Path, required: bool = True) -> None:
    ci_require = os.environ.get("CI_REQUIRE_ARTIFACTS") == "1"
    if not pt_path.exists():
        if ci_require and required:
            pytest.fail(f"Required PyTorch artifact missing under CI_REQUIRE_ARTIFACTS=1: {pt_path}")
        pytest.skip(f"PyTorch artifact {pt_path} not found")
    if not onnx_path.exists():
        if ci_require and required:
            pytest.fail(f"Required ONNX artifact missing under CI_REQUIRE_ARTIFACTS=1: {onnx_path}")
        pytest.skip(f"ONNX artifact {onnx_path} not found")


@pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime is not installed")
@pytest.mark.parametrize("bundle_name, pt_path, onnx_path, tolerance", BUNDLE_CONFIGS)
def test_onnx_pytorch_parity(
    bundle_name: str,
    pt_path: Path,
    onnx_path: Path,
    tolerance: float,
) -> None:
    check_artifacts_or_skip(pt_path, onnx_path, required=(bundle_name != "v1"))

    model_name = "knowledgator/gliclass-modern-base-v2.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load PyTorch model
    pt_model = GLiClassModel.from_pretrained(model_name).cpu().eval()
    pt_model.load_state_dict(load_file(str(pt_path)))

    # Load ONNX session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_queries = [
        (
            "Primary database leader node crashed with disk corruption.",
            ["p0_incident", "p2_degradation", "routine_maintenance", "__insufficient_evidence__"],
        ),
        (
            "Customer charged $42.00 twice on credit card.",
            ["dispute_chargeback", "freeze_account", "credit_limit", "__insufficient_evidence__"],
        ),
        (
            "The weather in San Francisco is foggy today.",
            ["financial_action", "cloud_action", "__insufficient_evidence__"],
        ),
    ]

    question = "Classify the following query into the appropriate category:"
    for text, labels in test_queries:
        prompt = build_model_input(question, text, labels)

        tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # PyTorch forward
        with torch.no_grad():
            pt_out = pt_model(input_ids=input_ids, attention_mask=attention_mask)
            pt_logits = pt_out.logits[0, : len(labels)].numpy()

        # ONNX forward
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
        ort_outs = session.run(None, ort_inputs)
        onnx_logits = ort_outs[0][0, : len(labels)]

        max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
        assert max_diff <= tolerance, (
            f"Bundle {bundle_name}: max diff {max_diff:.6f} exceeds tolerance {tolerance} "
            f"for query '{text[:30]}...'"
        )


@pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime is not installed")
@pytest.mark.parametrize("k", [3, 5, 9, 17, 25])
def test_padding_invariance_across_cardinalities(k: int) -> None:
    """Test that requests with varying K candidate counts maintain parity and ranking."""
    pt_path = Path("artifacts/v2/model.safetensors")
    onnx_path = Path("artifacts/v2/model.onnx")
    check_artifacts_or_skip(pt_path, onnx_path)

    model_name = "knowledgator/gliclass-modern-base-v2.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_model = GLiClassModel.from_pretrained(model_name).cpu().eval()
    pt_model.load_state_dict(load_file(str(pt_path)))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    text = "Cardholder was charged $120.00 twice on credit card at merchant CloudHost."
    labels = [f"candidate_choice_{i}" for i in range(k - 1)] + ["insufficient evidence"]
    question = "Classify the following query into the appropriate category:"

    prompt = build_model_input(question, text, labels)

    tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        pt_out = pt_model(input_ids=input_ids, attention_mask=attention_mask)
        pt_logits = pt_out.logits[0, :k].numpy()
        pt_probs = torch.softmax(torch.tensor(pt_logits), dim=-1).numpy()

    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    ort_outs = session.run(None, ort_inputs)
    onnx_logits = ort_outs[0][0, :k]
    onnx_probs = np.exp(onnx_logits - np.max(onnx_logits))
    onnx_probs = onnx_probs / np.sum(onnx_probs)

    logit_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    prob_diff = float(np.max(np.abs(pt_probs - onnx_probs)))

    assert logit_diff < 1e-3, f"Logit diff {logit_diff} exceeds 1e-3 for K={k}"
    assert prob_diff < 1e-4, f"Prob diff {prob_diff} exceeds 1e-4 for K={k}"
    assert np.argmax(pt_logits) == np.argmax(onnx_logits), f"Ranking mismatch for K={k}"
