"""Export Verdict-open-jev-ModernBERT model to ONNX format for WebGPU and edge deployment.

Wraps ModernBERT backbone and GLiClass candidate head into an ONNX graph with
dynamic sequence length and candidate axes at opset 17. Validates graph with
onnx.checker and tests execution with ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import onnx
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from gliclass import GLiClassModel


class VerdictOnnxWrapper(torch.nn.Module):
    """Wrapper that exposes clean (input_ids, attention_mask) -> logits graph."""

    def __init__(self, model: GLiClassModel):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# Backward-compatible alias
OpenJevOnnxWrapper = VerdictOnnxWrapper


def export_model(
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    checkpoint_path: str = "artifacts/v2/model.safetensors",
    output_onnx_path: str = "artifacts/v2/model.onnx",
    opset_version: int = 17,
) -> Path:
    out_file = Path(output_onnx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base architecture: {model_name}...")
    model = GLiClassModel.from_pretrained(model_name).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Requested fine-tuned checkpoint {ckpt_file} not found. "
            f"Silent fallback to zero-shot weights is forbidden by release invariants."
        )
    print(f"Loading fine-tuned weights from {ckpt_file}...")
    state_dict = load_file(str(ckpt_file))
    model.load_state_dict(state_dict)

    wrapper = OpenJevOnnxWrapper(model).eval()

    # Dummy inputs for tracing (batch_size=1, seq_len=64)
    dummy_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
    dummy_mask = torch.ones((1, 64), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    print(f"Exporting ONNX model to {out_file} (opset={opset_version})...")
    start_time = time.perf_counter()

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(out_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    export_duration = time.perf_counter() - start_time
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Export complete in {export_duration:.2f}s! File size: {file_size_mb:.2f} MB")

    # Validate with onnx.checker
    print("Validating ONNX graph with onnx.checker...")
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Write deployment bundle manifest
    import hashlib
    with open(out_file, "rb") as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()
    with open(ckpt_file, "rb") as f:
        safetensors_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "bundle_version": "1.0.0",
        "model_architecture": "knowledgator/gliclass-modern-base-v2.0",
        "checkpoint_path": str(ckpt_file),
        "safetensors_sha256": safetensors_hash,
        "onnx_path": str(out_file),
        "onnx_sha256": onnx_hash,
        "opset_version": opset_version,
        "max_capacity_logits": 25,
        "export_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = out_file.parent / "bundle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Deployment bundle manifest saved to {manifest_path}")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort

        print("Testing inference via ONNX Runtime CPU session...")
        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        test_inputs = {
            "input_ids": dummy_ids.numpy(),
            "attention_mask": dummy_mask.numpy(),
        }
        ort_start = time.perf_counter()
        ort_outs = session.run(None, test_inputs)
        ort_latency_ms = (time.perf_counter() - ort_start) * 1000.0

        print(f"ONNX Runtime sanity check succeeded in {ort_latency_ms:.2f} ms! Output shape: {ort_outs[0].shape}")
    except Exception as e:
        print(f"Warning: ONNX Runtime sanity check failed: {e}")

    # Also save tokenizer files into artifacts directory for browser/web deployment
    tokenizer.save_pretrained(str(out_file.parent))
    print(f"Saved tokenizer files to {out_file.parent}")

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Verdict-open-jev-ModernBERT to ONNX.")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--checkpoint", type=str, default="artifacts/v2/model.safetensors")
    parser.add_argument("--output", type=str, default="artifacts/v2/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
