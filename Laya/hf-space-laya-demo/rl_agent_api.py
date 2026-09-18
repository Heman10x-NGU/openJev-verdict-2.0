"""Jev-compatible inference for a saved RL Agent model: system_one(state, questions) -> typed answers."""
import json
import math
import os

import numpy as np
import torch

from rl_common import (QTYPES, amp_dtype, build_model, build_sequence, collate_items, confidence_from_probs,
                       render_options, temp_bucket)


def _verify_compatibility(model: torch.nn.Module, cfg: dict, weights: dict, model_dir: str):
    """Verify that the loaded checkpoint weights and config strictly match the expected architecture."""
    required_cfg = ["encoder", "head_layers"]
    missing_cfg = [k for k in required_cfg if k not in cfg]
    if missing_cfg:
        raise ValueError(
            f"Incompatible model config for {model_dir!r}: missing keys {missing_cfg}. "
            f"Ensure this is a valid RL Agent decision model."
        )

    required_prefixes = ("encoder.", "type_emb.", "scorer.", "act_head.")
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in weights.keys()):
            raise ValueError(
                f"Incompatible model weights for {model_dir!r}: checkpoint is missing '{prefix}' parameters. "
                f"Expected an RL Agent decision model with encoder and decision heads."
            )

    shape_mismatches = []
    missing_keys = []
    for name, param in model.named_parameters():
        if name not in weights:
            missing_keys.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            shape_mismatches.append(f"  - {name}: expected {tuple(param.shape)}, found {tuple(weights[name].shape)}")

    if shape_mismatches:
        err_details = "\n".join(shape_mismatches[:5])
        if len(shape_mismatches) > 5:
            err_details += f"\n  ... and {len(shape_mismatches) - 5} more mismatched layers."
        raise ValueError(
            f"Model architecture mismatch for {model_dir!r}:\n{err_details}\n"
            f"The checkpoint weights do not match the configured model architecture."
        )

    if missing_keys:
        raise ValueError(f"Model weights incomplete for {model_dir!r}: missing {len(missing_keys)} parameter tensors.")


class RLAgent:
    def __init__(self, model_dir, device=None):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Incompatible model: 'rl_agent_config.json' not found in {model_dir!r}.")

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Incompatible model: 'model.safetensors' not found in {model_dir!r}.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.model = build_model(self.cfg, encoder_dir=os.path.join(model_dir, "encoder"))

        weights = load_file(weights_path)
        _verify_compatibility(self.model, self.cfg, weights, model_dir)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.model.encoder.config.reference_compile = False  # torch.compile is a loss on small batches / few SMs (T4)
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.dtype = amp_dtype(self.cfg.get("amp_dtype", "fp16"))
        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.dtype = torch.float16  # e.g. a bf16-trained model evaluated on a T4

    @staticmethod
    def _to_internal(qdef):
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        return {"t": t, "ins": qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"]),
                "crit": crit}

    @torch.no_grad()
    def system_one(self, state, questions):
        """questions: {id: {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}} (Jev request shape)."""
        ids, items = list(questions.keys()), []
        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r: options do not fit in head_max_len=%d tokens" % (qid, self.cfg["head_max_len"]))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers), "label": -1,
                          "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
        b = collate_items([items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
            logits, act = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device),
                                     b["marker_pos"].to(self.device), b["marker_mask"].to(self.device), b["qtype"].to(self.device))
        logits, act = logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()
        answers, n_tokens = {}, int(b["attention_mask"].sum())
        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            z = logits[r, :k] / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {"model": "laya", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
