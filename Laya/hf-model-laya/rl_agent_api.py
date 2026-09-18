"""Jev-compatible inference for a saved RL Agent model: system_one(state, questions) -> typed answers."""
import json
import math
import os

import numpy as np
import torch

from rl_common import (QTYPES, amp_dtype, build_model, build_sequence, collate_items, confidence_from_probs,
                       render_options, temp_bucket)


class RLAgent:
    def __init__(self, model_dir, device=None):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.model = build_model(self.cfg, encoder_dir=os.path.join(model_dir, "encoder"))
        self.model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
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
        return {"model": "rl-agent", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}
