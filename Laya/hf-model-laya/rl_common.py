"""RL Agent shared code: config, Jev-style question rendering, model, proper-scoring rewards, metrics.

Kept Python 3.9 compatible so the same file runs on Kaggle and on a laptop smoke test.
"""
import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


# ----------------------------------------------------------------------------- config
def load_cfg(path: Optional[str] = None) -> Dict:
    path = path or os.environ.get("RL_AGENT_CFG", "rl_agent_config.json")
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- rendering
def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q: Dict) -> List[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q: Dict, max_len: int, head_max_len: int,
                   option_order: Optional[List[int]] = None, truncate_left: bool = False):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Returns input_ids and the positions of the per-option [MASK] markers (in the given option order).
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


# ----------------------------------------------------------------------------- model
class DecisionModel(nn.Module):
    """Pretrained bidirectional encoder (no LLM, no LoRA) + from-scratch decision head.

    Each option gets a [MASK] marker; the head scores markers -> softmax over the question's options.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))  # per qtype, fitted post-hoc in evaluate.py
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(layer, h, None, pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head sees the pooled sequence + detached summary of its own answer distribution
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    from transformers import AutoConfig, AutoModel
    if encoder_dir:  # offline: architecture only, weights come from the saved state dict
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)


# ----------------------------------------------------------------------------- rewards (strictly proper)
def proper_reward(q: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor, mask: torch.Tensor,
                  w_sph: float = 0.5, w_rps: float = 1.0, log_floor: float = -9.21) -> torch.Tensor:
    """q: [..., N, K] reported distributions, target: [N, K] (one-hot or soft) -> reward [..., N].

    log score + spherical score for all types, + ranked probability score for ordinal (score) questions.
    All three are strictly proper, so the only way to maximize reward is to report honest probabilities.
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


# ----------------------------------------------------------------------------- metrics (numpy, no sklearn)
def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(-conf)
    err = 1 - correct[order]
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- record -> model inputs
def episode_prefix_lengths(n_turns: int, max_prefixes: int) -> List[int]:
    if n_turns <= max_prefixes:
        return list(range(1, n_turns + 1))
    return sorted(set(int(round(x)) for x in np.linspace(1, n_turns, max_prefixes)))


def encode_record(rec: Dict, tok, cfg: Dict, rng: Optional[random.Random], train: bool) -> List[Dict]:
    """One stored record -> list of model sequences (one per question, or one per conversation prefix)."""
    items = []
    if rec.get("kind") == "episode":
        ep, q = rec["ep"], rec["qs"][0]
        lens = episode_prefix_lengths(len(ep["turns"]), cfg["max_prefixes"])
        for step, t in enumerate(lens):
            state = dict(ep["ctx"], conversation=ep["turns"][:t])
            ids, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"], truncate_left=True)
            if len(markers) != 2:
                continue
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES["noul"], "target": [1.0 - ep["y"], float(ep["y"])],
                          "label": int(ep["y"]), "episode": 1, "ep_step": step, "ep_len": len(lens), "src": rec.get("src", ""),
                          "prefix_frac": t / float(len(ep["turns"]))})
        return items
    for qi, q in enumerate(rec["qs"]):
        k = len(render_options(q))
        target = list(q["soft"]) if q.get("soft") else [1.0 if i == q["y"] else 0.0 for i in range(k)]
        order = list(range(k))
        if train and rng is not None and q["t"] != "score":
            rng.shuffle(order)
        ids, markers = build_sequence(tok, rec["state"], q, cfg["max_len"], cfg["head_max_len"], option_order=order)
        if len(markers) != k:
            continue  # options did not fit; skip rather than train on a truncated answer space
        target = [target[i] for i in order]
        label = order.index(q["y"]) if q.get("y") is not None else -1
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": target, "label": label,
                      "episode": 0, "ep_step": 0, "ep_len": 1, "src": rec.get("src", ""), "q_index": qi, "order": order})
    return items


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    ep_group = torch.full((n,), -1, dtype=torch.long)
    group_of = {}
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    # episodes: all prefixes of the same record share a group id (used for TD(lambda) targets)
    for i, it in enumerate(items):
        if it["episode"]:
            ep_group[i] = group_of.setdefault(it.get("rec_uid", -1 - i), len(group_of))
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items]),
            "episode": torch.tensor([it["episode"] for it in items], dtype=torch.bool), "ep_group": ep_group,
            "ep_step": torch.tensor([it["ep_step"] for it in items]), "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
            "n_tokens": int(att.sum())}


def pack_groups(groups: List[List[Dict]], max_tokens: int, max_seqs: int) -> List[List[List[Dict]]]:
    """Split one sampled batch into sub-batches using the *real* tokenized lengths, so padded tokens never exceed
    max_tokens (the index only stores estimates). A record's items stay together (TD targets need all prefixes)."""
    groups = sorted([g for g in groups if g], key=lambda g: max(len(it["ids"]) for it in g))
    subs, cur, cur_max, cur_n = [], [], 0, 0
    for g in groups:
        g_max, g_n = max(len(it["ids"]) for it in g), len(g)
        if g_max * g_n > max_tokens:  # one record bigger than the budget (only if max_tokens < max_len * n_items)
            step = max(1, max_tokens // g_max)
            for s in range(0, g_n, step):
                subs.append([g[s:s + step]])
            continue
        new_max, new_n = max(cur_max, g_max), cur_n + g_n
        if cur and (new_max * new_n > max_tokens or new_n > max_seqs):
            subs.append(cur)
            cur, new_max, new_n = [], g_max, g_n
        cur.append(g)
        cur_max, cur_n = new_max, new_n
    if cur:
        subs.append(cur)
    return subs


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float) -> torch.Tensor:
    """TD(lambda) soft targets for conversation prefixes: G_last = outcome, G_t = (1-lam) V_{t+1} + lam G_{t+1}."""
    target = batch["target"].clone()
    groups = batch["ep_group"]
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def make_token_batches(lengths: np.ndarray, nseq: np.ndarray, max_tokens: int, max_seqs: int, rng: np.random.RandomState,
                       chunk: int = 4096) -> List[List[int]]:
    """Length-bucketed batches of record indices under a padded-token budget."""
    order = rng.permutation(len(lengths))
    batches = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(lengths[part])]
        cur, cur_max, cur_n = [], 0, 0
        for i in part:
            ln, ns = int(lengths[i]), int(nseq[i])
            new_max, new_n = max(cur_max, ln), cur_n + ns
            if cur and (new_max * new_n > max_tokens or new_n > max_seqs):
                batches.append(cur)
                cur, new_max, new_n = [], ln, ns
            cur.append(int(i))
            cur_max, cur_n = new_max, new_n
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def temp_bucket(qtype: int, k: int) -> str:
    """Key for per-cardinality temperature fitting: a 2-option noul and a 20-option choice need different scaling."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def amp_dtype(name: Optional[str]) -> torch.dtype:
    """'bf16' on GPUs that support it (Ampere+, e.g. RTX 6000 Pro); 'fp16' on T4."""
    return torch.bfloat16 if name == "bf16" else torch.float16


@torch.no_grad()
def predict_items(model, items: List[Dict], pad_id: int = 0, device=None, max_tokens: int = 16384, use_amp: bool = True,
                  dtype: torch.dtype = torch.float16, max_seqs: int = 256, progress: str = ""):
    """Run the model over pre-encoded items; returns list of dicts with probs/logits (uncalibrated) and act probs."""
    import sys
    import time as _time
    model.eval()
    out = []
    t0, done_tok = _time.time(), 0
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    i = 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"]))
            j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], pad_id)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp and device.type == "cuda"):
            logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                b["marker_mask"].to(device), b["qtype"].to(device))
        logits, act = logits.float().cpu(), torch.softmax(act.float(), -1).cpu()
        done_tok += int(b["attention_mask"].sum())
        if progress and (j % max(1, len(order) // 2000) == 0 or j >= len(order)):
            el = _time.time() - t0
            eta = el * (len(order) - j) / max(1, j)
            sys.stdout.write("\r  [%s] %d/%d sequences | %.1fk tok/s | ETA %dm%02ds   " %
                             (progress, j, len(order), done_tok / max(el, 1e-9) / 1000, int(eta // 60), int(eta % 60)))
            sys.stdout.flush()
        for r, it in enumerate(sel):
            k = len(it["markers"])
            out.append((order[i + r], {"logits": logits[r, :k].detach().numpy(), "act": act[r].detach().numpy()}))
        i = j
    if progress:
        print("\r  [%s] %d sequences in %.0fs (%.1fk tok/s)%s" % (progress, len(order), _time.time() - t0,
                                                                  done_tok / max(_time.time() - t0, 1e-9) / 1000, " " * 20))
    out.sort(key=lambda x: x[0])
    model.train()
    return [o for _, o in out]
