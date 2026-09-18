"""Dataset loading, sequence construction, and case-level splits for typed-decisions.

Sequence layout follows the encoder marker-pointer design:
    [CLS] {qtype} question: {instructions} [SEP] [MASK]{opt0} [MASK]{opt1} ... [SEP] {state} [SEP]
The hidden state at each [MASK] marker is scored to produce one logit per option.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}

NOUL_DEFAULT = {
    "false": "no, the statement does not hold",
    "true": "yes, the statement holds",
}


@dataclass(frozen=True)
class Item:
    """One typed question, ready for the encoder."""

    case_id: str
    workflow: str
    qid: str
    qtype: int
    ids: Tuple[int, ...]
    markers: Tuple[int, ...]
    target: Tuple[float, ...]   # soft teacher distribution over options
    label: int                  # gold argmax label index
    option_keys: Tuple[str, ...]
    gold_score: float           # teacher expected level, score questions only


def _parse(row: Dict[str, Any], key: str) -> Any:
    value = row[key]
    return json.loads(value) if isinstance(value, str) else value


def render_options(qtype: str, criteria: Any) -> Tuple[List[str], List[str]]:
    """Return (rendered option texts, option keys) in a stable, label-aligned order."""
    if qtype == "choice":
        keys = list(criteria.keys())
        return [f"{k}: {criteria[k]}" if criteria[k] else k for k in keys], keys
    if qtype == "score":
        return [f"level {i}: {c}" for i, c in enumerate(criteria)], [str(i) for i in range(len(criteria))]
    crit = criteria or {}
    keys = ["false", "true"]
    return [f"{k}: {crit.get(k) or NOUL_DEFAULT[k]}" for k in keys], keys


def build_item(
    tokenizer: Any,
    row: Dict[str, Any],
    qid: str,
    qdef: Dict[str, Any],
    gold: Dict[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Sequence[int] | None = None,
) -> Item | None:
    """Tokenize one question. Returns None when the option markers do not fit in max_len."""
    qtype = qdef["type"]
    texts, keys = render_options(qtype, qdef.get("criteria"))
    order = list(option_order) if option_order is not None else list(range(len(texts)))

    mask_tok = tokenizer.mask_token
    instructions = str(qdef["instructions"]).replace(mask_tok, " ")

    head_ids = tokenizer(f"{qtype} question: {instructions}", add_special_tokens=False)["input_ids"]
    opt_ids: List[List[int]] = []
    for i in order:
        body = tokenizer(" " + texts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([tokenizer.mask_token_id] + body)

    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, budget)]

    ids: List[int] = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers: List[int] = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tokenizer.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    state = tokenizer(str(row["state"]).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = (ids + state + [tokenizer.sep_token_id])[:max_len]

    if any(m >= max_len for m in markers):
        return None

    probs = gold.get("probabilities", {})
    raw = [float(probs.get(keys[i], 0.0)) for i in order]
    total = sum(raw)
    target = [p / total for p in raw] if total > 0 else [1.0 / len(raw)] * len(raw)

    ordered_keys = [keys[i] for i in order]
    gold_label = str(gold.get("label", ""))
    label = ordered_keys.index(gold_label) if gold_label in ordered_keys else int(max(range(len(target)), key=target.__getitem__))

    return Item(
        case_id=str(row.get("id", "")),
        workflow=str(row.get("workflow", "")),
        qid=qid,
        qtype=QTYPES[qtype],
        ids=tuple(ids),
        markers=tuple(markers),
        target=tuple(target),
        label=label,
        option_keys=tuple(ordered_keys),
        gold_score=float(gold.get("score", 0.0)),
    )


def load_items(split: str, tokenizer: Any, limit: int | None = None) -> List[Item]:
    """Load one split of LocalLLaMA/typed-decisions as a flat list of Items."""
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    n = len(ds) if limit is None else min(len(ds), limit)
    items: List[Item] = []
    for i in range(n):
        row = ds[i]
        questions = _parse(row, "questions")
        gold = _parse(row, "gold")
        for qid, qdef in questions.items():
            if qid not in gold:
                continue
            item = build_item(tokenizer, row, qid, qdef, gold[qid])
            if item is not None:
                items.append(item)
    return items


def split_by_case(items: List[Item], fit: float = 0.70, calib: float = 0.15, seed: int = 0) -> Dict[str, List[Item]]:
    """Partition by case id so sibling questions never straddle a fold boundary."""
    case_ids = sorted({it.case_id for it in items})

    def bucket(case_id: str) -> float:
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    assignment: Dict[str, str] = {}
    for case_id in case_ids:
        u = bucket(case_id)
        assignment[case_id] = "fit" if u < fit else ("calib" if u < fit + calib else "dev")

    folds: Dict[str, List[Item]] = {"fit": [], "calib": [], "dev": []}
    for it in items:
        folds[assignment[it.case_id]].append(it)
    return folds


def collate(batch: List[Item], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch of Items into dense tensors."""
    n = len(batch)
    length = max(len(it.ids) for it in batch)
    kmax = max(len(it.markers) for it in batch)

    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attn = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(batch):
        ids[i, : len(it.ids)] = torch.tensor(it.ids, dtype=torch.long)
        attn[i, : len(it.ids)] = 1
        k = len(it.markers)
        mpos[i, :k] = torch.tensor(it.markers, dtype=torch.long)
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it.target, dtype=torch.float32)

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "label": torch.tensor([it.label for it in batch], dtype=torch.long),
        "qtype": torch.tensor([it.qtype for it in batch], dtype=torch.long),
    }
