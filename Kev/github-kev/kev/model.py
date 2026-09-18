"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option).
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos = list(S), [0] * len(S), list(range(len(S)))
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        br = [q_id] + user_tokens(tok, q["instr"])
        oi = []
        for o in q["options"]:
            br += [o_id] + user_tokens(tok, o) + [c_id]
            oi.append(len(br) - 1)
        br.append(d_id)
        if len(br) > max_branch - len(S):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids)
        ids += br; seg += [k] * len(br); pos += list(range(len(S), len(S) + len(br)))
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + i for i in oi])
    return {"ids": ids, "seg": seg, "pos": pos, "decide_idx": decide_idx, "opt_idx": opt_idx, "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    s = torch.tensor(seg, device=device)
    L = len(seg)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[None, :] == s[:, None]) | (s[None, :] == 0)
    allow = causal & same
    return torch.zeros(L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[None, None]


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None):
        super().__init__()
        # backbone only (no vocab head): we never generate text
        self.lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32, attn_implementation="eager").model
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size)
        self.device = device
        self.to(device)

    def hidden(self, enc):
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        mask = branch_mask(enc["seg"], self.device)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state[0]

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        h = self.hidden(enc)
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
