"""Unit tests for causal decision adapter and exact token boundary assertions."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from core.engine_causal import (
    CausalDecisionAdapter,
    TokenBoundaryError,
)
from core.primitives import (
    Choice,
    Level,
    Noul,
    Option,
    Score,
)


class MockTokenizer:
    """Mock tokenizer simulating single-token slots and prompt encoding."""

    def __init__(self, corrupt_slot: str | None = None, corrupt_boundary: str | None = None):
        self.corrupt_slot = corrupt_slot
        self.corrupt_boundary = corrupt_boundary
        self.pad_token_id = 0
        # Map letters A..P to token IDs 100..115
        self.char_to_id = {chr(65 + i): 100 + i for i in range(16)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.corrupt_slot and text == self.corrupt_slot:
            return [999, 998]  # Multi-token corruption
        if self.corrupt_boundary and len(text) > 1 and text.endswith(self.corrupt_boundary):
            # Corrupted prefix boundary where trailing char merges or splits
            base = [ord(c) % 50 + 1 for c in text[: -len(self.corrupt_boundary)]]
            return base + [888, 889]
        
        # Tokenize consistently: for each character, map if in char_to_id or fallback
        tokens = []
        for c in text:
            if c in self.char_to_id:
                tokens.append(self.char_to_id[c])
            else:
                tokens.append(ord(c) % 50 + 1)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        if len(token_ids) == 1 and token_ids[0] in self.id_to_char:
            return self.id_to_char[token_ids[0]]
        return "".join(self.id_to_char.get(t, chr((t % 26) + 65)) for t in token_ids)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        rendered = " ".join(m["content"] for m in messages)
        return f"<chat>{rendered}</chat>Answer:"


class MockCausalModel(torch.nn.Module):
    """Mock causal transformer returning fixed vocab logits."""

    def __init__(self, target_slot_idx: int = 0):
        super().__init__()
        self.target_slot_idx = target_slot_idx

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any):
        batch_size = input_ids.shape[0]
        vocab_size = 200
        # Create logits where token corresponding to target slot gets high logit
        logits = torch.zeros((batch_size, input_ids.shape[1], vocab_size), dtype=torch.float32)
        # Letter A is token 100, B is 101, etc.
        target_token = 100 + self.target_slot_idx
        logits[:, -1, target_token] = 10.0
        return SimpleNamespace(logits=logits)


def test_slot_tokens_validation_success() -> None:
    tok = MockTokenizer()
    slots = CausalDecisionAdapter.get_slot_tokens(tok, count=5)
    assert len(slots) == 5
    assert slots == [100, 101, 102, 103, 104]


def test_slot_tokens_validation_failure_on_multitoken() -> None:
    tok = MockTokenizer(corrupt_slot="B")
    with pytest.raises(TokenBoundaryError, match="encoded to 2 tokens"):
        CausalDecisionAdapter.get_slot_tokens(tok, count=3)


def test_boundary_assertion_failure() -> None:
    tok = MockTokenizer(corrupt_boundary="A")
    adapter = CausalDecisionAdapter(tokenizer=tok)
    query = Choice(
        id="q1",
        question="Select option",
        options=[
            Option(id="opt1", description="Option 1"),
            Option(id="opt2", description="Option 2"),
        ],
    )
    with pytest.raises(TokenBoundaryError, match="Prompt boundary corrupted tokenization"):
        adapter.render_prompt_and_validate_boundaries("Context...", query)


def test_causal_adapter_batched_evaluation() -> None:
    tok = MockTokenizer()
    model = MockCausalModel(target_slot_idx=1)  # Predict slot B
    adapter = CausalDecisionAdapter(model=model, tokenizer=tok)

    q_choice = Choice(
        id="q_choice",
        question="Routing question",
        options=[
            Option(id="opt_a", description="Alpha"),
            Option(id="opt_b", description="Beta"),
        ],
    )
    q_score = Score(
        id="q_score",
        question="Score question",
        levels=[
            Level(id="lvl1", description="Low", value=1.0),
            Level(id="lvl2", description="High", value=5.0),
        ],
    )
    q_noul = Noul(
        id="q_noul",
        proposition="The claim is supported.",
        semantics="conditional_on_sufficient_evidence_v2",
    )

    batch_res = adapter.evaluate(
        context="Evaluation context data.",
        queries=[q_choice, q_score, q_noul],
    )

    assert batch_res.forward_call_count == 1
    assert len(batch_res.results) == 3

    # Choice result (slot B corresponds to opt_b)
    c_res = batch_res.results[0]
    assert c_res.selected_id == "opt_b"
    assert c_res.selected_probability > 0.9

    # Score result (slot B corresponds to lvl2)
    s_res = batch_res.results[1]
    assert s_res.selected_level_id == "lvl2"
    assert s_res.selected_value == 5.0
    assert pytest.approx(s_res.expected_score, abs=0.1) == 5.0

    # Noul result (slot B corresponds to false)
    n_res = batch_res.results[2]
    assert n_res.selected_outcome == "false"
    assert not n_res.is_abstention
