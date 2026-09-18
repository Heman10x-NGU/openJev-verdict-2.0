import copy
import random

import pytest
import torch

from kev import evaluate
from kev.data import materialize
from kev.train import question_loss


def choice_request():
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?",
        "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"},
        "label": "size", "src": "fixture",
    }}}


def test_clean_evaluation_does_not_add_options(monkeypatch):
    observed = []

    def predict(tok, model, req):
        observed.append(copy.deepcopy(req))
        rec = materialize(req)
        return rec, [torch.ones(len(q["options"])) / len(q["options"]) for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    evaluate.test_accuracy(None, None, [choice_request() for _ in range(100)], random.Random(1))
    assert all(set(r["questions"]["reason"]["criteria"]) == {"size", "damage", "color"} for r in observed)


def test_none_removed_relabels_and_counts_complete_pairs(monkeypatch):
    observed = []

    def predict(tok, model, req):
        rec = materialize(req)
        observed.append(copy.deepcopy(req))
        return rec, [torch.nn.functional.one_hot(torch.tensor(q["label"]), len(q["options"])).float() for q in rec["questions"]]

    monkeypatch.setattr(evaluate, "_probs", predict)
    report = evaluate.test_none_of_the_above(None, None, [choice_request()], random.Random(1))
    assert report["n"] == 1
    assert report["true_option_present"]["acc"] == 1
    assert report["true_option_removed"]["picks_none_rate"] == 1
    assert observed[1]["questions"]["reason"]["label"] != "size"


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * question_loss(logits, {"label": y, "qtype": "score"}, "cpu", 0.5) for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


def frozen_request(i=0):
    r = choice_request()
    r["_meta"] = {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": "clean"}
    return r


def test_contrast_cases_preserve_groups_and_relabel():
    from kev.suite import contrast_cases
    original = frozen_request()
    present, absent, permuted = contrast_cases(original)
    assert present["questions"]["reason"]["label"] == "size"
    assert absent["questions"]["reason"]["label"] == "none_of_these"
    for record in (present, absent, permuted):
        assert record["_meta"]["group_id"] == original["_meta"]["id"]
        materialize(record)
    assert original == frozen_request()


def test_source_sampling_does_not_depend_on_other_sources(monkeypatch):
    from kev import data

    def convert(split, n, rng):
        value = rng.randrange(1000000)
        rng.origins = [{"row": value, "row_sha256": str(value), "text_sha256": str(value)}]
        return [choice_request()]

    monkeypatch.setattr(data, "SOURCES", {"agnews": (convert, "train", "test"), "mnli": (convert, "train", "test")})
    alone = data.build(1, only=["mnli"])
    together = data.build(1)
    assert alone[0] == next(r for r in together if r["_meta"]["source"] == "mnli")


def test_strict_encoding_rejects_truncation():
    from types import SimpleNamespace
    from kev.model import encode

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(range(len(text))))

        def convert_tokens_to_ids(self, text):
            return 1000

    rec = {"state": "abcdefgh", "questions": [{"instr": "q", "options": ["a", "b"], "label": 0}]}
    assert encode(Tokenizer(), rec, max_state=4)["state_truncated"]
    with pytest.raises(ValueError, match="state exceeds"):
        encode(Tokenizer(), rec, max_state=4, strict=True)


def test_locked_split_and_hash_verification(tmp_path):
    import json
    from kev.suite import digest, load_split, write_json
    for name in ("development", "test"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(frozen_request()) + "\n")
    manifest = {"files": {f"{name}.jsonl": {"sha256": digest(tmp_path / f"{name}.jsonl"), "records": 1} for name in ("development", "test")}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_split(tmp_path, "development")) == 1
    with pytest.raises(ValueError, match="locked test"):
        load_split(tmp_path, "test")
    (tmp_path / "development.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path, "development")


def test_api_payload_excludes_answers_and_metadata():
    from kev.benchmark import api_request
    clean = api_request(frozen_request())
    assert set(clean) == {"state", "questions"}
    assert set(clean["questions"]["reason"]) == {"type", "instructions", "criteria"}


def test_failed_prediction_cannot_produce_partial_score(tmp_path):
    import json
    from kev.benchmark import evaluate_records

    def fail(record):
        raise ValueError("invalid prediction")

    out = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate_records([frozen_request()], fail, out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["coverage"]["requested_records"] == 1
    assert failure["coverage"]["evaluated_records"] == 0
    assert failure["coverage"]["rejected_records"] == 1
    assert not (out / "report.json").exists()


def test_missing_answers_and_nonfinite_probabilities_fail():
    from kev.benchmark import prediction_rows, validate_distribution
    with pytest.raises(ValueError, match="answer IDs"):
        prediction_rows(frozen_request(), {"probabilities": {}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_distribution({"x": float("nan"), "y": 0.5}, ["x", "y"])
    with pytest.raises(ValueError, match="sum"):
        validate_distribution({"x": 0, "y": 0}, ["x", "y"])


def test_task_macro_and_record_bootstrap():
    from kev.benchmark import prediction_rows, summarize, paired_bootstrap
    pred = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}}
    rows = prediction_rows(frozen_request(), pred)
    report = summarize(rows)
    assert report["objective"] == pytest.approx(-report["clean"]["nll"])
    assert paired_bootstrap(rows, rows, samples=50)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows, [])


def test_trial_config_cannot_change_evaluator_or_read_test():
    from kev.experiment import validated_trial
    manifest = {"base_revisions": {"model": "pinned"}}
    assert validated_trial({"base": "model"}, manifest)["ord_w"] == 0
    for extra in ({"test": True}, {"command": "echo x"}, {"lr": -1}, {"epochs": 2.5}):
        with pytest.raises(ValueError):
            validated_trial({"base": "model", **extra}, manifest)
