"""Convert public labelled datasets into TypeSafe-shaped requests, then through api.to_record() so the
training format is byte-identical to what /v1/systemone feeds the model.

A "labelled request" is {"state": JSONContent, "questions": {id: {type, instructions, criteria, "label": ..., "src": str}}}
  label: choice -> option key, noul -> bool, score -> level index.
materialize() -> internal record {"state": str, "questions": [{"instr", "options", "label": int, "src"}]}
"""
import hashlib
import json
import random
from datasets import load_dataset
from .api import SystemOneRequest, to_record


REPOS = {"banking77": "legacy-datasets/banking77", "boolq": "google/boolq", "agnews": "fancyzhx/ag_news",
         "mnli": "nyu-mll/multi_nli", "sst5": "SetFit/sst5", "yelp": "Yelp/yelp_review_full"}


def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def _dataset(repo, split, rng):
    return load_dataset(repo, split=split, revision=getattr(rng, "revision", None))

NONE = "None of the above"
# "None of the above" options must appear both as the correct answer and as a wrong alternative, with varied
# wording, or the model learns "this wording => pick it" (it did, in the first training run).
NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None)]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}

AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise", "contradiction": "The hypothesis contradicts the premise"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]


def _wrap_state(text, rng):
    r = rng.random()
    if r < 0.15: return {"document": text}
    if r < 0.25: return {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    if r < 0.32: return [{"role": "customer", "content": text}]
    return text


def _instr(text, rng):
    return {"question": text, "focus": rng.choice(["Use only the information given.", "Pick the single best fit.", "Consider the whole message."])} if rng.random() < 0.15 else text


def _desc(desc, rng, p_null=0.3, p_struct=0.1):
    r = rng.random()
    if r < p_null: return None
    if r < p_null + p_struct: return {"what": desc}
    return desc


def _sample(ds, n, rng):
    rows = []
    rng.origins = []
    for i in rng.sample(range(len(ds)), min(n, len(ds))):
        row = ds[i]
        if row.get("label", 0) == -1:
            continue
        text = row.get("text", row.get("premise", row.get("passage", json.dumps(row, sort_keys=True))))
        normalized = " ".join(text.casefold().split())
        rng.origins.append({"row": i, "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
        rows.append(row)
    return rows


def _banking(split, n, rng):
    ds = _dataset("legacy-datasets/banking77", split=split, rng=rng)
    names = ds.features["label"].names
    out = []
    for ex in _sample(ds, n, rng):
        t = rng.choice(BANK_TEMPLATES)
        crit = {k: _desc(t.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in names}
        out.append({"state": _wrap_state(ex["text"], rng), "questions": {"intent": {"type": "choice", "instructions": _instr("Which banking intent best describes this customer message?", rng), "criteria": crit, "label": names[ex["label"]], "src": "banking77"}}})
    return out


def _boolq(split, n, rng):
    ds = _dataset("google/boolq", split=split, rng=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = {"type": "noul", "instructions": _instr(ex["question"].strip().rstrip("?") + "?", rng), "label": bool(ex["answer"]), "src": "boolq"}
        if rng.random() < 0.4: q["criteria"] = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
        out.append({"state": _wrap_state(ex["passage"], rng), "questions": {"answer": q}})
    return out


def _agnews(split, n, rng):
    ds = _dataset("fancyzhx/ag_news", split=split, rng=rng)
    keys = list(AG)
    out = []
    for ex in _sample(ds, n, rng):
        y = keys[ex["label"]]
        qs = {"topic": {"type": "choice", "instructions": _instr("What is the topic of this article?", rng), "criteria": {k: _desc(v, rng) for k, v in AG.items()}, "label": y, "src": "agnews"}}
        for k in rng.sample(keys, 2):
            qs[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?", "label": k == y, "src": "agnews_yn"}
        out.append({"state": _wrap_state(ex["text"], rng), "questions": qs})
    return out


def _mnli(split, n, rng):
    ds = _dataset("nyu-mll/multi_nli", split=split, rng=rng)
    keys = list(MNLI)
    return [{"state": _wrap_state(ex["premise"], rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the premise?', rng), "criteria": {k: _desc(v, rng) for k, v in MNLI.items()}, "label": keys[ex["label"]], "src": "mnli"}}}
            for ex in _sample(ds, n, rng) if ex["label"] >= 0]


def _sst5(split, n, rng):
    ds = _dataset("SetFit/sst5", split=split, rng=rng)
    return [{"state": _wrap_state(ex["text"], rng), "questions": {"sentiment": {"type": "score", "instructions": _instr("What is the sentiment of this review sentence?", rng), "criteria": list(SST5), "label": ex["label"], "src": "sst5"}}} for ex in _sample(ds, n, rng)]


def _yelp(split, n, rng):
    ds = _dataset("Yelp/yelp_review_full", split=split, rng=rng)
    out = []
    for ex in _sample(ds, n, rng):
        text = " ".join(ex["text"].split()[:220])
        qs = {"rating": {"type": "score", "instructions": _instr("How many stars did this reviewer give?", rng), "criteria": list(YELP), "label": ex["label"], "src": "yelp"},
              "recommend": {"type": "noul", "instructions": "Would this reviewer recommend the business?", "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"}, "label": ex["label"] >= 3, "src": "yelp_yn"}}
        out.append({"state": _wrap_state(text, rng), "questions": qs})
    return out


SOURCES = {"banking77": (_banking, "train", "test"), "boolq": (_boolq, "train", "validation"), "agnews": (_agnews, "train", "test"),
           "mnli": (_mnli, "train", "validation_matched"), "sst5": (_sst5, "train", "test"), "yelp": (_yelp, "train", "test")}


def build(n_per_source, split="train", seed=0, exclude=(), only=(), revisions=None):
    unknown = (set(exclude) | set(only)) - SOURCES.keys()
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    reqs = []
    for name, (fn, tr, te) in SOURCES.items():
        if name in exclude or (only and name not in only): continue
        rng = random.Random(source_seed(seed, name))
        rng.revision = (revisions or {}).get(REPOS[name])
        source_split = tr if split == "train" else te
        records = fn(source_split, n_per_source, rng)
        if len(records) != len(rng.origins):
            raise ValueError(f"provenance mismatch for {name}")
        for record, origin in zip(records, rng.origins):
            record["_meta"] = {**origin, "source": name, "repo": REPOS[name], "revision": rng.revision,
                               "split": source_split, "id": f"{name}/{source_split}/{origin['row']}"}
        reqs.extend(records)
    random.Random(seed).shuffle(reqs)
    return reqs


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative (true option kept); sometimes add an irrelevant distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q; continue
        crit, y = dict(q["criteria"]), q["label"]
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options); crit.pop(y); crit[nk] = nd; y = nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            nk, nd = rng.choice(none_options); crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            k = rng.choice(distractors); crit[k] = DISTRACTORS[k]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out


def materialize(req):
    """Labelled request -> internal record via the serving path (api.to_record), attaching int labels and src."""
    clean = {"state": req["state"], "questions": {qid: {k: v for k, v in q.items() if k not in ("label", "src")} for qid, q in req["questions"].items()}}
    rec, meta = to_record(SystemOneRequest.model_validate(clean))
    for q, m, (qid, src_q) in zip(rec["questions"], meta, req["questions"].items()):
        y = src_q["label"]
        q["label"] = int(y) if m["type"] == "noul" else m["keys"].index(y) if m["type"] == "choice" else int(y)
        q["src"] = src_q["src"]; q["qtype"] = m["type"]
    return rec
