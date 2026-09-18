import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from kev.data import REPOS, SOURCES, build, materialize, source_seed
from kev.model import encode, load_tokenizer

SPLITS = ("train", "calibration", "development", "test")
BASES = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test; never use it for search")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.jsonl"
    if digest(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def case_copy(record, variant):
    result = copy.deepcopy(record)
    result["_meta"]["group_id"] = record["_meta"]["id"]
    result["_meta"]["id"] += "/" + variant
    result["_meta"]["variant"] = variant
    return result


def contrast_cases(record, seed=0):
    candidates = [(qid, q) for qid, q in record["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not candidates:
        return []
    qid, q = candidates[0]
    nk = "none_of_these"
    if nk in q["criteria"]:
        raise ValueError("reserved contrast option collision")
    out = []
    for variant in ("none_present", "none_absent"):
        r = case_copy(record, variant)
        r["questions"] = {qid: copy.deepcopy(q)}
        rq = r["questions"][qid]
        rq["criteria"][nk] = "None of these options describes the answer"
        if variant == "none_absent":
            rq["criteria"].pop(rq["label"])
            rq["label"] = nk
        keys = list(rq["criteria"])
        random.Random(source_seed(seed, record["_meta"]["id"])).shuffle(keys)
        rq["criteria"] = {k: rq["criteria"][k] for k in keys}
        r["_meta"]["none_key"] = nk
        out.append(r)
    r = case_copy(record, "permuted")
    rng = random.Random(source_seed(seed, record["_meta"]["id"]))
    for q in r["questions"].values():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            rng.shuffle(keys)
            q["criteria"] = {k: q["criteria"][k] for k in keys}
    out.append(r)
    return out


def select_unique(records, count, seen, tokenizers, report):
    selected = []
    for record in sorted(records, key=lambda r: r["_meta"]["row_sha256"]):
        report["considered"] += 1
        key = record["_meta"]["text_sha256"]
        if key in seen:
            report["duplicate_state"] += 1
            continue
        try:
            rec = materialize(record)
            for tokenizer in tokenizers:
                enc = encode(tokenizer, rec, max_branch=960, strict=True)
                if len(enc["ids"]) > 2048:
                    raise ValueError("packed request exceeds 2048 tokens")
        except ValueError:
            report["context_rejected"] += 1
            continue
        record["_meta"].update(group_id=record["_meta"]["id"], variant="clean")
        seen.add(key)
        selected.append(record)
        report["accepted"] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)}/{count} records fit the common context policy")


def freeze(directory, train=300, calibration=40, development=80, test=80, seed=20260918, holdout=("mnli", "sst5")):
    from huggingface_hub import HfApi

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite frozen suite {directory}")
    hub = HfApi()
    revisions = {repo: hub.dataset_info(repo).sha for repo in REPOS.values()}
    base_revisions = {base: hub.model_info(base).sha for base in BASES}
    tokenizers = [load_tokenizer(base, revision=revision) for base, revision in base_revisions.items()]
    manifest = {
        "version": 1, "seed": seed, "holdout_sources": list(holdout),
        "dataset_revisions": revisions, "base_revisions": base_revisions,
        "context": {"max_state": 384, "max_branch": 1024, "max_packed": 2048, "truncate": False},
        "selection": "Normalized exact-state deduplication across partitions; common tokenizer context admission; no fuzzy decontamination or pretraining-contamination claim.",
        "legacy_checkpoints": "Training/calibration overlap for pre-manifest checkpoints is unknown; exploratory only.",
        "objective": "Negative macro-average clean development NLL, equal weight per task; raw probabilities.",
        "files": {}, "admission": {},
    }
    partitions = {split: [] for split in SPLITS}
    seen = set()
    for source in SOURCES:
        report = Counter()
        train_pool = build(max(3 * (train + calibration), 800), "train", seed, only=[source], revisions=revisions)
        test_pool = build(max(3 * (development + test), 600), "test", seed, only=[source], revisions=revisions)
        chosen = select_unique(test_pool, development + test, seen, tokenizers, report)
        partitions["development"].extend(chosen[:development])
        partitions["test"].extend(chosen[development:])
        if source not in holdout:
            chosen = select_unique(train_pool, train + calibration, seen, tokenizers, report)
            partitions["calibration"].extend(chosen[:calibration])
            partitions["train"].extend(chosen[calibration:])
        manifest["admission"][source] = dict(report)
        print(f"froze {source}: {dict(report)}", flush=True)
    for split in ("development", "test"):
        extras = []
        per_source = Counter()
        for record in partitions[split]:
            source = record["_meta"]["source"]
            if per_source[source] < 12:
                variants = contrast_cases(record, seed)
                for variant in variants:
                    for tok in tokenizers:
                        encode(tok, materialize(variant), strict=True)
                extras.extend(variants)
                per_source[source] += bool(variants)
        partitions[split].extend(extras)
    directory.mkdir(parents=True)
    for split, records in partitions.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records),
                                        "questions": sum(len(r["questions"]) for r in records)}
    manifest["code_hashes"] = {name: digest(Path(__file__).parent / name) for name in ("data.py", "api.py", "model.py", "suite.py")}
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest["files"], indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--calibration", type=int, default=40)
    ap.add_argument("--development", type=int, default=80)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()
    if min(a.train, a.calibration, a.development, a.test) < 1:
        ap.error("all split sizes must be positive")
    freeze(a.out, a.train, a.calibration, a.development, a.test, a.seed)


if __name__ == "__main__":
    main()
