import argparse, json, math, os, random, resource, sys, time
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from .data import build, augment, materialize, source_seed
from .suite import digest, load_split, write_json
from .model import DecisionModel, load_tokenizer, encode


def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            out["questions"].append({**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w):
    """Cross-entropy, optionally plus the normalized ranked probability score for ordered levels."""
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n_per_source", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    ap.add_argument("--perm_kl", type=float, default=0.0, help="weight of symmetric KL between predictions under two option orders")
    ap.add_argument("--perm_frac", type=float, default=0.3, help="fraction of records that get the second permuted forward pass")
    ap.add_argument("--ord_w", type=float, default=0.0, help="weight of ranked probability score for Score questions")
    ap.add_argument("--suite", help="frozen suite directory; train only on its training partition")
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15)
    ap.add_argument("--out", default="runs/kev")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if min(a.epochs, a.accum, a.n_per_source, a.lora) < 1:
        ap.error("epochs, accum, n_per_source and lora must be positive")
    if a.lr <= 0 or min(a.ord_w, a.perm_kl) < 0 or not 0 <= a.perm_frac <= 1:
        ap.error("invalid learning rate or loss weights")
    out_dir = Path(a.out)
    if out_dir.exists():
        ap.error("refusing to overwrite an existing run")
    out_dir.mkdir(parents=True)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    manifest = json.loads((Path(a.suite) / "manifest.json").read_text()) if a.suite else None
    revision = manifest["base_revisions"][a.base] if manifest else None
    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision)
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]
    reqs = load_split(a.suite, "train") if manifest else build(a.n_per_source, "train", a.seed, exclude=holdout)
    if not reqs:
        raise ValueError("empty training set")
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision,
                                                "ordinal_objective": "ranked_probability_score", "holdout": holdout})
    print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
          f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(reqs) / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    model.train(); t0 = time.time(); run = Counter(); step = 0; seen = 0
    tokens_seen = peak_mps = 0
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        for i, req in enumerate(reqs):
            item_rng = random.Random(source_seed(a.seed, f"{ep}:{req['_meta']['id']}"))
            rec = materialize(augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract))  # fresh permutation / distractors each epoch
            enc = encode(tok, rec, strict=True)
            if len(enc["ids"]) > 2048:
                raise ValueError("training request exceeds 2048 packed tokens")
            tokens_seen += len(enc["ids"])
            logits = model(enc)
            loss = sum(question_loss(z, q, dev, a.ord_w) for z, q in zip(logits, rec["questions"])) / len(logits)
            run["ce"] += loss.item()
            if a.perm_kl > 0 and item_rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
                rec2, perms = permuted_copy(rec, item_rng)
                logits2 = model(encode(tok, rec2, strict=True)); kl, n = 0.0, 0
                tokens_seen += len(enc["ids"])
                for z1, z2, perm in zip(logits, logits2, perms):
                    if perm is None: continue
                    lp1 = F.log_softmax(z1, -1); lp2 = F.log_softmax(z2, -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
                    kl = kl + 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum")); n += 1
                kl = kl / n; loss = loss + a.perm_kl * kl; run["kl"] += kl.item(); run["kl_n"] += 1
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            group_size = min(a.accum, len(reqs) - (i // a.accum) * a.accum)
            if dev == "mps":
                peak_mps = max(peak_mps, torch.mps.current_allocated_memory())
            (loss / group_size).backward(); run["n"] += 1; seen += 1
            if (i + 1) % a.accum == 0 or i + 1 == len(reqs):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if dev == "mps": torch.mps.empty_cache()
                if step % 10 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {run['ce']/run['n']:.3f} kl {run['kl']/max(run['kl_n'],1):.3f} {(time.time()-t0)/seen:.2f}s/rec", flush=True)
                    run = Counter()
    os.makedirs(a.out, exist_ok=True)
    model.lm.save_pretrained(a.out)
    torch.save({"head": model.head.state_dict(), "base": a.base, "base_revision": revision, "lora": a.lora,
                "holdout": holdout, "args": vars(a), "suite_sha256": suite_hash}, f"{a.out}/head.pt")
    tok.save_pretrained(a.out)
    write_json(out_dir / "training_metrics.json", {"wall_seconds": time.time() - t0, "records_seen": seen,
               "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
               "optimizer_steps": step, "forward_tokens": tokens_seen,
               "peak_observed_mps_bytes": peak_mps,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
