"""Publish a trained run to the Hugging Face Hub.

Repo naming: jaredpalmer/kev-<base size>, e.g. kev-0.5b (Qwen2.5-0.5B), kev-0.6b (Qwen3-0.6B), kev-4b, kev-8b.
The exact base checkpoint is recorded in the model card's `base_model` field and in head.pt.

    uv run python -m kev.publish --run runs/kev --repo jaredpalmer/kev-0.5b
    uv run python -m kev.publish --run runs/kev2 --repo jaredpalmer/kev-0.5b --message "v0.2: none-of-the-above fix"

Uploads: adapter, head.pt, tokenizer files, eval.json, training log (if found), and MODEL_CARD.md as README.md
with the repo id and run name filled in. Requires `hf auth login`.
"""
import argparse, json, os, re, shutil, tempfile
import torch
from huggingface_hub import HfApi

FILES = ["adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "eval.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True, help="e.g. jaredpalmer/kev-0.5b")
    ap.add_argument("--message", default=None)
    ap.add_argument("--card", default="MODEL_CARD.md")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    meta = torch.load(f"{a.run}/head.pt", map_location="cpu")
    base, run_name = meta["base"], os.path.basename(a.run.rstrip("/"))
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)

    with tempfile.TemporaryDirectory() as tmp:
        for f in FILES:
            src = f"{a.run}/{f}"
            if os.path.exists(src): shutil.copy(src, tmp)
            else: print(f"skip {f} (not found)")
        # runs before task_type was set saved null; the Hub warns about it and PEFT treats both the same for a bare backbone
        cfg_path = f"{tmp}/adapter_config.json"; cfg = json.load(open(cfg_path))
        if not cfg.get("task_type"): cfg["task_type"] = "FEATURE_EXTRACTION"; json.dump(cfg, open(cfg_path, "w"), indent=2)
        for log in (f"runs/logs/train_{run_name}.log", f"runs/train_{run_name}.log", f"runs/train.log" if run_name == "kev" else ""):
            if log and os.path.exists(log): shutil.copy(log, f"{tmp}/train.log"); break

        card = open(a.card).read()
        card = re.sub(r"^base_model: .*$", f"base_model: {base}", card, flags=re.M)
        if "base_model_relation:" not in card: card = card.replace(f"base_model: {base}", f"base_model: {base}\nbase_model_relation: adapter")
        card = card.replace("- Code, training recipe, evaluation and demo:", f"- Hub: [{a.repo}](https://huggingface.co/{a.repo}) (this repo, run `{run_name}`)\n- Code, training recipe, evaluation and demo:")
        open(f"{tmp}/README.md", "w").write(card)

        ev = json.load(open(f"{tmp}/eval.json")) if os.path.exists(f"{tmp}/eval.json") else {}
        acc = ev.get("accuracy_calibration", {}).get("ALL", {})
        msg = a.message or f"Upload {run_name} (base {base}; acc {acc.get('acc', float('nan')):.3f}, ECE {acc.get('ece', float('nan')):.3f})"
        url = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model", commit_message=msg)
        print(url)


if __name__ == "__main__":
    main()
