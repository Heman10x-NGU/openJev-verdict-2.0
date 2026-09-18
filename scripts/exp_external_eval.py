"""Experiment E9: TypeSafe Public Evaluation Cases Benchmark.

Evaluates the decision engine on the identical public evaluation subset from TypeSafe
(evals.typesafe.ai) across the 4 enterprise workflows (agent trace observability,
customer service, invoice processing, security incidents).
Compares accuracy against TypeSafe Jev and DiffusionGemma open-jev side-by-side.

Writes receipt to reports/v2/exp_e9_external_cases.json.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.formatting import build_model_input
from gliclass import GLiClassModel


def canonical(val: Any) -> str:
    if val is True:
        return "true"
    if val is False:
        return "false"
    return str(val).lower() if isinstance(val, str) else str(val)


def winner(probabilities: dict[str, float] | None) -> str | None:
    if not probabilities:
        return None
    maximum = max(probabilities.values())
    winners = [k for k, v in probabilities.items() if abs(v - maximum) < 1e-9]
    return winners[0] if len(winners) == 1 else None


def load_typesafe_cases(cases_dir: Path) -> list[dict[str, Any]]:
    items = []

    for path in sorted(cases_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for case_id, case in data["cases"].items():
            jev = case["models"]["typesafe"]
            for node in jev["nodes"]:
                if not node["ran"]:
                    continue
                for qid, index in node["questions"].items():
                    question = data["questions"][index]
                    typ = question["type"]

                    # Consensus reference
                    ref = case.get("reference_answers", {}).get(node["node"], {}).get(qid)
                    consensus: dict[str, float] = collections.defaultdict(float)
                    if ref and "sets" in ref:
                        for item in ref["sets"]:
                            probs = item.get("probabilities") or {canonical(item["value"]): 1.0}
                            for k, v in probs.items():
                                consensus[canonical(k)] += v / len(ref["sets"])
                    target = winner(consensus)

                    # Jev answer
                    ans = node["answers"][qid]
                    if typ == "noul":
                        j_probs = {"false": 1.0 - ans["noul"], "true": ans["noul"]}
                        j_reported = (ans["noul"] >= 0.5)
                    else:
                        j_probs = {canonical(k): v for k, v in ans.get("probabilities", {}).items()} if ans.get("probabilities") else None
                        j_reported = ans.get("choice") if typ == "choice" else ans.get("score")

                    j_val = winner(j_probs)
                    if j_val is None and not j_probs and j_reported is not None:
                        j_val = canonical(j_reported)

                    # Keep only questions with valid target and Jev answer (the 337 public scored subset)
                    if target is None or j_val is None:
                        continue

                    if typ == "noul":
                        options = ["false", "true"]
                        descriptions = [
                            f"Proposition is false: {question['instructions']}",
                            f"Proposition is true: {question['instructions']}",
                        ]
                    elif typ == "score":
                        options = [str(i) for i in range(len(question["criteria"]))]
                        descriptions = [
                            f"Score {i}: {crit}" for i, crit in enumerate(question["criteria"])
                        ]
                    else:
                        options = [str(k) for k in question["criteria"]]
                        descriptions = [
                            f"{k}: {crit}" for k, crit in question["criteria"].items()
                        ]

                    identity = "/".join((path.stem, case_id, node["node"], qid))
                    document = json.dumps(data["documents"][node["doc"]], ensure_ascii=False)
                    q_text = question.get("instructions", "Select the appropriate evaluation option:")

                    items.append(
                        {
                            "id": identity,
                            "workflow": path.stem,
                            "case_id": case_id,
                            "node": node["node"],
                            "qid": qid,
                            "type": typ,
                            "question": q_text,
                            "context": document,
                            "options": [canonical(o) for o in options],
                            "descriptions": descriptions,
                            "target": target,
                            "jev": j_val,
                        }
                    )

    return items


def run_external_eval(
    cases_dir: str = "oss/open-jev/typesafe-public-evals",
    checkpoint_dir: str = "artifacts/v2",
    reports_dir: str = "reports/v2",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    device_name: str = "mps",
) -> dict[str, Any]:
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    c_dir = Path(cases_dir)
    if not c_dir.exists():
        raise FileNotFoundError(f"Cases directory {c_dir} not found.")

    if device_name == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    items = load_typesafe_cases(c_dir)
    print(f"Loaded {len(items)} scored question instances from {cases_dir}.")

    # Load model
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint_candidates = [
        Path(checkpoint_dir) / "model.safetensors",
        Path(checkpoint_dir) / "openjev_modernbert.safetensors",
    ]
    ckpt_path = next((cp for cp in checkpoint_candidates if cp.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}")

    state_dict = load_file(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    by_workflow: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"scored": 0, "ours_correct": 0, "jev_correct": 0})
    by_type: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"scored": 0, "ours_correct": 0, "jev_correct": 0})
    scored_count = len(items)
    ours_total_correct = 0
    jev_total_correct = 0

    with torch.no_grad():
        for item in items:
            target = item["target"]
            j_val = item["jev"]

            prompt = build_model_input(item["question"], item["context"][:1200], item["descriptions"])
            tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**tokens)

            k = len(item["descriptions"])
            c_logits = outputs.logits[0, :k].float().cpu()
            pred_idx = int(torch.argmax(c_logits).item())
            predicted_option = item["options"][pred_idx]

            is_ours_correct = bool(predicted_option == target)
            is_jev_correct = bool(j_val == target)

            wf = item["workflow"]
            typ = item["type"]

            by_workflow[wf]["scored"] += 1
            by_workflow[wf]["ours_correct"] += int(is_ours_correct)
            by_workflow[wf]["jev_correct"] += int(is_jev_correct)

            by_type[typ]["scored"] += 1
            by_type[typ]["ours_correct"] += int(is_ours_correct)
            by_type[typ]["jev_correct"] += int(is_jev_correct)

            ours_total_correct += int(is_ours_correct)
            jev_total_correct += int(is_jev_correct)

    wf_summary = {}
    for wf, stats in by_workflow.items():
        n = stats["scored"]
        wf_summary[wf] = {
            "scored": n,
            "ours_accuracy": stats["ours_correct"] / n if n > 0 else 0.0,
            "ours_correct": stats["ours_correct"],
            "jev_accuracy": stats["jev_correct"] / n if n > 0 else 0.0,
            "jev_correct": stats["jev_correct"],
        }

    type_summary = {}
    for typ, stats in by_type.items():
        n = stats["scored"]
        type_summary[typ] = {
            "scored": n,
            "ours_accuracy": stats["ours_correct"] / n if n > 0 else 0.0,
            "ours_correct": stats["ours_correct"],
            "jev_accuracy": stats["jev_correct"] / n if n > 0 else 0.0,
            "jev_correct": stats["jev_correct"],
        }

    receipt = {
        "experiment": "E9_typesafe_public_evals",
        "benchmark_source": "evals.typesafe.ai (public subsets)",
        "total_scored_questions": scored_count,
        "overall": {
            "ours_correct": ours_total_correct,
            "ours_accuracy": ours_total_correct / scored_count if scored_count > 0 else 0.0,
            "openjev_diffusiongemma_accuracy": 298 / 337,  # 88.43%
            "typesafe_jev_accuracy": jev_total_correct / scored_count if scored_count > 0 else 0.0,
        },
        "by_workflow": wf_summary,
        "by_type": type_summary,
    }

    out_file = rep_path / "exp_e9_external_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E9 receipt to {out_file}")
    print(f"Scored Questions: {scored_count}")
    print(f"Verdict-open-jev Accuracy: {receipt['overall']['ours_accuracy']*100:.2f}% ({ours_total_correct}/{scored_count})")
    print(f"Open-Jev (DiffusionGemma): {receipt['overall']['openjev_diffusiongemma_accuracy']*100:.2f}% (298/337)")
    print(f"TypeSafe Jev:              {receipt['overall']['typesafe_jev_accuracy']*100:.2f}% ({jev_total_correct}/{scored_count})")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E9 External TypeSafe Public Evals.")
    parser.add_argument("--cases_dir", type=str, default="oss/open-jev/typesafe-public-evals")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_external_eval(
        cases_dir=args.cases_dir,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
