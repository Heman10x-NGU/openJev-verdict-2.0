import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from kev.benchmark import api_request, evaluate_records
from kev.suite import digest, load_split, write_json

PRICE_PER_MILLION = 0.042


def provision_key(scope):
    result = subprocess.run(["vercel", "ai-gateway", "api-keys", "create", "--scope", scope,
                             "--name", "kev-jev-evaluation", "--limit", "1", "--refresh-period", "none",
                             "--expiration", "7d", "--no-color", "--non-interactive"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Vercel key creation failed; inspect account permissions in the Vercel dashboard")
    text = result.stdout + result.stderr
    match = re.search(r"\bvck_[A-Za-z0-9_-]+", text)
    if not match:
        raise RuntimeError("Key was created but its output format was not recognized; no key output was logged")
    return match.group()


class JevPredictor:
    def __init__(self, key, budget=0.1, max_calls=700):
        self.budget, self.max_calls = budget, max_calls
        self.calls, self.input_tokens, self.output_tokens = 0, 0, 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        worker = Path(__file__).resolve().parents[1] / "playground/scripts/jev-evaluate.mjs"
        self.process = subprocess.Popen(["node", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                        env={**os.environ, "AI_GATEWAY_API_KEY": key})

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __call__(self, record):
        if self.calls >= self.max_calls or (self.input_tokens + 65536) * PRICE_PER_MILLION / 1e6 > self.budget:
            raise RuntimeError("Jev evaluation reached the request/token cost cap")
        request = api_request(record)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Jev SDK worker exited without a response")
        result = json.loads(line)
        self.calls += 1
        if "error" in result:
            raise RuntimeError(f"Jev request failed: {result['error']['name']} (HTTP {result['error']['status']})")
        usage = result["usage"]
        if usage.get("inputTokens") is None:
            raise RuntimeError("Jev returned no input token usage; cannot account for cost")
        self.input_tokens += usage["inputTokens"]
        self.output_tokens += usage.get("outputTokens") or 0
        probabilities = {}
        for qid, q in record["questions"].items():
            answer = result["answers"][qid]
            if q["type"] == "noul":
                probabilities[qid] = {"false": 1 - answer["probability"], "true": answer["probability"]}
            else:
                probabilities[qid] = answer["probabilities"]
        return {**result, "probabilities": probabilities}

    def accounting(self):
        return {"model": "typesafe-ai/jev", "model_revision": "Gateway alias; provider revision not exposed by SDK result",
                "started_at": self.started_at, "calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "listed_input_usd_per_million": PRICE_PER_MILLION,
                "estimated_usd": self.input_tokens * PRICE_PER_MILLION / 1e6,
                "budget_usd": self.budget, "sdk": "ai@7.0.105", "zero_data_retention_requested": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provision-scope", help="Explicitly authorize a $1 non-renewing, seven-day key on this team")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--max-calls", type=int, default=700)
    a = ap.parse_args()
    if not 0 < a.budget <= 1 or not 1 <= a.max_calls <= 2000:
        ap.error("budget must be in (0, 1] and max-calls in [1, 2000]")
    if Path(a.out).exists():
        ap.error("output directory already exists")
    records = load_split(a.suite, "development")
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key and a.provision_scope:
        key = provision_key(a.provision_scope)
    if not key:
        ap.error("Set AI_GATEWAY_API_KEY or explicitly select --provision-scope")
    predictor = JevPredictor(key, a.budget, a.max_calls)
    try:
        report, _ = evaluate_records(records, predictor, a.out)
        report.update(suite_sha256=digest(Path(a.suite) / "manifest.json"), split="development", provider=predictor.accounting())
        write_json(Path(a.out) / "report.json", report)
        print(json.dumps({"clean": report["clean"], "variants": report["variants"], "provider": report["provider"]}, indent=2))
    finally:
        predictor.close()
        if Path(a.out).exists():
            write_json(Path(a.out) / "usage.json", predictor.accounting())


if __name__ == "__main__":
    main()
