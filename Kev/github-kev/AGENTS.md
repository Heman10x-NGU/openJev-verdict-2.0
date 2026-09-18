# kev — prototype of a Jev-style decision model

Causal LM (Qwen2.5-0.5B + LoRA) run prefill-only with a block-causal mask (shared state prefix,
isolated question branches) and a pointer readout over option boundary tokens, trained with log loss
on converted public datasets (Banking77, BoolQ, AG News, MNLI, SST-5, Yelp). No text generation.
See README.md (deep dive) and MODEL_CARD.md (checkpoint recipe + metrics). README follows the Vercel Labs house style (tagline, for-the-badge badges, Highlights, Title Case sections, API tables, Authors + License); MODEL_CARD.md is formal.

## Commands
- Env: `uv sync` (torch MPS, transformers, peft, datasets)
- Train: `uv run python -m kev.train --n_per_source 1500 --epochs 2 --out runs/kev` (~1h45m on M5 32GB)
  - `--holdout mnli,sst5` excludes sources (out-of-source eval); `--perm_kl/--perm_frac` permutation-consistency KL;
    `--ord_w` ordinal term for Score. Only one training process at a time: two on MPS slow each other ~10x.
- Eval:  `uv run python -m kev.evaluate --run runs/kev --n_per_source 150 --baseline --baseline_instruct Qwen/Qwen2.5-0.5B-Instruct`
  -> `runs/kev/eval.json` (acc/ECE/NLL per source, temperature scaling, permutation, IIA, isolation, packed-vs-separate, held-out sources)
- Smoke: `--n_per_source 40 --accum 4 --out runs/smoke` (~1 min)
- Research suite: `evals/decision-v1` (frozen, checksummed; train/calibration/development/test; manifest pins dataset + base
  revisions). `kev.suite` freezes; `kev.benchmark --run X --suite evals/decision-v1 --out runs/...` scores development;
  `--allow-test` is the only way to read the locked test. `kev.experiment --plan experiments/*.json` runs config-only trials
  (bounded allowlist, provenance, coverage/isolation gates, results.jsonl ledger, `--wait-pid` to queue behind a training job).
  `kev.jev` scores Jev via Vercel AI Gateway (AI SDK 7 `experimental_evaluate`, node worker in `playground/scripts/`;
  needs `AI_GATEWAY_API_KEY` or `--provision-scope`; budget-capped). `kev.compare` pairs two result dirs (record-clustered
  bootstrap). Historical checkpoints (`runs/kev`, `runs/kev2`) overlap the suite's training data: exploratory only.
  `--ord_w` is now the ranked probability score (proper); the old |E[level]-y| term was removed. `--perm_kl`/`--ord_w` default 0.
- Plot (README figure): `uv run python -m kev.plot --logs runs/logs/train_kev.log:kev-0.5b --eval runs/kev/eval.json --out docs/training.png`
  Training logs to keep go in `runs/logs/train_*.log` (only path under runs/ besides eval.json that is committed).
- Publish: `uv run python -m kev.publish --run runs/<run> --repo jaredpalmer/kev-<size>` (needs `hf auth login`). Repos are named by
  base model size (kev-0.5b = Qwen2.5-0.5B); versions within a size are Hub tags (`hf repos tag create jaredpalmer/kev-0.5b vX.Y`).
  Collection: huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd. `--run` in serve/evaluate accepts a Hub id.
- Serve: `uv run --extra serve python -m kev.serve --run runs/kev --port 8008` (falls back to runs/smoke)
  - TypeSafe-compatible: `POST /v1/systemone`, `GET /v1/models` (no auth). Playground routes under `/api/*`.
  - SDK: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`
- Extra endpoints for the demo: `POST /v1/systemone/permute` (one Choice under n option orders), `POST /v1/systemone/separate`
  (each question alone; packed-vs-separate comparison). `/v1/systemone` also returns `latency_ms`.
- Web demo: `cd playground && npm run dev -- -p 3001` (:3000 is used by another project). Next 16 app router; `/kev/*` is
  rewritten to the FastAPI server (`KEV_API`, default http://127.0.0.1:8009). Presets live in `playground/src/lib/kev.ts`.
  - `/chess` (`src/components/chess-game.tsx`, `src/lib/chess.ts`, chess.js): legal moves -> Choice options, board -> state, Score for eval;
    games in localStorage key `kev.chess.v1`. Python twin: `examples/chess_selfplay.py` (typesafe-sdk + python-chess, `--extra examples`).
  - React Compiler lint forbids sync setState in effects; schedule via setTimeout or move into handlers.
  - Next 16 dev only trusts `localhost`; other hostnames need `allowedDevOrigins` or the page SSRs but never hydrates
    (no console errors). `127.0.0.1` is allowed in `next.config.ts`. Verify hydration with `agent-browser` (CDP), not curl.
- Unit tests (no weights, CI): `uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py -q`
- API tests (server must be up): `KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q`

## Layout
- `kev/data.py`      dataset -> typed records, permutation / none-of-the-above / distractor augmentation
- `kev/model.py`     encode(), branch_mask(), PointerHead, DecisionModel
- `kev/train.py`     LoRA fine-tune, batch size 1 with grad accumulation (variable-length custom masks)
- `kev/evaluate.py`  acc/ECE, permutation stability, IIA shift, isolation probe, packed-vs-separate
- `kev/plot.py`      loss curve(s) from train logs + accuracy-vs-baselines bars from eval.json
- `kev/api.py`       TypeSafe request/response models; Noul/Choice/Score -> pointer options; confidence formulas
- `kev/serve.py`     FastAPI: /v1/systemone (+ /v1/models) and /api/* playground routes
- `tests/test_api.py` conformance against the docs' example requests + official SDK

## Notes
- Delimiters reuse existing Qwen special tokens (`<|fim_prefix|>` etc.) to avoid resizing embeddings;
  peft `trainable_token_indices` leaked memory on MPS.
- `output_hidden_states=True` on MPS blows memory; use the bare `.model` backbone's `last_hidden_state`.
- `PolyAI/banking77` uses a loading script (unsupported); use `legacy-datasets/banking77`.
- User text is tokenized via `model.user_tokens()`, which rewrites `<|name|>` -> `<¦name¦>` so callers cannot forge
  option/branch delimiter tokens (the fast tokenizer ignores `split_special_tokens`).
- Training data is built as TypeSafe-shaped requests and goes through `api.to_record()` (`data.materialize`),
  so train and serve text are identical.
