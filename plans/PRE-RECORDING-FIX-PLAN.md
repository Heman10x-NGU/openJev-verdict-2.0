# Pre-recording fix plan

Four defects visible on camera. All are in `webgpu-demo/` plus one artifact file.
Scope is deliberately small: nothing here changes a benchmark number or a
published claim.

---

## D1. Stale timing survives preset changes

`loadPreset()` at `webgpu-demo/index.html:976` never resets `statusText` or
`latencyBadge`. After one inference, switching presets leaves the previous
timing on screen while the policy panel reads STANDBY. Three of the four
screenshots show an identical "Evaluated in 134.6ms (133.0ms GPU forward pass)"
with no inference having run for that preset.

On video this reads as a fabricated number.

**Fix:** in `loadPreset()`, and in the candidate add/remove/shuffle handlers,
reset `latencyBadge.textContent` to `"⚡ Ready"` and `statusText.textContent` to
the idle string. Clear any rendered probabilities at the same time, so a preset
switch never shows stale bars beside a stale time.

## D2. Single-sample timing is noisy

`runInference` in `webgpu-demo/worker.js` measures one `performance.now()` delta
per click (lines 263 to 315). WebGPU queue scheduling and readback make a single
sample swing widely, which is why the same build reports 90ms, 134.6ms, and
192.8ms.

**Fix:** run the forward pass `N = 3` times per click inside the worker, keep the
median, and return `p50_ms`, `min_ms`, and `max_ms` in `timings_ms`. Three passes
at roughly 100ms each still feels instant. Only the forward pass repeats;
tokenization and post-processing stay single-shot.

Add `runs: N` to the receipt so the downloaded JSON says how the number was
produced.

## D3. Warmup does not cover real input shapes

`initEngine` warms up once with a synthetic one-label prompt
(`webgpu-demo/worker.js:240`). WebGPU compiles shader variants per tensor shape,
so the first real query at K=4 and roughly 99 tokens still pays compilation. That
is the most likely cause of the 192.8ms outlier on the eclipse preset, which has
a different token count (101) from the others (99).

**Fix:** after the existing warmup, run two more warmups at representative shapes,
K=4 and K=8, with roughly 100-token contexts, and discard the results. Report
readiness only after all warmups complete.

## D4. Temperature 1.43 is applied and shown on screen

`artifacts/v2/calibrator.json` carries `temperature: 1.4265148639678955`, the
header renders `T: 1.43`, and `worker.js` divides logits by it on every run.

`reports/v2/evaluation_report_v2.json` shows uncalibrated is better on every
metric: NLL 0.1731 against 0.1768, Brier 0.0756 against 0.0785, ECE equal-width
1.13% against 3.35%, ECE adaptive 0.83% against 2.87%.

**Fix:** set the shipped `artifacts/v2/calibrator.json` temperature to 1.0 and
note in the file that the fitted value is retained in the evaluation report as a
measured result. Do not delete the fitted value from the report. Do not change
any published metric.

Side effect, which is wanted: confidences get sharper on screen.

---

## Labelling

The status line currently reads:

```
Evaluated in 134.6ms (133.0ms GPU forward pass)
```

This does not say which backend, which precision, or how many samples. Replace
with a form that carries all three, for example:

```
Forward pass 128.4ms median of 3 (WebGPU, fp16) · 121.0 to 139.7ms
```

When the WASM fallback is active, the label must say WASM and fp32.

---

## Optional, only if time allows

`reports/v2/exp_e7_latency.json` measures fp16 at 99.77ms against fp32 at
35.58ms on CPU, because ONNX Runtime CPU has no native fp16 kernels. On WebGPU
the ordering may reverse. If both files are present, time the two in the browser
and load whichever is faster on the active backend. Report the measurement; do
not assume the answer.

---

## Out of scope

Do not touch: any file under `reports/`, `scripts/`, `core/`, or `tests/`; any
benchmark number; the README; the preset text. This plan changes display
correctness and one shipped constant, nothing else.

## Acceptance

1. Switching presets clears the timing, the badge, and the probability bars.
2. Three consecutive runs of the same preset report medians within a visibly
   tight band rather than a 90 to 192ms spread.
3. The first run after load is not an outlier against the next five.
4. The status line names backend, precision, sample count, and range.
5. The header reads `T: 1.00`.
6. `pytest tests/ -q` still passes and both render `--check` scripts still pass.
