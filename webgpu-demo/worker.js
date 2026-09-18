/**
 * WebWorker executing genuine in-browser Verdict-ModernBERT decision inference.
 * Powered by ONNX Runtime Web (WebGPU / WASM) and @huggingface/transformers.
 */

let ort = null;
let AutoTokenizer = null;
let env = null;

let depsPromise = null;
let depsLoaded = false;

async function ensureDependencies() {
  if (depsLoaded) return;
  if (depsPromise) return depsPromise;

  depsPromise = (async () => {
    // 1. Load ONNX Runtime Web (try local node_modules, fallback to CDN)
    try {
      ort = await import("./node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs");
    } catch (e1) {
      console.warn("Local onnxruntime-web import failed, falling back to CDN:", e1);
      ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.bundle.min.mjs");
    }

    // Configure ORT WASM paths explicitly (RC-2)
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

    // 2. Load Transformers.js (RC-1: try local node_modules first, fallback to unminified CDN transformers.js)
    try {
      const mod = await import("./node_modules/@huggingface/transformers/dist/transformers.js");
      AutoTokenizer = mod.AutoTokenizer;
      env = mod.env;
    } catch (e2) {
      console.warn("Local @huggingface/transformers import failed, falling back to CDN:", e2);
      const mod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.js");
      AutoTokenizer = mod.AutoTokenizer;
      env = mod.env;
    }

    // Configure Transformers.js environment flags properly (RC-1)
    env.allowLocalModels = true;
    env.allowRemoteModels = true;
    env.localModelPath = "./";

    depsLoaded = true;
  })();

  return depsPromise;
}

// Start loading dependencies immediately
ensureDependencies().catch(err => {
  console.warn("Dependency pre-fetch notice:", err);
});

const CACHE_NAME = "verdict-model-cache-v2";

let session = null;
let tokenizer = null;
let promptContract = null;
let activeBackend = "WASM";
let activeModelFile = "";
let activePrecision = "fp32";
let calibratorTemp = null;
let modelSizeBytes = 0;
let inferenceQueue = Promise.resolve();

async function fetchWithCacheAndProgress(url, onProgress) {
  let cache = null;
  if (typeof caches !== "undefined") {
    try {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        const buf = await cached.arrayBuffer();
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return buf;
      }
    } catch (e) {
      console.warn("Cache API lookup failed, fetching from network:", e);
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  }

  const contentLengthHeader = res.headers.get("Content-Length");
  const totalBytes = contentLengthHeader ? parseInt(contentLengthHeader, 10) : 0;
  let receivedBytes = 0;

  const reader = res.body.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;
    if (onProgress) onProgress(receivedBytes, totalBytes);
  }

  const combined = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  if (cache) {
    try {
      await cache.put(url, new Response(combined.buffer, {
        headers: { "Content-Type": "application/octet-stream" }
      }));
    } catch (e) {
      console.warn("Failed to store model in Cache API:", e);
    }
  }

  return combined.buffer;
}

// Synthetic warm-up context sized to match the real presets (roughly 100 tokens end to end)
const WARMUP_CONTEXT = "The customer wrote in about a recent event on their account and asked what happens next. " +
  "They mention a date, a reference number, and an earlier reply from the support team. " +
  "The note is ordinary English and about as long as a typical support ticket body.";

function buildWarmupPrompt(candidateCount) {
  const labels = [];
  for (let i = 1; i < candidateCount; i++) {
    labels.push(`${promptContract.label_marker}candidate option number ${i}`);
  }
  labels.push(`${promptContract.label_marker}${promptContract.abstention_description}`);

  const formattedText = promptContract.input_template
    .replace("{question}", promptContract.default_question)
    .replace("{context}", WARMUP_CONTEXT);

  return `${labels.join("")}${promptContract.sep_marker}${formattedText}`;
}

async function runWarmupPass(prompt) {
  const tokens = await tokenizer(prompt, { truncation: true, max_length: 1024 });
  await session.run({
    input_ids: tokens.input_ids,
    attention_mask: tokens.attention_mask
  });
}

async function initEngine(basePath = "artifacts_v2") {
  try {
    // 0. Ensure dependencies are loaded
    await ensureDependencies();

    // 1. Session leak fix: Release previous session handle if present
    if (session) {
      try {
        await session.release();
      } catch (relErr) {
        console.warn("Error releasing previous session:", relErr);
      }
      session = null;
    }

    // 2. Load prompt contract
    self.postMessage({ type: "PROGRESS", stage: "loading_contract", message: "Loading prompt contract..." });
    const contractRes = await fetch("prompt_contract.json");
    if (!contractRes.ok) {
      throw new Error(`Failed to fetch prompt_contract.json: HTTP ${contractRes.status}`);
    }
    promptContract = await contractRes.json();

    // 3. Load calibrator
    self.postMessage({ type: "PROGRESS", stage: "loading_calibrator", message: "Loading model calibrator..." });
    let calRes = await fetch(`${basePath}/calibrator_modernbert.json`);
    if (!calRes.ok) {
      calRes = await fetch(`${basePath}/calibrator.json`);
    }
    if (!calRes.ok) {
      throw new Error(`Calibrator not found in ${basePath} (checked calibrator_modernbert.json and calibrator.json)`);
    }
    const calData = await calRes.json();
    if (!calData.temperature) {
      throw new Error("Invalid calibrator file: missing 'temperature' parameter");
    }
    calibratorTemp = parseFloat(calData.temperature);

    // 4. Load tokenizer (RC-1: ensure tokenizer loading from basePath works locally with fallback)
    self.postMessage({ type: "PROGRESS", stage: "loading_tokenizer", message: "Loading ModernBERT BPE tokenizer..." });
    try {
      tokenizer = await AutoTokenizer.from_pretrained(basePath, { local_files_only: true });
    } catch (e1) {
      try {
        tokenizer = await AutoTokenizer.from_pretrained(`./${basePath}`, { local_files_only: true });
      } catch (e2) {
        try {
          tokenizer = await AutoTokenizer.from_pretrained(basePath);
        } catch (e3) {
          console.warn("Local tokenizer loading failed, falling back to HuggingFace hub:", e3);
          tokenizer = await AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-base-v2.0");
        }
      }
    }

    // 5. Download model with streaming progress & Cache API
    self.postMessage({ type: "PROGRESS", stage: "loading_model", message: "Downloading ONNX model weights..." });
    const candidateModelUrls = [
      `${basePath}/model_fp16.onnx`,
      `${basePath}/model.onnx`,
      `${basePath}/openjev_modernbert.onnx`
    ];
    let modelBuffer = null;
    let loadedModelUrl = "";

    for (const testUrl of candidateModelUrls) {
      try {
        modelBuffer = await fetchWithCacheAndProgress(testUrl, (received, total) => {
          self.postMessage({
            type: "PROGRESS",
            stage: "downloading_weights",
            received,
            total,
            percent: total > 0 ? ((received / total) * 100).toFixed(1) : null,
            message: `Downloading model weights: ${(received / (1024 * 1024)).toFixed(1)} MB` + (total > 0 ? ` / ${(total / (1024 * 1024)).toFixed(1)} MB` : "")
          });
        });
        loadedModelUrl = testUrl;
        break;
      } catch (err) {
        console.warn(`Model not found at ${testUrl}, trying next candidate...`);
      }
    }

    if (!modelBuffer) {
      throw new Error(`Could not find valid ONNX model in ${basePath} (tried model_fp16.onnx, model.onnx, openjev_modernbert.onnx)`);
    }

    modelSizeBytes = modelBuffer.byteLength;
    const modelSizeMb = (modelSizeBytes / (1024 * 1024)).toFixed(1);

    // Record which ONNX file actually loaded so the UI can name the real precision
    activeModelFile = loadedModelUrl.split("/").pop();
    activePrecision = /fp16/i.test(activeModelFile) ? "fp16" : "fp32";

    // 6. False backend fix: Attempt WebGPU alone first; catch and attempt WASM alone
    let usedProvider = "WASM";
    if (typeof navigator !== "undefined" && navigator.gpu) {
      try {
        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all"
        });
        usedProvider = "WebGPU";
      } catch (gpuErr) {
        console.warn("WebGPU initialization failed, falling back to WASM:", gpuErr);
      }
    }

    if (!session) {
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
      usedProvider = "WASM";
    }

    activeBackend = usedProvider;

    // 7. Warm-up inferences
    // WebGPU compiles a shader variant per tensor shape, so a single synthetic warmup
    // leaves the first real query paying compilation. Warm the shapes real presets use
    // (K=4 and K=8 candidates at roughly 100-token contexts) and discard the outputs.
    self.postMessage({ type: "PROGRESS", stage: "warming_up", message: "Compiling shader variants (warm-up passes)..." });
    const warmupPrompt = `${promptContract.label_marker}sample${promptContract.label_marker}${promptContract.abstention_description}${promptContract.sep_marker}warmup`;
    await runWarmupPass(warmupPrompt);
    await runWarmupPass(buildWarmupPrompt(4));
    await runWarmupPass(buildWarmupPrompt(8));

    self.postMessage({
      type: "READY",
      backend: activeBackend,
      temperature: calibratorTemp,
      modelSizeMb: Number(modelSizeMb)
    });
  } catch (err) {
    self.postMessage({
      type: "ERROR",
      message: `Initialization failed: ${err.message || err}`
    });
  }
}

// Forward-pass repeats per click. Only the forward pass repeats; tokenization
// and post-processing stay single-shot.
const FORWARD_RUNS = 3;

async function runInference(payload) {
  const { id, text, question, candidates, threshold } = payload;
  const startTime = performance.now();

  if (!session || !tokenizer || !promptContract) {
    throw new Error("Engine not initialized");
  }

  if (candidates.length > promptContract.max_candidates) {
    throw new Error(`Candidate count ${candidates.length} exceeds maximum model capacity of ${promptContract.max_candidates}`);
  }

  // 1. Format prompt strictly from prompt_contract.json
  const t0 = performance.now();
  const resolvedQuestion = question || promptContract.default_question;
  const formattedText = promptContract.input_template
    .replace("{question}", resolvedQuestion)
    .replace("{context}", text);

  const labelPrefix = candidates.map(c => `${promptContract.label_marker}${c.desc}`).join("");
  const prompt = `${labelPrefix}${promptContract.sep_marker}${formattedText}`;

  // 2. Tokenize with explicit bounds
  const tokenized = await tokenizer(prompt, { truncation: true, max_length: 1024 });
  const tokenizationMs = performance.now() - t0;
  const tokenCount = tokenized.input_ids.dims ? tokenized.input_ids.dims[1] : (tokenized.input_ids.length || 0);
  const isTruncated = tokenCount >= 1024;

  // 3. Run ONNX model FORWARD_RUNS times and keep the median.
  // A single performance.now() sample swings widely with WebGPU queue scheduling and readback.
  const feeds = {
    input_ids: tokenized.input_ids,
    attention_mask: tokenized.attention_mask
  };

  const forwardSamples = [];
  let results = null;
  for (let run = 0; run < FORWARD_RUNS; run++) {
    const t1 = performance.now();
    results = await session.run(feeds);
    forwardSamples.push(performance.now() - t1);
  }

  const sortedSamples = [...forwardSamples].sort((a, b) => a - b);
  const inferenceMs = sortedSamples[Math.floor(sortedSamples.length / 2)];
  const minForwardMs = sortedSamples[0];
  const maxForwardMs = sortedSamples[sortedSamples.length - 1];

  // 4. Readback and slice candidate logits
  const t2 = performance.now();
  const rawOutput = results.logits.data; // Float32Array
  const numCandidates = candidates.length;
  const candidateLogits = Array.from(rawOutput.slice(0, numCandidates));

  // 5. Calibrate logits with temperature T
  const T = (typeof calibratorTemp === "number" && calibratorTemp > 0) ? calibratorTemp : 1.0;
  const scaledLogits = candidateLogits.map(l => l / T);

  // 6. Stable Softmax
  const maxLogit = Math.max(...scaledLogits);
  const exps = scaledLogits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);
  const postprocessingMs = performance.now() - t2;
  const wallClockMs = performance.now() - startTime;
  // Single-run equivalent: tokenization + median forward pass + post-processing.
  const totalMs = tokenizationMs + inferenceMs + postprocessingMs;

  // 7. Find winner & policy action
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const selectedCandidate = candidates[maxIdx];
  const selectedProb = probs[maxIdx];
  const isAbstention = selectedCandidate.id === promptContract.abstention_id;
  const isAutonomous = !isAbstention && selectedProb >= threshold;

  let policyAction = "";
  if (isAbstention) {
    policyAction = "route_tier1_support";
  } else if (isAutonomous) {
    policyAction = `execute_${selectedCandidate.id}`;
  } else {
    policyAction = "escalate_supervisor_review";
  }

  const isCalibratedScope = (numCandidates === 5);
  const probField = isCalibratedScope ? "calibrated_probabilities" : "probabilities";

  // Build verifiable decision receipt
  const receipt = {
    receipt_version: "2.0.0",
    model_id: "Verdict-ModernBERT-151M",
    execution_backend: activeBackend,
    model_precision: activePrecision,
    model_file: activeModelFile,
    timestamp: new Date().toISOString(),
    query_id: id || "web_query",
    input_token_count: tokenCount,
    truncated: isTruncated,
    candidate_count: numCandidates,
    calibration: {
      status: isCalibratedScope ? "calibrated_for_scope" : "unvalidated_scope",
      temperature: T,
      method: "L-BFGS scalar temperature scaling"
    },
    temperature_applied: T,
    raw_logits: candidateLogits,
    [probField]: Object.fromEntries(
      candidates.map((c, i) => [c.id, Number(probs[i].toFixed(5))])
    ),
    selected: {
      id: selectedCandidate.id,
      probability: Number(selectedProb.toFixed(5)),
      is_abstention: isAbstention
    },
    policy: {
      threshold: threshold,
      is_autonomous: isAutonomous,
      action: policyAction
    },
    timings_ms: {
      tokenization: Number(tokenizationMs.toFixed(2)),
      forward_readback: Number(inferenceMs.toFixed(2)),
      runs: FORWARD_RUNS,
      p50_ms: Number(inferenceMs.toFixed(2)),
      min_ms: Number(minForwardMs.toFixed(2)),
      max_ms: Number(maxForwardMs.toFixed(2)),
      samples_ms: forwardSamples.map(s => Number(s.toFixed(2))),
      postprocessing: Number(postprocessingMs.toFixed(2)),
      total_end_to_end: Number(totalMs.toFixed(2)),
      wall_clock_all_runs: Number(wallClockMs.toFixed(2))
    }
  };

  self.postMessage({
    type: "RESULT",
    id,
    candidates: candidates.map((c, i) => ({
      ...c,
      prob: probs[i],
      rawLogit: candidateLogits[i]
    })),
    selected: selectedCandidate,
    selectedProb,
    isAbstention,
    isAutonomous,
    policyAction,
    timings: receipt.timings_ms,
    receipt
  });
}

// Module-level serialized queue for message handling (RC-6)
self.onmessage = function (e) {
  const { type, payload } = e.data;
  if (type === "INIT") {
    inferenceQueue = inferenceQueue
      .catch(() => {})
      .then(async () => {
        try {
          await initEngine(payload?.basePath || "artifacts_v2");
        } catch (err) {
          self.postMessage({
            type: "ERROR",
            message: `Initialization failed: ${err.message || err}`
          });
        }
      });
  } else if (type === "INFER") {
    inferenceQueue = inferenceQueue
      .catch(() => {})
      .then(async () => {
        try {
          await runInference(payload);
        } catch (err) {
          self.postMessage({
            type: "INFER_ERROR",
            id: payload?.id,
            message: err.message || String(err)
          });
        }
      });
  }
};
