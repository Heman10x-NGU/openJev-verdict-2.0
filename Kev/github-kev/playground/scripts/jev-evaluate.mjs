import { createInterface } from "node:readline";
import { experimental_evaluate as evaluate } from "ai";

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  try {
    const request = JSON.parse(line);
    const started = performance.now();
    const result = await evaluate({
      model: "typesafe-ai/jev",
      state: request.state,
      questions: Object.fromEntries(Object.entries(request.questions).map(([id, q]) => [id, {
        ...q,
        type: q.type === "noul" ? "boolean" : q.type,
      }])),
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(60000),
      providerOptions: { gateway: { zeroDataRetention: true } },
    });
    process.stdout.write(JSON.stringify({
      answers: result.answers,
      usage: result.usage,
      latency_ms: performance.now() - started,
      model: "typesafe-ai/jev",
      confidence: result.providerMetadata?.typesafe?.confidence,
    }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: { name: error.name, status: error.statusCode ?? null } }) + "\n");
  }
}
