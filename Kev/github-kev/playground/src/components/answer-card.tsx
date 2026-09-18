import type { Answer, Question } from "@/lib/kev";

// One shared lane layout for every bar on the page: label | track | value.
// Only the fill length varies between rows, so lengths are comparable across cards.
const LANES = "grid grid-cols-[minmax(0,6.5rem)_minmax(0,1fr)_2.75rem] items-center gap-x-3";

function Bar({ label, p, top, delta, mono }: { label: string; p: number; top: boolean; delta?: number; mono?: boolean }) {
  const showDelta = delta !== undefined && Math.abs(delta) >= 0.01;
  return (
    <div className={`${LANES} text-[12px] leading-5`}>
      <span className={`truncate ${mono ? "font-mono" : ""} ${top ? "text-foreground" : "text-muted-foreground"}`} title={label}>{label}</span>
      <span className="relative block h-1 min-w-0 rounded-full bg-muted" aria-hidden>
        <span className={`absolute inset-y-0 left-0 rounded-full ${top ? "bg-foreground" : "bg-muted-foreground/50"}`} style={{ width: `${Math.round(p * 100)}%` }} />
      </span>
      <span className="text-right tabular-nums">
        <span className={top ? "text-foreground" : "text-muted-foreground"}>{p.toFixed(2)}</span>
        {showDelta && <span className="ml-1 text-[10px] text-muted-foreground">{delta > 0 ? "+" : ""}{delta.toFixed(2)}</span>}
      </span>
    </div>
  );
}

function instructionText(q?: Question) {
  if (!q) return "";
  return typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions);
}

export function AnswerCard({ id, question, answer, compare }: { id: string; question?: Question; answer: Answer; compare?: Answer }) {
  const instr = instructionText(question);
  const confidence = "confidence" in answer ? answer.confidence : undefined;
  const headline =
    answer.type === "noul" ? (answer.noul >= 0.5 ? "yes" : "no")
    : answer.type === "choice" ? answer.choice
    : `${answer.score.toFixed(2)} of ${Object.keys(answer.legend).length - 1}`;
  const detail =
    answer.type === "noul" ? `p(yes) ${answer.noul.toFixed(2)}`
    : `confidence ${confidence!.toFixed(2)}`;

  return (
    <section aria-labelledby={`q-${id}`} className="grid gap-3 rounded-md border border-border bg-card px-4 py-3 md:grid-cols-[11rem_minmax(0,1fr)] md:gap-6">
      <div className="flex min-w-0 flex-col">
        <h3 id={`q-${id}`} className="flex items-baseline gap-1.5 text-[12px] text-muted-foreground">
          <span className="min-w-0 truncate font-mono" title={id}>{id}</span>
          <span className="shrink-0">· {answer.type}</span>
        </h3>
        {instr && <p className="mt-0.5 line-clamp-2 text-[13px] leading-5 text-foreground" title={instr}>{instr}</p>}
        <p className="mt-1.5 flex flex-wrap items-baseline gap-x-2">
          <span className={`text-base font-medium tracking-tight ${answer.type === "choice" ? "font-mono" : ""}`}>{headline}</span>
          <span className="whitespace-nowrap text-[12px] tabular-nums text-muted-foreground">{detail}</span>
        </p>
      </div>

      <div className="flex min-w-0 flex-col justify-center">
        {answer.type === "noul" && (
          <>
            <Bar label="yes" p={answer.noul} top={answer.noul >= 0.5} delta={compare?.type === "noul" ? answer.noul - compare.noul : undefined} />
            <Bar label="no" p={1 - answer.noul} top={answer.noul < 0.5} />
          </>
        )}
        {answer.type === "choice" &&
          Object.entries(answer.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([k, p]) => (
              <Bar key={k} label={k} mono p={p} top={k === answer.choice} delta={compare?.type === "choice" ? p - (compare.probabilities[k] ?? 0) : undefined} />
            ))}
        {answer.type === "score" &&
          Object.entries(answer.probabilities).map(([k, p]) => (
            <Bar key={k} label={`${k}  ${answer.legend[k]}`} p={p} top={Number(k) === Math.round(answer.score)} delta={compare?.type === "score" ? p - (compare.probabilities[k] ?? 0) : undefined} />
          ))}
      </div>
    </section>
  );
}
