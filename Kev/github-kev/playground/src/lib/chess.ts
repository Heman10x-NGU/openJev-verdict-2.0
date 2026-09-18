// Chess via the decision model: legal moves become the options of one Choice question,
// the position (board, FEN, history) is the state. A Score question rates the position in the same pass.
import { Chess, type Move } from "chess.js";
import { api, type SystemOneRequest, type SystemOneResponse } from "@/lib/kev";

export const PIECE_NAMES: Record<string, string> = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
export const EVAL_LEVELS = ["Black is clearly winning", "Black is better", "Roughly equal", "White is better", "White is clearly winning"];

export function describeMove(m: Move): string {
  const parts = [`${PIECE_NAMES[m.piece]} ${m.from} to ${m.to}`];
  if (m.captured) parts.push(`captures ${PIECE_NAMES[m.captured]}`);
  if (m.promotion) parts.push(`promotes to ${PIECE_NAMES[m.promotion]}`);
  if (m.flags.includes("k") || m.flags.includes("q")) parts[0] = m.flags.includes("k") ? "castles kingside" : "castles queenside";
  if (m.san.endsWith("#")) parts.push("checkmate");
  else if (m.san.endsWith("+")) parts.push("gives check");
  return parts.join(", ");
}

export function positionState(chess: Chess) {
  const side = chess.turn() === "w" ? "White" : "Black";
  const history = chess.history();
  const moves = history.length ? history.map((m, i) => (i % 2 === 0 ? `${i / 2 + 1}. ${m}` : m)).join(" ") : "(game start)";
  return {
    game: "chess",
    side_to_move: side,
    board: chess.ascii(),
    fen: chess.fen(),
    moves_so_far: moves,
    in_check: chess.inCheck(),
  };
}

export function buildRequest(chess: Chess): { req: SystemOneRequest; legal: Move[] } {
  const legal = chess.moves({ verbose: true });
  const side = chess.turn() === "w" ? "White" : "Black";
  const criteria: Record<string, string> = {};
  for (const m of legal) criteria[m.san] = describeMove(m);
  return {
    legal,
    req: {
      state: positionState(chess),
      model: "kev-latest",
      questions: {
        move: {
          type: "choice",
          instructions: `You are playing ${side}. Choose the best legal move for ${side} in this position. Prefer captures of undefended pieces, checks that win material, and moves that develop pieces toward the center.`,
          criteria,
        },
        evaluation: {
          type: "score",
          instructions: "Who is better in this position, before the move is played?",
          criteria: EVAL_LEVELS,
        },
      },
    },
  };
}

export type ModelMove = {
  san: string;
  probabilities: Record<string, number>;
  confidence: number;
  evaluation: number; // expected level 0..4
  evalConfidence?: number; // absent in games saved by older versions
  evalProbabilities: Record<string, number>;
  latency_ms: number;
  input_tokens: number;
  n_legal: number;
};

export async function askModel(chess: Chess, sample = false): Promise<ModelMove> {
  const { req, legal } = buildRequest(chess);
  const r: SystemOneResponse = await api.systemOne(req);
  const a = r.answers.move;
  const e = r.answers.evaluation;
  if (a.type !== "choice" || e.type !== "score") throw new Error("unexpected answer types");
  let san = a.choice;
  if (sample) {
    let u = Math.random();
    for (const [k, p] of Object.entries(a.probabilities)) { u -= p; if (u <= 0) { san = k; break; } }
  }
  if (!legal.some((m) => m.san === san)) san = a.choice; // should not happen: options are the legal moves
  return { san, probabilities: a.probabilities, confidence: a.confidence, evaluation: e.score, evalConfidence: e.confidence, evalProbabilities: e.probabilities, latency_ms: r.latency_ms ?? 0, input_tokens: r.usage.input_tokens, n_legal: legal.length };
}

// ---- persistence -------------------------------------------------------------------------------

export type Mode = "self" | "white" | "black"; // who the human plays; "self" = model vs model

export type SavedGame = {
  id: string;
  startedAt: number;
  mode: Mode;
  pgn: string;
  moves: { san: string; by: "human" | "model"; model?: ModelMove }[];
  result?: string;
};

const KEY = "kev.chess.v1";

export function loadGames(): SavedGame[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(KEY) ?? "[]"); } catch { return []; }
}

export function saveGames(games: SavedGame[]) {
  localStorage.setItem(KEY, JSON.stringify(games.slice(-20)));
}

export function resultText(chess: Chess): string | undefined {
  if (!chess.isGameOver()) return undefined;
  if (chess.isCheckmate()) return chess.turn() === "w" ? "0-1 (checkmate)" : "1-0 (checkmate)";
  if (chess.isStalemate()) return "½-½ (stalemate)";
  if (chess.isThreefoldRepetition()) return "½-½ (repetition)";
  if (chess.isInsufficientMaterial()) return "½-½ (insufficient material)";
  if (chess.isDrawByFiftyMoves()) return "½-½ (fifty-move rule)";
  return "½-½ (draw)";
}
