"use client";

import type { Chess, Square } from "chess.js";

const GLYPH: Record<string, string> = { wk: "♔", wq: "♕", wr: "♖", wb: "♗", wn: "♘", wp: "♙", bk: "♚", bq: "♛", br: "♜", bb: "♝", bn: "♞", bp: "♟" };
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

export function ChessBoard({ chess, selected, targets, lastMove, flipped, onSquare, disabled }: {
  chess: Chess;
  selected: Square | null;
  targets: Set<string>;
  lastMove: { from: string; to: string } | null;
  flipped: boolean;
  onSquare: (sq: Square) => void;
  disabled: boolean;
}) {
  const board = chess.board();
  const ranks = flipped ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0]; // board()[0] is rank 8
  const files = flipped ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  return (
    <div className="grid aspect-square w-full max-w-[34rem] grid-cols-8 overflow-hidden rounded-md border border-border" role="grid" aria-label="Chess board">
      {ranks.map((r) =>
        files.map((f) => {
          const sq = `${FILES[f]}${8 - (7 - r)}` as Square;
          const piece = board[7 - r][f];
          const dark = (r + f) % 2 === 0;
          const isSel = selected === sq;
          const isTarget = targets.has(sq);
          const isLast = lastMove?.from === sq || lastMove?.to === sq;
          return (
            <button key={sq} type="button" onClick={() => onSquare(sq)} disabled={disabled} aria-label={`${sq}${piece ? ` ${piece.color === "w" ? "white" : "black"} ${piece.type}` : ""}`}
              className={`relative flex aspect-square items-center justify-center text-[clamp(1.25rem,4.2vw,2.4rem)] leading-none outline-none transition-colors
                ${dark ? "bg-muted" : "bg-background"}
                ${isLast ? "bg-foreground/10" : ""}
                ${isSel ? "ring-2 ring-inset ring-foreground" : ""}
                ${disabled ? "cursor-default" : "hover:bg-foreground/5"}
                focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring`}>
              {piece && <span className={piece.color === "w" ? "text-foreground drop-shadow-[0_0_1px_var(--background)]" : "text-foreground"} style={piece.color === "w" ? { WebkitTextStroke: "0.5px var(--foreground)", color: "var(--background)" } : undefined}>{GLYPH[piece.color + piece.type]}</span>}
              {isTarget && <span aria-hidden className={`absolute ${piece ? "inset-1 rounded-sm border-2 border-foreground/50" : "size-2.5 rounded-full bg-foreground/40"}`} />}
              {f === files[0] && <span aria-hidden className="absolute left-1 top-0.5 text-[9px] text-muted-foreground">{8 - (7 - r)}</span>}
              {r === ranks[ranks.length - 1] && <span aria-hidden className="absolute bottom-0.5 right-1 text-[9px] text-muted-foreground">{FILES[f]}</span>}
            </button>
          );
        }),
      )}
    </div>
  );
}
