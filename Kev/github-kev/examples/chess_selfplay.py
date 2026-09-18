"""Model vs model chess through the TypeSafe SDK against a local kev server.

Every move is one Choice question whose options are the legal moves; a Score question rates the
position in the same request. Prints each move with its probability and the top alternatives.

    uv run --extra serve --extra examples python examples/chess_selfplay.py --plies 40
    uv run --extra serve --extra examples python examples/chess_selfplay.py --sample --pgn out.pgn
"""
import argparse, random
import chess, chess.pgn
from typesafe_sdk import Choice, Score, TypeSafeClient

PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
LEVELS = ["Black is clearly winning", "Black is better", "Roughly equal", "White is better", "White is clearly winning"]


def describe(board: chess.Board, mv: chess.Move) -> str:
    if board.is_castling(mv): parts = ["castles kingside" if chess.square_file(mv.to_square) > 4 else "castles queenside"]
    else: parts = [f"{PIECE[board.piece_type_at(mv.from_square)]} {chess.square_name(mv.from_square)} to {chess.square_name(mv.to_square)}"]
    if board.is_capture(mv): parts.append("captures " + PIECE[board.piece_type_at(mv.to_square) or chess.PAWN])
    if mv.promotion: parts.append(f"promotes to {PIECE[mv.promotion]}")
    board.push(mv); check, mate = board.is_check(), board.is_checkmate(); board.pop()
    if mate: parts.append("checkmate")
    elif check: parts.append("gives check")
    return ", ".join(parts)


def state(board: chess.Board) -> dict:
    side = "White" if board.turn else "Black"
    sans, b = [], chess.Board()
    for i, mv in enumerate(board.move_stack):
        sans.append((f"{i // 2 + 1}. " if i % 2 == 0 else "") + b.san(mv)); b.push(mv)
    return {"game": "chess", "side_to_move": side, "board": str(board), "fen": board.fen(), "moves_so_far": " ".join(sans) or "(game start)", "in_check": board.is_check()}


def choose(client: TypeSafeClient, board: chess.Board, sample: bool):
    side = "White" if board.turn else "Black"
    legal = {board.san(mv): mv for mv in board.legal_moves}
    r = client.system_one(
        state=state(board),
        questions={
            "move": Choice(instructions=f"You are playing {side}. Choose the best legal move for {side} in this position. Prefer captures of undefended pieces, checks that win material, and moves that develop pieces toward the center.",
                           criteria={san: describe(board, mv) for san, mv in legal.items()}),
            "evaluation": Score(instructions="Who is better in this position, before the move is played?", criteria=LEVELS),
        },
    )
    a, e = r.choices["move"], r.scores["evaluation"]
    san = random.choices(list(a.probabilities), weights=list(a.probabilities.values()))[0] if sample else a.choice
    return legal[san], san, a, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--plies", type=int, default=60)
    ap.add_argument("--sample", action="store_true", help="sample the move from the distribution instead of taking the argmax")
    ap.add_argument("--pgn", help="write the game to this PGN file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    board = chess.Board()
    with TypeSafeClient(api_key="local", base_url=a.base_url, model="kev-latest") as client:
        for ply in range(a.plies):
            if board.is_game_over(): break
            mv, san, ans, ev = choose(client, board, a.sample)
            top = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])[:4]
            alts = "  ".join(f"{k} {p:.2f}" for k, p in top if k != san)
            print(f"{ply // 2 + 1:>3}{'.' if ply % 2 == 0 else '…'} {san:<7} p={ans.probabilities[san]:.2f} conf={ans.confidence:.2f}  eval={ev.score:.2f}/4  | {alts}")
            board.push(mv)
    print("\nresult:", board.result(claim_draw=True), "|", board.outcome(claim_draw=True).termination.name.lower() if board.outcome(claim_draw=True) else "unfinished", f"| {len(board.move_stack)} plies")
    if a.pgn:
        game = chess.pgn.Game.from_board(board); game.headers.update(Event="kev self-play", White="kev-0.5b", Black="kev-0.5b", Result=board.result(claim_draw=True))
        open(a.pgn, "w").write(str(game)); print("wrote", a.pgn)


if __name__ == "__main__":
    main()
