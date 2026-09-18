import type { Metadata } from "next";
import { ChessGame } from "@/components/chess-game";

export const metadata: Metadata = { title: "kev · chess", description: "A decision model playing chess: every move is a Choice question." };

export default function ChessPage() {
  return <ChessGame />;
}
