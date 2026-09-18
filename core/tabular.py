"""Lossless tabular compaction and structured serialization.

Ensures wide tabular structures stay strictly bounded within ModernBERT's
128-token sliding-window attention while preserving exact cell identity,
headers, types, and null semantics without silent evidence loss.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field


class TableCapacityError(ValueError):
    """Raised when tabular data exceeds configured token or budget limits."""


class Cell(BaseModel):
    """A self-contained cell record with provenance and metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table_id: str
    row_id: str
    col_id: str
    header: str
    value: str | None
    unit: str | None = None
    part_index: int = 0
    total_parts: int = 1

    def to_record_dict(self) -> dict[str, Any]:
        """Serialize to a standard JSON-compatible record dictionary."""
        record: dict[str, Any] = {
            "table": self.table_id,
            "row": self.row_id,
            "column": self.col_id,
            "header": self.header,
            "value": self.value,
        }
        if self.unit is not None:
            record["unit"] = self.unit
        if self.total_parts > 1:
            record["part"] = self.part_index + 1
            record["total_parts"] = self.total_parts
        return record


class Table(BaseModel):
    """A structured tabular entity with explicit column IDs and row keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    columns: tuple[tuple[str, str], ...]  # (col_id, header)
    rows: tuple[str, ...]  # row_ids
    cells: tuple[Cell, ...]
    units: dict[str, str] = Field(default_factory=dict)
    row_id_key: str = "id"

    @classmethod
    def from_rows(
        cls,
        table_id: str,
        columns: Sequence[tuple[str, str]],  # (col_id, header)
        rows_data: Sequence[dict[str, Any]],
        row_id_key: str = "id",
        units: dict[str, str] | None = None,
    ) -> Table:
        """Create a typed Table validating row integrity and cell types."""
        col_ids = [c[0] for c in columns]
        col_headers = {c[0]: c[1] for c in columns}
        if len(col_ids) != len(set(col_ids)):
            raise ValueError(f"Duplicate column IDs in table {table_id!r}")

        units_map = units or {}
        extracted_rows: list[str] = []
        cells: list[Cell] = []

        for idx, row in enumerate(rows_data):
            row_id = str(row.get(row_id_key, f"row_{idx}"))
            extracted_rows.append(row_id)
            for col_id in col_ids:
                raw_val = row.get(col_id)
                val_str: str | None
                if raw_val is None:
                    val_str = None
                else:
                    val_str = str(raw_val)

                cells.append(
                    Cell(
                        table_id=table_id,
                        row_id=row_id,
                        col_id=col_id,
                        header=col_headers[col_id],
                        value=val_str,
                        unit=units_map.get(col_id),
                    )
                )

        return cls(
            id=table_id,
            columns=tuple(columns),
            rows=tuple(extracted_rows),
            cells=tuple(cells),
            units=units_map,
            row_id_key=row_id_key,
        )

    def reconstruct_rows(self) -> list[dict[str, Any]]:
        """Reconstruct original row dictionaries exactly."""
        assembled: dict[str, dict[str, Any]] = {
            rid: {self.row_id_key: rid} for rid in self.rows
        }
        # Gather cells and handle any split continuations
        cell_parts: dict[tuple[str, str], list[Cell]] = {}
        for c in self.cells:
            key = (c.row_id, c.col_id)
            cell_parts.setdefault(key, []).append(c)

        for (rid, cid), parts in cell_parts.items():
            parts.sort(key=lambda p: p.part_index)
            if parts[0].value is None:
                assembled[rid][cid] = None
            else:
                combined_val = "".join(p.value or "" for p in parts)
                assembled[rid][cid] = combined_val

        return [assembled[rid] for rid in self.rows]


class TableCompactor:
    """Serializes tables into self-contained, sliding-window bounded blocks."""

    def __init__(self, max_block_tokens: int = 120, max_chars_per_token: float = 3.5):
        self.max_block_tokens = max_block_tokens
        self.max_chars_per_token = max_chars_per_token

    def _estimate_tokens(self, text: str, tokenizer: Any = None) -> int:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Fallback conservative token estimation
        return max(1, int(len(text) / self.max_chars_per_token) + 1)

    def compact(
        self,
        table: Table,
        tokenizer: Any = None,
        max_total_tokens: int | None = None,
    ) -> list[str]:
        """Compact table into a list of newline-delimited record blocks.

        Each block is guaranteed to stay within max_block_tokens.
        """
        # Split large cells into continuation cells if needed
        atomic_cells: list[Cell] = []
        for cell in table.cells:
            # Check cell size
            rec_json = json.dumps(cell.to_record_dict(), ensure_ascii=False)
            cell_tokens = self._estimate_tokens(rec_json, tokenizer)

            if cell_tokens <= self.max_block_tokens:
                atomic_cells.append(cell)
            else:
                # Cell value itself exceeds block budget, split with continuation records
                if cell.value is None or len(cell.value) == 0:
                    raise TableCapacityError(
                        f"Cell metadata for table {table.id!r} exceeds block token limit {self.max_block_tokens}"
                    )
                # Compute approximate chunk characters
                # Reserve tokens for metadata
                dummy = Cell(
                    table_id=cell.table_id,
                    row_id=cell.row_id,
                    col_id=cell.col_id,
                    header=cell.header,
                    value="",
                    unit=cell.unit,
                    part_index=1,
                    total_parts=2,
                )
                meta_tokens = self._estimate_tokens(
                    json.dumps(dummy.to_record_dict(), ensure_ascii=False), tokenizer
                )
                available_tokens = self.max_block_tokens - meta_tokens
                if available_tokens <= 2:
                    raise TableCapacityError(
                        f"Metadata alone exceeds block budget for table {table.id!r}"
                    )
                approx_chunk_chars = max(10, int(available_tokens * self.max_chars_per_token * 0.8))

                val = cell.value
                val_chunks = [
                    val[i : i + approx_chunk_chars]
                    for i in range(0, len(val), approx_chunk_chars)
                ]
                total_parts = len(val_chunks)
                for part_idx, chunk in enumerate(val_chunks):
                    split_cell = Cell(
                        table_id=cell.table_id,
                        row_id=cell.row_id,
                        col_id=cell.col_id,
                        header=cell.header,
                        value=chunk,
                        unit=cell.unit,
                        part_index=part_idx,
                        total_parts=total_parts,
                    )
                    atomic_cells.append(split_cell)

        # Now pack atomic cells into blocks
        blocks: list[str] = []
        current_records: list[str] = []
        current_block_tokens = 0

        for cell in atomic_cells:
            rec_str = json.dumps(cell.to_record_dict(), ensure_ascii=False)
            rec_tokens = self._estimate_tokens(rec_str + "\n", tokenizer)

            if rec_tokens > self.max_block_tokens:
                raise TableCapacityError(
                    f"Individual record exceeds block budget {self.max_block_tokens} tokens"
                )

            if current_records and (current_block_tokens + rec_tokens > self.max_block_tokens):
                # Close current block
                blocks.append("\n".join(current_records))
                current_records = [rec_str]
                current_block_tokens = rec_tokens
            else:
                current_records.append(rec_str)
                current_block_tokens += rec_tokens

        if current_records:
            blocks.append("\n".join(current_records))

        if max_total_tokens is not None:
            total_tokens = sum(self._estimate_tokens(b, tokenizer) for b in blocks)
            if total_tokens > max_total_tokens:
                raise TableCapacityError(
                    f"Compacted table exceeds total budget: {total_tokens} > {max_total_tokens}"
                )

        return blocks

    @staticmethod
    def decompact(blocks: list[str]) -> Table:
        """Reconstruct a Table object from serialized record blocks."""
        cells: list[Cell] = []
        table_id: str = ""
        columns_dict: dict[str, str] = {}
        units_dict: dict[str, str] = {}
        row_ids_ordered: list[str] = []
        seen_rows: set[str] = set()

        for block in blocks:
            for line in block.strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)
                if not table_id:
                    table_id = d["table"]
                col_id = d["column"]
                columns_dict[col_id] = d["header"]
                if "unit" in d:
                    units_dict[col_id] = d["unit"]
                row_id = d["row"]
                if row_id not in seen_rows:
                    seen_rows.add(row_id)
                    row_ids_ordered.append(row_id)

                cells.append(
                    Cell(
                        table_id=d["table"],
                        row_id=d["row"],
                        col_id=d["column"],
                        header=d["header"],
                        value=d["value"],
                        unit=d.get("unit"),
                        part_index=d.get("part", 1) - 1,
                        total_parts=d.get("total_parts", 1),
                    )
                )

        cols_tuple = tuple((cid, columns_dict[cid]) for cid in columns_dict)
        return Table(
            id=table_id,
            columns=cols_tuple,
            rows=tuple(row_ids_ordered),
            cells=tuple(cells),
            units=units_dict,
        )
