"""Unit tests for tabular compaction and exact lossless reconstruction."""

import pytest

from core.tabular import Table, TableCapacityError, TableCompactor


def test_table_lossless_roundtrip() -> None:
    columns = [("item", "Item Description"), ("qty", "Quantity"), ("price", "Unit Price")]
    rows_data = [
        {"id": "row_1", "item": "Cloud Compute Hours", "qty": "100", "price": "0.50"},
        {"id": "row_2", "item": "Storage (GB)", "qty": "500", "price": "0.02"},
        {"id": "row_3", "item": "Bandwidth Out", "qty": None, "price": "0.00"},  # null qty
        {"id": "row_4", "item": "Discount Promo", "qty": "1", "price": ""},      # empty string price
    ]
    table = Table.from_rows(
        table_id="invoice_101",
        columns=columns,
        rows_data=rows_data,
        units={"price": "USD"},
    )

    compactor = TableCompactor(max_block_tokens=120)
    blocks = compactor.compact(table)
    assert len(blocks) >= 1

    # Verify decompact
    reconstructed_table = compactor.decompact(blocks)
    assert reconstructed_table.id == "invoice_101"
    assert reconstructed_table.columns == table.columns
    assert reconstructed_table.units == {"price": "USD"}

    reconstructed_rows = reconstructed_table.reconstruct_rows()
    assert len(reconstructed_rows) == 4
    assert reconstructed_rows[0] == rows_data[0]
    assert reconstructed_rows[1] == rows_data[1]
    # Check null preservation vs empty string
    assert reconstructed_rows[2]["qty"] is None
    assert reconstructed_rows[3]["price"] == ""


def test_table_continuation_records_for_oversized_cell() -> None:
    columns = [("desc", "Detailed Log")]
    long_text = "ERROR_TRACE_" * 50  # ~600 characters
    rows_data = [{"id": "r1", "desc": long_text}]

    table = Table.from_rows(table_id="log_table", columns=columns, rows_data=rows_data)
    compactor = TableCompactor(max_block_tokens=50)  # Very tight budget to force splitting
    blocks = compactor.compact(table)

    assert len(blocks) > 1

    reconstructed_table = compactor.decompact(blocks)
    reconstructed_rows = reconstructed_table.reconstruct_rows()
    assert reconstructed_rows[0]["desc"] == long_text


def test_table_capacity_error_on_exceeded_total_budget() -> None:
    columns = [("c1", "Column 1")]
    rows_data = [{"id": f"r_{i}", "c1": f"data_value_{i}"} for i in range(50)]
    table = Table.from_rows(table_id="large_table", columns=columns, rows_data=rows_data)

    compactor = TableCompactor(max_block_tokens=120)
    with pytest.raises(TableCapacityError, match="exceeds total budget"):
        compactor.compact(table, max_total_tokens=20)


def test_table_custom_row_id_key() -> None:
    columns = [("name", "Name"), ("role", "Role")]
    rows_data = [
        {"user_id": "u_1", "name": "Alice", "role": "admin"},
        {"user_id": "u_2", "name": "Bob", "role": "viewer"},
    ]
    table = Table.from_rows(
        table_id="users_tbl",
        columns=columns,
        rows_data=rows_data,
        row_id_key="user_id",
    )
    reconstructed = table.reconstruct_rows()
    assert reconstructed == rows_data
    assert "user_id" in reconstructed[0]
    assert "id" not in reconstructed[0]
