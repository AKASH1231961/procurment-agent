"""
Inventory + supplier-discovery DB layer for the Agentic Procurement
Negotiation System.

Covers the slice of the pipeline described in the proposal as:
"An inventory check identifies an item that needs restocking and
hands the task to the Buyer Agent. The Buyer Agent discovers a
candidate supplier..." -- up to the point where a candidate is
selected and handed off to actual negotiation (Phase 2+, A2A).

Framework-agnostic: wrap these as tools in whatever you use for the
Buyer Agent (Claude API tool-calling, MCP server, etc.) per the
proposal's tech stack.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "procurement.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(schema_path: str = "schema.sql") -> None:
    with get_conn() as conn, open(schema_path) as f:
        conn.executescript(f.read())


def add_item(sku, name, current_stock, reorder_threshold,
             category=None, unit="pcs", reorder_quantity=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO items
               (sku, name, category, unit, current_stock, reorder_threshold, reorder_quantity)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sku, name, category, unit, current_stock, reorder_threshold, reorder_quantity),
        )


# ---------------------------------------------------------------
# Inventory Monitor. Run on a schedule (cron, a loop, an
# orchestrator's timer node). Opens a procurement_task for any item
# that's short and doesn't already have one in flight -- this is
# what stops the same shortage from spawning duplicate tasks every
# time the monitor runs.
# ---------------------------------------------------------------
def get_open_restock_needs():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM items
               WHERE current_stock <= reorder_threshold
                 AND item_id NOT IN (
                     SELECT item_id FROM procurement_tasks
                     WHERE status NOT IN ('settled', 'failed', 'cancelled')
                 )"""
        ).fetchall()

        new_tasks = []
        for item in rows:
            target_qty = item["reorder_quantity"] or (
                item["reorder_threshold"] * 2 - item["current_stock"]
            )
            cur = conn.execute(
                """INSERT INTO procurement_tasks (item_id, target_quantity, status)
                   VALUES (?, ?, 'open')""",
                (item["item_id"], target_qty),
            )
            new_tasks.append({
                "task_id": cur.lastrowid,
                "item_id": item["item_id"],
                "sku": item["sku"],
                "name": item["name"],
                "target_quantity": target_qty,
            })
        return new_tasks


# ---------------------------------------------------------------
# Buyer Agent, sourcing step: called once per candidate the web
# search tool surfaces for a given task.
# ---------------------------------------------------------------
def log_candidate_supplier(task_id, supplier_name, url=None, contact_info=None,
                            estimated_price=None, currency="USD", snippet=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO candidate_suppliers
               (task_id, supplier_name, url, contact_info, estimated_price, currency, snippet)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (task_id, supplier_name, url, contact_info, estimated_price, currency, snippet),
        )
        conn.execute(
            "UPDATE procurement_tasks SET status='sourcing' WHERE task_id=? AND status='open'",
            (task_id,),
        )


def get_candidates(task_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM candidate_suppliers WHERE task_id=? ORDER BY discovered_at",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def clear_candidates(task_id):
    """Remove the supplier candidates associated with a completed sourcing task."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM candidate_suppliers WHERE task_id=?",
            (task_id,),
        )


# ---------------------------------------------------------------
# Buyer Agent, evaluation step: the agent itself -- per the
# proposal, "evaluates them against procurement policy" -- scores
# candidates and picks who to open a negotiation with. This just
# records that decision; the policy logic lives in the agent, not here.
# ---------------------------------------------------------------
def select_candidate(candidate_id, evaluation_note=None):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT task_id FROM candidate_suppliers WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        conn.execute(
            """UPDATE candidate_suppliers SET evaluation_status='selected', evaluation_note=?
               WHERE candidate_id=?""",
            (evaluation_note, candidate_id),
        )
        conn.execute(
            """UPDATE candidate_suppliers SET evaluation_status='rejected'
               WHERE task_id=? AND candidate_id!=? AND evaluation_status='new'""",
            (row["task_id"], candidate_id),
        )
        conn.execute(
            "UPDATE procurement_tasks SET status='negotiating' WHERE task_id=?",
            (row["task_id"],),
        )


# ---------------------------------------------------------------
# Closing the loop: called once negotiation + payment settle
# (Phase 4/5), or if a task fails / gets cancelled.
# ---------------------------------------------------------------
def close_task(task_id, outcome, quantity_received=None):
    assert outcome in ("settled", "failed", "cancelled")
    with get_conn() as conn:
        conn.execute(
            "UPDATE procurement_tasks SET status=?, closed_at=? WHERE task_id=?",
            (outcome, datetime.now(timezone.utc).isoformat(), task_id),
        )
        if outcome == "settled" and quantity_received:
            item = conn.execute(
                "SELECT item_id FROM procurement_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            conn.execute(
                "UPDATE items SET current_stock = current_stock + ? WHERE item_id=?",
                (quantity_received, item["item_id"]),
            )


if __name__ == "__main__":
    # Smoke test: shortage -> task opened -> candidate found -> selected.
    init_db("schema.sql")
    add_item(sku="BOLT-8MM", name="8mm Hex Bolt", current_stock=12,
             reorder_threshold=25, reorder_quantity=100, category="hardware")

    tasks = get_open_restock_needs()
    print("Opened tasks:", tasks)

    if tasks:
        tid = tasks[0]["task_id"]
        log_candidate_supplier(tid, supplier_name="Acme Fasteners",
                                url="https://acmefasteners.example",
                                estimated_price=0.08,
                                snippet="Bulk 8mm hex bolts, ships same day")
        candidates = get_candidates(tid)
        print("Candidates:", candidates)
        select_candidate(candidates[0]["candidate_id"], evaluation_note="Lowest price, in-stock")