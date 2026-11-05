"""Read-only supervisor dashboard for inventory and agent negotiations.

Run with ``python3 supervisor_dashboard.py`` and open http://127.0.0.1:5000.
Run the inventory agent in a separate terminal; live dialogue appears when an
authorized negotiation is actually running.
"""

import json
import math
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, make_response, render_template, request

from negotiation_activity import get_active_sessions
from negotiation_metrics import (
    deal_quality,
    latency,
    learning_progress,
    price_value as _price_value,
    retrieval_effectiveness,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "procurement.db"
LOG_PATH = PROJECT_ROOT / "negotiation_logs.md"
ACTIVE_TASK_STATUSES = ("open", "sourcing", "negotiating")
AGENT_SCRIPT = PROJECT_ROOT / "inventory_langchain_agent.py"
MAX_RUN_OUTPUT_LINES = 2000

app = Flask(__name__)


@app.template_filter("pct")
def _fmt_pct(value: float | None, digits: int = 0, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:{'+' if signed else ''}.{digits}f}%"


@app.template_filter("duration")
def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m {rest:02d}s"


@app.template_filter("when")
def _fmt_when(value: str | None) -> str:
    if not value:
        return "—"
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return stamp.strftime("%d %b %Y, %H:%M:%S")
_run_lock = threading.Lock()
_agent_run: dict[str, Any] = {
    "status": "idle",
    "lines": [],
    "next_line_id": 0,
    "returncode": None,
    "error": None,
}


def _append_run_output(text: str) -> None:
    with _run_lock:
        _agent_run["lines"].append({
            "id": _agent_run["next_line_id"],
            "text": text,
        })
        _agent_run["next_line_id"] += 1
        if len(_agent_run["lines"]) > MAX_RUN_OUTPUT_LINES:
            del _agent_run["lines"][:-MAX_RUN_OUTPUT_LINES]


def _consume_agent_output(process: subprocess.Popen[str]) -> None:
    stream = process.stdout
    if stream is None:
        with _run_lock:
            _agent_run["status"] = "failed"
            _agent_run["error"] = "Agent output stream was not available."
        return

    try:
        for line in stream:
            _append_run_output(line.rstrip("\r\n"))
    finally:
        returncode = process.wait()
        with _run_lock:
            _agent_run["returncode"] = returncode
            _agent_run["status"] = "completed" if returncode == 0 else "failed"


def _agent_run_snapshot(after: int = -1) -> dict[str, Any]:
    with _run_lock:
        lines = list(_agent_run["lines"])
        return {
            "status": _agent_run["status"],
            "lines": [line for line in lines if line["id"] > after],
            "first_line_id": lines[0]["id"] if lines else _agent_run["next_line_id"],
            "next_line_id": _agent_run["next_line_id"],
            "returncode": _agent_run["returncode"],
            "error": _agent_run["error"],
        }


def _field(text: str, label: str) -> str | None:
    match = re.search(
        rf"^- {re.escape(label)}: (.*)$",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1).strip().strip("`") if match else None


def _parse_sessions(log_text: str) -> list[dict[str, Any]]:
    sessions = []
    sections = log_text.split("## Negotiation session ")[1:]
    now = datetime.now(timezone.utc)

    for section in sections:
        header = re.match(r"`([^`]+)`", section)
        if not header:
            continue

        started_at = _field(section, "Started (UTC)")
        try:
            start_time = (
                datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                if started_at
                else None
            )
            if start_time and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except ValueError:
            start_time = None

        outcome_match = re.search(
            r"^\*\*Outcome: ([a-z_]+)\*\*$",
            section,
            flags=re.MULTILINE,
        )
        outcome = outcome_match.group(1) if outcome_match else None
        if outcome:
            status = outcome.replace("_", " ").title()
        elif start_time and (now - start_time).total_seconds() <= 300:
            status = "In progress"
        else:
            status = "Incomplete"

        messages = []
        tactical_advice = []
        message_sections = re.split(
            r"^### T(\d+) [—-] (Buyer|Seller) [—-] ([A-Z]+)\s*$",
            section,
            flags=re.MULTILINE,
        )
        for index in range(1, len(message_sections), 4):
            body = message_sections[index + 3]
            says = re.search(r"^- Says: (.*)$", body, flags=re.MULTILINE)
            timestamp = _field(body, "Logged (UTC)")
            turn = int(message_sections[index])
            message_type = message_sections[index + 2]
            messages.append({
                "turn": turn,
                "speaker": message_sections[index + 1],
                "message_type": message_type,
                "timestamp": timestamp,
                "text": says.group(1).strip() if says else "",
                "price": _message_price(body, message_type),
            })
            for advice in re.finditer(
                r"^### Retrieved tactical advice\s*\n(.*?)(?=^### |^\*\*Outcome|\Z)",
                body,
                flags=re.MULTILINE | re.DOTALL,
            ):
                tactical_advice.append({
                    "after_turn": turn,
                    "round": _field(advice.group(1), "Round"),
                    "triggers": _field(advice.group(1), "Triggers"),
                    "status": _field(advice.group(1), "Retrieval status"),
                })

        evaluation = None
        evaluation_marker = "### Parent agent evaluation"
        if evaluation_marker in section:
            evaluation_text = section.split(evaluation_marker, 1)[1]
            evaluation = {
                "feedback": _field(evaluation_text, "Feedback"),
                "strategy": _field(evaluation_text, "Next-session strategy"),
            }

        sessions.append({
            "id": header.group(1),
            "started_at": started_at,
            "seller": _field(section, "Seller agent") or "Unknown supplier",
            "supplier": _field(section, "Supplier")
            or _field(section, "Seller agent")
            or "Unknown supplier",
            "product": _field(section, "Product") or "Unknown item",
            "quantity": _field(section, "Quantity"),
            "opening_offer": _field(section, "Opening offer"),
            "max_price": _field(section, "Buyer maximum"),
            "benchmark_price": _field(section, "Supplier benchmark price"),
            "status": status,
            "outcome": outcome,
            "agreed_price": _field(section, "Agreed price"),
            "reason": _field(section, "Reason"),
            "messages": messages,
            "tactical_advice": tactical_advice,
            "evaluation": evaluation,
        })

    return list(reversed(sessions))


def _message_price(body: str, message_type: str) -> float | None:
    """Per-unit price a message carries, from its JSON payload or its summary line."""
    if message_type == "REJECT":
        return None
    payload = re.search(r"```json\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
    if payload:
        try:
            data = json.loads(payload.group(1))
        except ValueError:
            data = {}
        for key in ("proposed_price_per_unit", "counter_price_per_unit", "agreed_price_per_unit"):
            if isinstance(data.get(key), (int, float)):
                return float(data[key])
    says = re.search(r"^- Says: (.*)$", body, flags=re.MULTILINE)
    return _price_value(says.group(1)) if says else None


def _read_token_usage(log_text: str) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "total": 0}
    for section in re.split(r"^### LLM token usage\s*$", log_text, flags=re.MULTILINE)[1:]:
        for key in totals:
            match = re.search(rf"^- {key.title()} tokens: (\d+)$", section, flags=re.MULTILINE)
            if match:
                totals[key] += int(match.group(1))
    return totals


def _read_database() -> dict[str, Any]:
    data: dict[str, Any] = {
        "items": [],
        "tasks": [],
        "suppliers": [],
        "database_error": None,
        "task_completion": {
            "settled_tasks": 0,
            "closed_tasks": 0,
            "rate": None,
        },
    }
    if not DATABASE_PATH.is_file():
        data["database_error"] = f"Database not found: {DATABASE_PATH.name}"
        return data

    database_uri = f"{DATABASE_PATH.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(database_uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            task_counts = connection.execute(
                """
                SELECT
                    COUNT(*) AS closed_tasks,
                    COALESCE(SUM(CASE WHEN status='settled' THEN 1 ELSE 0 END), 0)
                        AS settled_tasks
                FROM procurement_tasks
                WHERE closed_at IS NOT NULL
                  AND status IN ('settled', 'failed', 'cancelled')
                """
            ).fetchone()
            closed_tasks = task_counts["closed_tasks"]
            settled_tasks = task_counts["settled_tasks"]
            data["task_completion"] = {
                "settled_tasks": settled_tasks,
                "closed_tasks": closed_tasks,
                "rate": settled_tasks / closed_tasks if closed_tasks else None,
            }
            data["items"] = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT item_id, sku, name, category, unit, current_stock,
                           reorder_threshold, reorder_quantity, updated_at
                    FROM items
                    ORDER BY
                        CASE WHEN current_stock <= reorder_threshold THEN 0 ELSE 1 END,
                        name
                    LIMIT 100
                    """
                )
            ]
            data["tasks"] = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT tasks.task_id, tasks.target_quantity, tasks.status,
                           tasks.opened_at, tasks.closed_at, items.name AS item_name,
                           items.sku
                    FROM procurement_tasks AS tasks
                    JOIN items ON items.item_id = tasks.item_id
                    ORDER BY tasks.opened_at DESC, tasks.task_id DESC
                    LIMIT 100
                    """
                )
            ]
            data["suppliers"] = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT candidate_id, task_id, supplier_name, estimated_price,
                           currency, evaluation_status, discovered_at
                    FROM candidate_suppliers
                    ORDER BY discovered_at DESC, candidate_id DESC
                    LIMIT 100
                    """
                )
            ]
    except sqlite3.Error as error:
        data["database_error"] = f"Could not read procurement database: {error}"

    return data


def _negotiation_metrics(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        session for session in sessions
        if session["outcome"] in {"agreed", "rejected"}
    ]
    agreements = [session for session in completed if session["outcome"] == "agreed"]
    compliant = 0
    budget_evaluable = 0
    price_quality_values = []
    missing_benchmarks = 0

    for session in agreements:
        agreed_price = _price_value(session.get("agreed_price"))
        max_price = _price_value(session.get("max_price"))
        benchmark_price = _price_value(session.get("benchmark_price"))
        if agreed_price is not None and max_price is not None:
            budget_evaluable += 1
            if agreed_price <= max_price:
                compliant += 1
        if agreed_price is not None and benchmark_price is not None:
            price_quality_values.append(
                (benchmark_price - agreed_price) / benchmark_price
            )
        else:
            missing_benchmarks += 1

    def breakdown(key: str) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for session in completed:
            label = session.get(key) or "Unknown"
            groups.setdefault(label, []).append(session)
        return [
            {
                "name": label,
                "agreements": sum(row["outcome"] == "agreed" for row in rows),
                "completed": len(rows),
                "rate": sum(row["outcome"] == "agreed" for row in rows) / len(rows),
            }
            for label, rows in sorted(
                groups.items(),
                key=lambda group: (-len(group[1]), group[0].casefold()),
            )
        ]

    return {
        "completed_negotiations": len(completed),
        "agreements": len(agreements),
        "agreement_rate": len(agreements) / len(completed) if completed else None,
        "agreement_by_product": breakdown("product"),
        "agreement_by_supplier": breakdown("supplier"),
        "budget_compliance_rate": (
            compliant / budget_evaluable if budget_evaluable else None
        ),
        "budget_compliant_agreements": compliant,
        "budget_evaluable_agreements": budget_evaluable,
        "price_quality_rate": (
            sum(price_quality_values) / len(price_quality_values)
            if price_quality_values
            else None
        ),
        "price_quality_sample_size": len(price_quality_values),
        "missing_benchmarks": missing_benchmarks,
        **deal_quality(sessions),
        **latency(sessions),
        **learning_progress(sessions),
        "retrieval": retrieval_effectiveness(sessions),
    }


def get_dashboard_snapshot() -> dict[str, Any]:
    log_text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.is_file() else ""
    sessions = _parse_sessions(log_text)
    active_sessions = get_active_sessions()
    active_by_id = {session["session_id"]: session for session in active_sessions}
    for session in sessions:
        live_session = active_by_id.pop(session["id"], None)
        if live_session:
            session["status"] = "In progress"
            session["messages"] = live_session["messages"]
    for live_session in active_by_id.values():
        sessions.insert(0, {
            "id": live_session["session_id"],
            "started_at": live_session["started_at"],
            "seller": live_session["seller"],
            "supplier": live_session["seller"],
            "product": live_session["product"],
            "quantity": str(live_session["quantity"]),
            "opening_offer": f"{live_session['opening_offer']:.2f} per unit",
            "max_price": f"{live_session['max_price']:.2f} per unit",
            "status": "In progress",
            "outcome": None,
            "agreed_price": None,
            "reason": None,
            "messages": live_session["messages"],
            "evaluation": None,
        })
    database = _read_database()
    metrics = _negotiation_metrics(sessions)
    active_session_count = sum(session["status"] == "In progress" for session in sessions)
    items = database["items"]
    tasks = database["tasks"]
    return {
        **database,
        "sessions": sessions[:30],
        **metrics,
        "active_sessions": active_session_count,
        "shortages": sum(item["current_stock"] <= item["reorder_threshold"] for item in items),
        "active_tasks": sum(task["status"] in ACTIVE_TASK_STATUSES for task in tasks),
        "token_usage": _read_token_usage(log_text),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


@app.get("/")
def dashboard():
    return render_template("dashboard.html", snapshot=get_dashboard_snapshot())


@app.get("/snapshot")
def snapshot():
    response = make_response(
        render_template("_dashboard_content.html", snapshot=get_dashboard_snapshot())
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/run-agent")
def run_agent():
    raw_price = request.form.get("max_price", "").strip()
    try:
        max_price = float(raw_price)
    except ValueError:
        return jsonify(error="Enter a valid authorized maximum price."), 400
    if not math.isfinite(max_price) or max_price <= 0:
        return jsonify(error="Maximum price must be a finite positive number."), 400
    if not AGENT_SCRIPT.is_file():
        return jsonify(error=f"Agent entry point not found: {AGENT_SCRIPT.name}"), 500

    with _run_lock:
        if _agent_run["status"] == "running":
            return jsonify(error="An agent run is already in progress."), 409
        _agent_run.update({
            "status": "running",
            "lines": [],
            "next_line_id": 0,
            "returncode": None,
            "error": None,
        })
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(AGENT_SCRIPT),
                    "--max-price",
                    str(max_price),
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            _agent_run["status"] = "failed"
            _agent_run["error"] = f"Could not start the agent: {error}"
            return jsonify(error=_agent_run["error"]), 500

    threading.Thread(
        target=_consume_agent_output,
        args=(process,),
        name="procurement-agent-output",
        daemon=True,
    ).start()
    return jsonify(status="running"), 202


@app.get("/run-status")
def run_status():
    raw_after = request.args.get("after", "-1")
    try:
        after = int(raw_after)
    except ValueError:
        return jsonify(error="'after' must be a line cursor integer."), 400
    if after < -1:
        return jsonify(error="'after' cannot be less than -1."), 400
    response = jsonify(_agent_run_snapshot(after))
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
