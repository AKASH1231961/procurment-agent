import sqlite3
import io
from datetime import datetime, timedelta, timezone

import supervisor_dashboard as dashboard
import negotiation_activity


def test_parse_sessions_reads_dialogue_outcome_and_supervisor_feedback():
    log_text = """## Negotiation session `session-1`

- Started (UTC): 2026-11-05T10:00:00+00:00
- Seller agent: `seller_one`
- Supplier: Supplier One
- Product: Widget
- Quantity: 4
- Opening offer: 90.00 per unit
- Buyer maximum: 110.00 per unit
- Maximum rounds: 3
- Supplier benchmark price: 105.00 per unit

### T001 — Buyer — PROPOSE

- Logged (UTC): 2026-11-05T10:00:01+00:00
- Says: proposes 90 per unit.

### T002 — Seller — COUNTER

- Logged (UTC): 2026-11-05T10:00:02+00:00
- Says: counters at 100 per unit.

**Outcome: agreed**
- Agreed price: 100.00 per unit

### Parent agent evaluation

- Feedback: Try a midpoint counter next time.
- Next-session strategy: `counter_once`
"""

    session = dashboard._parse_sessions(log_text)[0]

    assert session["id"] == "session-1"
    assert session["status"] == "Agreed"
    assert session["supplier"] == "Supplier One"
    assert session["benchmark_price"] == "105.00 per unit"
    assert session["agreed_price"] == "100.00 per unit"
    assert [message["message_type"] for message in session["messages"]] == [
        "PROPOSE",
        "COUNTER",
    ]
    assert session["evaluation"] == {
        "feedback": "Try a midpoint counter next time.",
        "strategy": "counter_once",
    }


def test_snapshot_summarizes_database_and_negotiation_log(tmp_path, monkeypatch):
    database_path = tmp_path / "procurement.db"
    log_path = tmp_path / "negotiations.md"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE items (
                item_id INTEGER PRIMARY KEY, sku TEXT, name TEXT, category TEXT,
                unit TEXT, current_stock REAL, reorder_threshold REAL,
                reorder_quantity REAL, updated_at TEXT
            );
            CREATE TABLE procurement_tasks (
                task_id INTEGER PRIMARY KEY, item_id INTEGER, target_quantity REAL,
                status TEXT, opened_at TEXT, closed_at TEXT
            );
            CREATE TABLE candidate_suppliers (
                candidate_id INTEGER PRIMARY KEY, task_id INTEGER,
                supplier_name TEXT, estimated_price REAL, currency TEXT,
                evaluation_status TEXT, discovered_at TEXT
            );
            INSERT INTO items VALUES (1, 'SKU-1', 'Widget', 'parts', 'pcs', 2, 5, 10, 'now');
            INSERT INTO procurement_tasks VALUES (1, 1, 10, 'sourcing', 'now', NULL);
            INSERT INTO procurement_tasks VALUES (2, 1, 10, 'settled', 'then', 'done');
            INSERT INTO procurement_tasks VALUES (3, 1, 10, 'failed', 'then', 'done');
            INSERT INTO candidate_suppliers VALUES (1, 1, 'Supplier A', 4.5, 'USD', 'new', 'now');
            """
        )
    log_path.write_text(
        """## Negotiation session `session-old`
- Started (UTC): 2026-11-05T10:00:00+00:00
- Product: Widget
**Outcome: agreed**
### LLM token usage
- Input tokens: 10
- Output tokens: 5
- Total tokens: 15
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "DATABASE_PATH", database_path)
    monkeypatch.setattr(dashboard, "LOG_PATH", log_path)

    snapshot = dashboard.get_dashboard_snapshot()

    assert snapshot["shortages"] == 1
    assert snapshot["active_tasks"] == 1
    assert snapshot["agreement_rate"] == 1
    assert snapshot["task_completion"] == {
        "settled_tasks": 1,
        "closed_tasks": 2,
        "rate": 0.5,
    }
    assert snapshot["token_usage"] == {"input": 10, "output": 5, "total": 15}
    assert snapshot["suppliers"][0]["supplier_name"] == "Supplier A"


def test_negotiation_metrics_split_rates_and_exclude_missing_benchmarks():
    sessions = [
        {
            "outcome": "agreed",
            "product": "Widget",
            "supplier": "Supplier A",
            "agreed_price": "80.00 per unit",
            "max_price": "100.00 per unit",
            "benchmark_price": "100.00 per unit",
        },
        {
            "outcome": "agreed",
            "product": "Widget",
            "supplier": "Supplier B",
            "agreed_price": "60.00 per unit",
            "max_price": "50.00 per unit",
            "benchmark_price": "50.00 per unit",
        },
        {
            "outcome": "agreed",
            "product": "Gadget",
            "supplier": "Supplier A",
            "agreed_price": "40.00 per unit",
            "max_price": "50.00 per unit",
            "benchmark_price": None,
        },
        {
            "outcome": "rejected",
            "product": "Gadget",
            "supplier": "Supplier B",
        },
        {
            "outcome": None,
            "product": "Widget",
            "supplier": "Supplier A",
        },
    ]

    metrics = dashboard._negotiation_metrics(sessions)

    assert metrics["agreement_rate"] == 0.75
    assert metrics["budget_compliance_rate"] == 2 / 3
    assert metrics["budget_compliant_agreements"] == 2
    assert metrics["budget_evaluable_agreements"] == 3
    assert metrics["price_quality_sample_size"] == 2
    assert metrics["price_quality_rate"] == 0
    assert metrics["missing_benchmarks"] == 1
    assert metrics["agreement_by_product"] == [
        {"name": "Gadget", "agreements": 1, "completed": 2, "rate": 0.5},
        {"name": "Widget", "agreements": 2, "completed": 2, "rate": 1},
    ]
    assert metrics["agreement_by_supplier"] == [
        {"name": "Supplier A", "agreements": 2, "completed": 2, "rate": 1},
        {"name": "Supplier B", "agreements": 1, "completed": 2, "rate": 0.5},
    ]


def test_recent_unfinished_session_is_marked_in_progress():
    started_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    log_text = (
        "## Negotiation session `live-session`\n"
        f"- Started (UTC): {started_at.isoformat()}\n"
        "- Product: Widget\n"
    )

    session = dashboard._parse_sessions(log_text)[0]

    assert session["status"] == "In progress"


def test_dashboard_and_refresh_routes_render_monitoring_sections():
    client = dashboard.app.test_client()

    for path in ("/", "/snapshot"):
        response = client.get(path)

        assert response.status_code == 200
        page = response.get_data(as_text=True)
        assert "Supervisor overview" in page
        assert "Inventory watch" in page
        assert "Negotiation conversations" in page
        if path == "/":
            assert 'id="max-price"' in page
            assert "Run agent" in page
            assert 'id="agent-run-output"' in page


def test_sidebar_navigation_targets_exist_and_scroll_script_preserves_position():
    response = dashboard.app.test_client().get("/")
    page = response.get_data(as_text=True)
    target_ids = ("overview", "inventory", "tasks", "conversations")

    for target_id in target_ids:
        assert f'href="#{target_id}"' in page
        assert f'id="{target_id}"' in page
    assert "target.scrollIntoView(" in page
    assert "window.scrollTo(0, scrollPosition)" in page


def test_run_agent_validates_authorized_price_and_streams_fixed_command(
    monkeypatch,
):
    launched_commands = []

    class FakeProcess:
        stdout = io.StringIO("Inventory alert\nAgent completed\n")

        def wait(self):
            return 0

    class ImmediateThread:
        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    def fake_popen(command, **kwargs):
        launched_commands.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(dashboard.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)
    client = dashboard.app.test_client()

    invalid = client.post("/run-agent", data={"max_price": "not-a-price"})
    assert invalid.status_code == 400
    assert launched_commands == []

    started = client.post("/run-agent", data={"max_price": "25.5"})
    assert started.status_code == 202
    command, options = launched_commands[0]
    assert command == [
        dashboard.sys.executable,
        "-u",
        str(dashboard.AGENT_SCRIPT),
        "--max-price",
        "25.5",
    ]
    assert "shell" not in options
    assert options["stdout"] == dashboard.subprocess.PIPE
    assert options["stderr"] == dashboard.subprocess.STDOUT

    state = client.get("/run-status?after=-1").get_json()
    assert state["status"] == "completed"
    assert [line["text"] for line in state["lines"]] == [
        "Inventory alert",
        "Agent completed",
    ]


def test_active_dialogue_is_read_from_live_activity_store(tmp_path, monkeypatch):
    activity_path = tmp_path / "activity.db"
    monkeypatch.setattr(negotiation_activity, "ACTIVITY_DB_PATH", activity_path)
    monkeypatch.setattr(dashboard, "LOG_PATH", tmp_path / "empty-log.md")
    monkeypatch.setattr(
        dashboard,
        "DATABASE_PATH",
        tmp_path / "missing-procurement.db",
    )
    negotiation_activity.start_session(
        session_id="live-session",
        product="Widget",
        seller="seller_one",
        quantity=4,
        opening_offer=90,
        max_price=110,
    )
    negotiation_activity.record_message(
        session_id="live-session",
        turn_number=1,
        speaker="Buyer",
        message_type="PROPOSE",
        message_text="proposes 90 per unit.",
    )
    negotiation_activity.record_message(
        session_id="live-session",
        turn_number=2,
        speaker="Seller",
        message_type="COUNTER",
        message_text="counters at 100 per unit.",
    )

    snapshot = dashboard.get_dashboard_snapshot()

    assert snapshot["active_sessions"] == 1
    assert snapshot["sessions"][0]["id"] == "live-session"
    assert snapshot["sessions"][0]["status"] == "In progress"
    assert [message["message_type"] for message in snapshot["sessions"][0]["messages"]] == [
        "PROPOSE",
        "COUNTER",
    ]

    negotiation_activity.finish_session("live-session", "agreed")
    assert negotiation_activity.get_active_sessions() == []
