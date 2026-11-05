import pytest

import negotiation_metrics as metrics
import supervisor_dashboard as dashboard


def _msg(turn, who, kind, ts, key=None, price=None):
    payload = f'```json\n{{"{key}": {price}}}\n```\n' if key else ""
    return f"\n### T{turn:03d} — {who} — {kind}\n\n- Logged (UTC): 2026-11-05T10:00:{ts:02d}+00:00\n\n{payload}"


ADVICE = (
    "\n### Retrieved tactical advice\n\n- Round: 1\n- Triggers: large_gap\n"
    "- Retrieval status: examples_found\n- Corpus examples: a1\n"
)


def _session(sid, start, product, strategy, body, outcome, agreed=None, maxp=110):
    return (
        f"\n## Negotiation session `{sid}`\n\n- Started (UTC): {start}\n- Product: {product}\n"
        f"- Quantity: 10\n- Opening offer: 80.00 per unit\n- Buyer maximum: {maxp}.00 per unit\n"
        + body
        + f"\n**Outcome: {outcome}**\n"
        + (f"- Agreed price: {agreed} per unit\n" if agreed else "")
        + f"\n### Parent agent evaluation\n\n- Feedback: f\n- Next-session strategy: `{strategy}`\n"
    )


LOG = (
    # s1: advised round, gap 40 -> 10 (75% closure), then agreed at 100
    _session(
        "s1", "2026-11-05T10:00:00+00:00", "Widget", "counter_once",
        _msg(1, "Buyer", "PROPOSE", 0, "proposed_price_per_unit", 80)
        + _msg(2, "Seller", "COUNTER", 2, "counter_price_per_unit", 120) + ADVICE
        + _msg(3, "Buyer", "COUNTER", 3, "counter_price_per_unit", 90)
        + _msg(4, "Seller", "COUNTER", 6, "counter_price_per_unit", 100)
        + _msg(5, "Buyer", "ACCEPT", 7, "agreed_price_per_unit", 100)
        + _msg(6, "Seller", "CONFIRM", 8, "agreed_price_per_unit", 100),
        "agreed", "100.00",
    )
    # s2: same product, coached by counter_once, agreed at 90
    + _session(
        "s2", "2026-11-05T11:00:00+00:00", "Widget", "standard",
        _msg(1, "Buyer", "PROPOSE", 0, "proposed_price_per_unit", 80)
        + _msg(2, "Seller", "ACCEPT", 1, "agreed_price_per_unit", 80),
        "agreed", "80.00",
    )
)


def sessions():
    return dashboard._parse_sessions(LOG)


def test_parser_extracts_message_prices_and_advice():
    s1 = next(s for s in sessions() if s["id"] == "s1")
    assert [m["price"] for m in s1["messages"]] == [80, 120, 90, 100, 100, 100]
    assert s1["tactical_advice"] == [
        {"after_turn": 2, "round": "1", "triggers": "large_gap", "status": "examples_found"}
    ]


def test_deal_quality():
    result = metrics.deal_quality(sessions())
    assert result["savings_vs_budget"] == pytest.approx((10 / 110 + 30 / 110) / 2)
    assert result["savings_vs_first_ask"] == pytest.approx((120 - 100) / 120)
    assert result["total_cost_avoided"] == pytest.approx(10 * 10 + 30 * 10)


def test_latency_uses_message_timestamps():
    result = metrics.latency(sessions())
    assert result["max_session_seconds"] == 8
    # s1 seller replies: 2, 3, 1 s; s2: 1 s
    assert result["avg_seller_response_seconds"] == pytest.approx((2 + 3 + 1 + 1) / 4)
    assert result["avg_buyer_decision_seconds"] == pytest.approx((1 + 1) / 2)


def test_learning_progress_tracks_change_and_applied_strategy():
    result = metrics.learning_progress(sessions())
    row = result["learning_by_product"][0]
    assert row["product"] == "Widget"
    assert row["change"] == pytest.approx(30 / 110 - 10 / 110)
    assert result["learning_improved_products"] == 1
    assert result["savings_by_applied_strategy"] == [
        {"strategy": "counter_once", "sessions": 1, "avg_savings": pytest.approx(30 / 110)}
    ]


def test_retrieval_effectiveness_compares_advised_and_other_rounds():
    result = metrics.retrieval_effectiveness(sessions())
    assert result["retrievals"] == 1
    assert result["advised"]["rounds"] == 1
    assert result["advised"]["avg_gap_closure"] == pytest.approx(0.75)
    # Round at turn 4 (gap 10) is followed by CONFIRM -> agreed, not advised.
    assert result["not_advised"]["agreed_next"] == 1
    assert result["agreement_rate_with_retrieval"] == 1
    assert result["sessions_without_retrieval"] == 1
