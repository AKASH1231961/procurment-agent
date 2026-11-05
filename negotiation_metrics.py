"""Deal-quality, latency, learning and retrieval metrics over parsed negotiation sessions.

Sessions are the dicts built by ``supervisor_dashboard._parse_sessions``. Only
completed (agreed / rejected) sessions are scored.
"""

import math
import re
from datetime import datetime
from statistics import mean, median
from typing import Any


def price_value(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    price = float(match.group())
    return price if math.isfinite(price) and price > 0 else None


def _time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def _completed(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    done = [s for s in sessions if s.get("outcome") in {"agreed", "rejected"}]
    return sorted(done, key=lambda s: s.get("started_at") or "")


def _savings(session: dict[str, Any]) -> float | None:
    """Fraction of the buyer's budget saved: (max - agreed) / max."""
    agreed = price_value(session.get("agreed_price"))
    limit = price_value(session.get("max_price"))
    if session.get("outcome") != "agreed" or agreed is None or limit is None:
        return None
    return (limit - agreed) / limit


def _first_seller_ask(session: dict[str, Any]) -> float | None:
    for message in session.get("messages", []):
        if message["speaker"] == "Seller" and message["message_type"] == "COUNTER":
            return message.get("price")
    return None


def deal_quality(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    agreed = [s for s in _completed(sessions) if s["outcome"] == "agreed"]
    vs_budget, vs_ask, cost_avoided = [], [], 0.0
    for session in agreed:
        saving = _savings(session)
        if saving is None:
            continue
        vs_budget.append(saving)
        agreed_price = price_value(session["agreed_price"])
        limit = price_value(session["max_price"])
        cost_avoided += (limit - agreed_price) * (price_value(session.get("quantity")) or 0)
        ask = _first_seller_ask(session)
        if ask:
            vs_ask.append((ask - agreed_price) / ask)
    return {
        "savings_vs_budget": _avg(vs_budget),
        "savings_vs_budget_sample": len(vs_budget),
        "savings_vs_first_ask": _avg(vs_ask),
        "savings_vs_first_ask_sample": len(vs_ask),
        "total_cost_avoided": cost_avoided,
    }


def latency(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Session duration plus per-turn latency from the logged message timestamps.

    Seller latency: buyer message -> seller reply (the seller's LLM call).
    Buyer latency: seller reply -> next buyer message (decision, retrieval, LLM).
    """
    durations, seller_turns, buyer_turns = [], [], []
    for session in _completed(sessions):
        stamps = [(m["speaker"], _time(m.get("timestamp"))) for m in session.get("messages", [])]
        stamps = [(speaker, t) for speaker, t in stamps if t]
        if len(stamps) < 2:
            continue
        durations.append((stamps[-1][1] - stamps[0][1]).total_seconds())
        for (prev_speaker, prev_t), (speaker, t) in zip(stamps, stamps[1:]):
            if speaker == prev_speaker:
                continue
            (seller_turns if speaker == "Seller" else buyer_turns).append(
                (t - prev_t).total_seconds()
            )
    return {
        "avg_session_seconds": _avg(durations),
        "max_session_seconds": max(durations) if durations else None,
        "latency_sessions": len(durations),
        "avg_seller_response_seconds": _avg(seller_turns),
        "median_seller_response_seconds": median(seller_turns) if seller_turns else None,
        "avg_buyer_decision_seconds": _avg(buyer_turns),
    }


def learning_progress(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Savings vs budget across repeated sessions of the same product."""
    by_product: dict[str, list[dict[str, Any]]] = {}
    for session in _completed(sessions):
        by_product.setdefault(session.get("product") or "Unknown", []).append(session)

    rows, coached = [], {}
    for product, runs in by_product.items():
        series = [s for s in (_savings(run) for run in runs) if s is not None]
        previous_strategy = None
        for run in runs:
            saving = _savings(run)
            if saving is not None and previous_strategy:
                coached.setdefault(previous_strategy, []).append(saving)
            evaluation = run.get("evaluation")
            if evaluation and evaluation.get("strategy"):
                previous_strategy = evaluation["strategy"]
        if len(runs) < 2:
            continue
        rows.append({
            "product": product,
            "sessions": len(runs),
            "agreements": sum(run["outcome"] == "agreed" for run in runs),
            "first_savings": series[0] if series else None,
            "latest_savings": series[-1] if series else None,
            "change": series[-1] - series[0] if len(series) >= 2 else None,
            "savings_series": series,
        })
    rows.sort(key=lambda row: (-row["sessions"], row["product"].casefold()))
    changes = [row["change"] for row in rows if row["change"] is not None]
    return {
        "learning_by_product": rows,
        "learning_avg_change": _avg(changes),
        "learning_improved_products": sum(change > 0 for change in changes),
        "learning_comparable_products": len(changes),
        "savings_by_applied_strategy": [
            {"strategy": name, "sessions": len(values), "avg_savings": mean(values)}
            for name, values in sorted(coached.items())
        ],
    }


def _bucket(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    closures = [r["closure"] for r in rounds if r["closure"] is not None]
    return {
        "rounds": len(rounds),
        "agreed_next": sum(r["agreed"] for r in rounds),
        "avg_gap_closure": _avg(closures),
        "closure_sample": len(closures),
    }


def retrieval_effectiveness(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare what the seller does after a retrieval-advised round with other rounds.

    A round is a seller COUNTER. Its gap is seller price minus the buyer's latest
    offer. The follow-up is the next seller message: ACCEPT/CONFIRM counts as agreed,
    a new COUNTER gives gap closure (gap_before - gap_after) / gap_before. Advised
    rounds are hard ones (large gap or stalled), so this is a comparison, not a
    controlled experiment.
    """
    advised: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    retrievals = found = 0
    advised_sessions = []

    for session in _completed(sessions):
        advice_turns = set()
        for advice in session.get("tactical_advice", []):
            retrievals += 1
            found += advice.get("status") == "examples_found"
            advice_turns.add(advice["after_turn"])
        if advice_turns:
            advised_sessions.append(session)

        buyer_offer = None
        messages = session.get("messages", [])
        for index, message in enumerate(messages):
            if message["speaker"] == "Buyer" and message.get("price") and message["message_type"] != "ACCEPT":
                buyer_offer = message["price"]
            if not (
                message["speaker"] == "Seller"
                and message["message_type"] == "COUNTER"
                and message.get("price")
                and buyer_offer
            ):
                continue
            gap = message["price"] - buyer_offer
            follow_up = next(
                (m for m in messages[index + 1:] if m["speaker"] == "Seller"), None
            )
            if gap <= 0 or follow_up is None:
                continue
            round_row = {"closure": None, "agreed": False}
            if follow_up["message_type"] in {"ACCEPT", "CONFIRM"}:
                round_row["agreed"] = True
            elif follow_up["message_type"] == "COUNTER" and follow_up.get("price"):
                next_offer = next(
                    (m["price"] for m in messages[index + 1:]
                     if m["speaker"] == "Buyer" and m.get("price")),
                    buyer_offer,
                )
                round_row["closure"] = (gap - (follow_up["price"] - next_offer)) / gap
            else:
                continue
            (advised if message["turn"] in advice_turns else other).append(round_row)

    def agree_rate(rows: list[dict[str, Any]]) -> float | None:
        return sum(s["outcome"] == "agreed" for s in rows) / len(rows) if rows else None

    plain = [s for s in _completed(sessions) if s not in advised_sessions]
    return {
        "retrievals": retrievals,
        "retrievals_with_examples": found,
        "advised": _bucket(advised),
        "not_advised": _bucket(other),
        "agreement_rate_with_retrieval": agree_rate(advised_sessions),
        "agreement_rate_without_retrieval": agree_rate(plain),
        "sessions_with_retrieval": len(advised_sessions),
        "sessions_without_retrieval": len(plain),
    }
