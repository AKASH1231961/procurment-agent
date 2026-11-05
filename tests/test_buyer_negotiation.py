import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buyer_negotiation import (
    get_agreement_metrics,
    resolve_buyer_terms,
    run_buyer_negotiation,
)
import negotiation_activity
from seller_agent import NegotiationMessage, PricingStrategy, SellerAgent


class CounteringSeller:
    agent_id = "seller"

    def __init__(self, counter_prices):
        self.counter_prices = iter(counter_prices)
        self.messages = []

    def handle_message(self, message):
        self.messages.append(message)
        if message.message_type in {"PROPOSE", "COUNTER"}:
            try:
                counter_price = next(self.counter_prices)
            except StopIteration:
                return NegotiationMessage(
                    message_type="ACCEPT",
                    session_id=message.session_id,
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    timestamp="",
                    payload={
                        "agreed_price_per_unit": message.payload.get(
                            "counter_price_per_unit",
                            message.payload.get("proposed_price_per_unit"),
                        ),
                    },
                )
            return NegotiationMessage(
                message_type="COUNTER",
                session_id=message.session_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                timestamp="",
                payload={"counter_price_per_unit": counter_price},
            )
        if message.message_type == "ACCEPT":
            return NegotiationMessage(
                message_type="CONFIRM",
                session_id=message.session_id,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                timestamp="",
                payload={
                    "agreed_price_per_unit": message.payload["agreed_price_per_unit"],
                    "total_amount": message.payload["agreed_price_per_unit"]
                    * message.payload["quantity"],
                },
            )
        return NegotiationMessage(
            message_type="ACKNOWLEDGE",
            session_id=message.session_id,
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            timestamp="",
            payload={},
        )


def test_buyer_accepts_seller_counter_within_scheduled_offer():
    seller = CounteringSeller([102.0])

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=110.0,
        max_rounds=1,
    )

    assert result["status"] == "agreed"
    assert result["agreed_price_per_unit"] == 102.0
    assert [message.message_type for message in seller.messages] == [
        "PROPOSE",
        "ACCEPT",
    ]


def test_negotiation_log_records_available_supplier_benchmark(tmp_path):
    log_path = tmp_path / "benchmark-negotiation.md"

    run_buyer_negotiation(
        seller=CounteringSeller([92.0]),
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        session_id="benchmark-session",
        log_path=log_path,
        benchmark_price=95.0,
    )

    assert "- Supplier benchmark price: 95.00 per unit" in log_path.read_text(
        encoding="utf-8"
    )


def test_negotiation_messages_are_visible_while_the_session_is_active(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        negotiation_activity,
        "ACTIVITY_DB_PATH",
        tmp_path / "activity.db",
    )
    observations = []

    class MonitoringSeller(CounteringSeller):
        def handle_message(self, message):
            observations.append(negotiation_activity.get_active_sessions())
            return super().handle_message(message)

    result = run_buyer_negotiation(
        seller=MonitoringSeller([102.0]),
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=110.0,
        max_rounds=1,
        session_id="live-widget-session",
        log_path=tmp_path / "negotiations.md",
    )

    assert result["status"] == "agreed"
    assert observations[0][0]["status"] == "in_progress"
    assert observations[0][0]["messages"][0]["message_type"] == "PROPOSE"
    assert observations[1][0]["messages"][-1]["message_type"] == "ACCEPT"
    assert negotiation_activity.get_active_sessions() == []


def test_parent_agent_feedback_changes_next_same_product_session(tmp_path):
    log_path = tmp_path / "coached-negotiations.md"

    first_result = run_buyer_negotiation(
        seller=CounteringSeller([92.0]),
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        session_id="first-widget-session",
        log_path=log_path,
    )

    assert first_result["status"] == "agreed"
    assert "first counter" in first_result["parent_feedback"]

    second_seller = CounteringSeller([96.0, 96.0])
    second_result = run_buyer_negotiation(
        seller=second_seller,
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        session_id="second-widget-session",
        log_path=log_path,
    )

    buyer_messages = [
        message for message in second_seller.messages
        if message.from_agent == "buyer_agent"
    ]
    assert second_result["status"] == "agreed"
    assert second_result["parent_feedback_applied"] == first_result["parent_feedback"]
    assert buyer_messages[1].message_type == "COUNTER"
    assert buyer_messages[1].payload["counter_price_per_unit"] == 93.0
    assert all(
        message.payload.get("counter_price_per_unit", 0) <= 100.0
        for message in buyer_messages
    )
    assert second_result["agreed_price_per_unit"] <= 100.0


def test_buyer_increases_offer_until_seller_accepts(tmp_path):
    seller = CounteringSeller([108.0, 108.0, 100.0])
    log_path = tmp_path / "negotiations.md"

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=110.0,
        max_rounds=5,
        session_id="session-log-test",
        log_path=log_path,
    )

    buyer_counters = [
        message.payload["counter_price_per_unit"]
        for message in seller.messages
        if message.message_type == "COUNTER"
    ]
    assert result["status"] == "agreed"
    assert buyer_counters == [94.0, 98.0]
    assert max(buyer_counters) <= 110.0
    log = log_path.read_text(encoding="utf-8")
    assert "session-log-test" in log
    assert "### T001 — Buyer — PROPOSE" in log
    assert "### T002 — Seller — COUNTER" in log
    assert "### T006 — Seller — COUNTER" in log
    assert "### T007 — Buyer — ACCEPT" in log
    assert "### T008 — Seller — CONFIRM" in log
    assert "counters at 108.0 per unit" in log
    assert '"counter_price_per_unit": 108.0' in log
    assert "**Outcome: agreed**" in log

    run_buyer_negotiation(
        seller=CounteringSeller([]),
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=110.0,
        max_rounds=5,
        session_id="second-session",
        log_path=log_path,
    )
    appended_log = log_path.read_text(encoding="utf-8")
    assert "session-log-test" in appended_log
    assert "second-session" in appended_log
    assert appended_log.count("### T001 — Buyer — PROPOSE") == 2


def test_buyer_rejects_counter_above_maximum_at_round_limit():
    seller = CounteringSeller([120.0, 120.0])

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=2,
    )

    assert result["status"] == "rejected"
    assert result["rounds"] == 2
    assert [message.message_type for message in seller.messages] == [
        "PROPOSE",
        "COUNTER",
        "REJECT",
    ]
    assert seller.messages[1].payload["counter_price_per_unit"] == 95.0


def test_retrieval_adds_tactical_wording_without_changing_buyer_offer(
    monkeypatch,
    tmp_path,
):
    import buyer_negotiation

    retrieval_queries = []

    def fake_retrieve(query):
        retrieval_queries.append(query)
        return [{
            "id": "example-1",
            "category": "procurement",
            "context": "bulk_order",
            "scenario": "The supplier counter is above budget.",
            "advice": "Ask whether terms can narrow the price gap.",
        }]

    class FakeModel:
        def invoke(self, messages):
            class Response:
                content = "Could the order volume help narrow the price gap?"
                usage_metadata = {
                    "input_tokens": 120,
                    "output_tokens": 14,
                    "total_tokens": 134,
                }
                response_metadata = {"model_name": "test-buyer-model"}

            return Response()

    monkeypatch.setattr(
        buyer_negotiation,
        "retrieve_procurement_examples",
        fake_retrieve,
    )
    seller = CounteringSeller([130.0, 130.0, 130.0])

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=10,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        log_path=tmp_path / "negotiations.md",
        strategy_model=FakeModel(),
    )

    buyer_counters = [
        message
        for message in seller.messages
        if message.message_type == "COUNTER"
    ]
    assert result["status"] == "agreed"
    assert len(retrieval_queries) == 2
    assert "large_gap" in result["tactical_advice"][0]["triggers"]
    assert (
        "seller_not_conceding_after_buyer_increase"
        in result["tactical_advice"][1]["triggers"]
    )
    assert buyer_counters[0].payload["counter_price_per_unit"] == 92.0
    assert buyer_counters[0].payload["buyer_message"] == (
        "Could the order volume help narrow the price gap?"
    )
    log = (tmp_path / "negotiations.md").read_text(encoding="utf-8")
    assert "- Agent: `buyer_agent`" in log
    assert "- Model: `test-buyer-model`" in log
    assert "- Input tokens: 120" in log
    assert "- Output tokens: 14" in log
    assert "- Total tokens: 134" in log
    assert max(
        message.payload["counter_price_per_unit"] for message in buyer_counters
    ) <= 100.0


def test_seller_token_usage_is_logged_for_each_seller_llm_call(
    monkeypatch,
    tmp_path,
):
    class FakeSellerModel:
        model = "test-seller-model"

        def invoke(self, messages):
            class Response:
                content = (
                    '{"action": "COUNTER", "counter_price": 95.0, '
                    '"reasoning": "A reasonable counter."}'
                )
                usage_metadata = {
                    "input_tokens": 80,
                    "output_tokens": 12,
                    "total_tokens": 92,
                }
                response_metadata = {}

            return Response()

    seller = SellerAgent(
        agent_id="seller_usage_test",
        company_name="Test Supplier",
        product_catalog={
            "Widget": PricingStrategy(
                base_price=100.0,
                reservation_price=90.0,
                target_price=100.0,
            )
        },
        use_llm=False,
    )
    monkeypatch.setattr(seller, "use_llm", True)
    monkeypatch.setattr(seller, "llm", FakeSellerModel(), raising=False)
    log_path = tmp_path / "seller-usage.md"

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=4,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=2,
        session_id="seller-usage-test",
        log_path=log_path,
    )

    assert result["status"] == "agreed"
    log = log_path.read_text(encoding="utf-8")
    assert "- Agent: `seller_usage_test`" in log
    assert "- Model: `test-seller-model`" in log
    assert "- Session: `seller-usage-test`" in log
    assert "- Input tokens: 80" in log
    assert "- Output tokens: 12" in log
    assert "- Total tokens: 92" in log


def test_retrieval_triggers_when_the_buyer_seller_gap_stays_level(
    monkeypatch,
    tmp_path,
):
    import buyer_negotiation

    monkeypatch.setattr(
        buyer_negotiation,
        "retrieve_procurement_examples",
        lambda query: [{
            "id": "example-2",
            "category": "vendor_negotiation",
            "context": "price_gap",
            "scenario": "A vendor's counter remains above budget.",
            "advice": "Ask about changing contract terms.",
        }],
    )

    class FakeModel:
        def invoke(self, messages):
            class Response:
                content = "Could we discuss terms that help close the gap?"

            return Response()

    result = run_buyer_negotiation(
        seller=CounteringSeller([130.0, 132.0, 134.0]),
        product="Widget",
        quantity=10,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        log_path=tmp_path / "gap-negotiations.md",
        strategy_model=FakeModel(),
    )

    assert result["status"] == "agreed"
    assert "buyer_seller_gap_not_closing" in result["tactical_advice"][1]["triggers"]


def test_buyer_rejects_invalid_budget_and_offer():
    seller = CounteringSeller([])

    try:
        run_buyer_negotiation(
            seller=seller,
            product="Widget",
            quantity=1,
            initial_offer=101.0,
            max_price=100.0,
        )
    except ValueError as error:
        assert "cannot exceed" in str(error)
    else:
        raise AssertionError("Expected invalid offer to raise ValueError")


def test_buyer_and_rule_based_seller_complete_negotiation(tmp_path):
    seller = SellerAgent(
        agent_id="seller_integration",
        company_name="Test Supplier",
        product_catalog={
            "Widget": PricingStrategy(
                base_price=100.0,
                reservation_price=90.0,
                target_price=95.0,
            )
        },
        use_llm=False,
    )

    result = run_buyer_negotiation(
        seller=seller,
        product="Widget",
        quantity=10,
        initial_offer=90.0,
        max_price=100.0,
        max_rounds=5,
        log_path=tmp_path / "negotiations.md",
    )

    assert result["status"] == "agreed"
    assert result["agreed_price_per_unit"] == 93.8
    assert result["total_amount"] == 938.0
    assert seller.active_sessions["buyer-negotiation"]["status"] == "agreed"


def test_tool_buyer_terms_use_only_documented_offer_and_quantity_defaults():
    assert resolve_buyer_terms(max_price=200.0) == (50, 180.0)
    assert resolve_buyer_terms(
        max_price=200.0,
        quantity=30,
        initial_offer=175.0,
    ) == (30, 175.0)


def test_agreement_metrics_count_completed_outcomes_from_negotiation_log(tmp_path):
    log_path = tmp_path / "negotiations.md"
    log_path.write_text(
        "**Outcome: agreed**\n"
        "**Outcome: rejected**\n"
        "**Outcome: in_progress**\n",
        encoding="utf-8",
    )

    assert get_agreement_metrics(log_path) == {
        "completed_negotiations": 2,
        "agreements": 1,
        "agree_rate": 0.5,
    }


def test_agreement_metrics_are_empty_when_log_does_not_exist(tmp_path):
    assert get_agreement_metrics(tmp_path / "missing.md") == {
        "completed_negotiations": 0,
        "agreements": 0,
        "agree_rate": None,
    }
