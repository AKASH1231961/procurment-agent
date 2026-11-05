import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seller_agent import NegotiationMessage, PricingStrategy, SellerAgent


def test_rule_based_offer_accepts_when_above_target():
    pricing = PricingStrategy(
        base_price=100.0,
        reservation_price=90.0,
        target_price=100.0,
        volume_discount_threshold=10,
        volume_discount_rate=0.05,
        flexibility="moderate",
    )

    agent = SellerAgent(
        agent_id="seller_test",
        company_name="Test Supplier",
        product_catalog={"Widget": pricing},
        use_llm=False,
    )

    message = NegotiationMessage(
        message_type="PROPOSE",
        session_id="session-1",
        from_agent="buyer",
        to_agent=agent.agent_id,
        timestamp="2026-01-01T00:00:00",
        payload={
            "product": "Widget",
            "quantity": 5,
            "proposed_price_per_unit": 105.0,
        },
    )

    response = agent.handle_message(message)

    assert response.message_type == "ACCEPT"
    assert response.payload["agreed_price_per_unit"] == 105.0
    assert "meets or exceeds our listed price" in response.payload["reasoning"]


def test_offer_above_listed_price_is_accepted_without_llm():
    pricing = PricingStrategy(
        base_price=0.46,
        reservation_price=0.414,
        target_price=0.437,
    )
    agent = SellerAgent(
        agent_id="seller_listed_price_test",
        company_name="Test Supplier",
        product_catalog={"8mm Hex Bolt": pricing},
        use_llm=False,
    )
    llm_calls = []

    def reject_with_llm(*args):
        llm_calls.append(args)
        return {"action": "REJECT", "reasoning": "LLM refusal"}

    agent.use_llm = True
    agent._decide_with_llm = reject_with_llm
    message = NegotiationMessage(
        message_type="PROPOSE",
        session_id="above-listed-price",
        from_agent="buyer",
        to_agent=agent.agent_id,
        timestamp="",
        payload={
            "product": "8mm Hex Bolt",
            "quantity": 10,
            "proposed_price_per_unit": 90.0,
        },
    )

    response = agent.handle_message(message)

    assert response.message_type == "ACCEPT"
    assert response.payload["agreed_price_per_unit"] == 90.0
    assert "listed price of $0.46" in response.payload["reasoning"]
    assert llm_calls == []


def test_rule_based_offer_generates_counter_when_between_reservation_and_target():
    pricing = PricingStrategy(
        base_price=100.0,
        reservation_price=90.0,
        target_price=100.0,
        volume_discount_threshold=10,
        volume_discount_rate=0.05,
        flexibility="moderate",
    )

    agent = SellerAgent(
        agent_id="seller_test_2",
        company_name="Test Supplier",
        product_catalog={"Widget": pricing},
        use_llm=False,
    )

    message = NegotiationMessage(
        message_type="PROPOSE",
        session_id="session-2",
        from_agent="buyer",
        to_agent=agent.agent_id,
        timestamp="2026-01-01T00:00:00",
        payload={
            "product": "Widget",
            "quantity": 5,
            "proposed_price_per_unit": 95.0,
        },
    )

    response = agent.handle_message(message)

    assert response.message_type == "COUNTER"
    assert response.payload["counter_price_per_unit"] > 95.0
    assert response.payload["counter_price_per_unit"] <= 100.0
    assert "below our target" in response.payload["reasoning"]
    assert "moderate approach moves 60%" in response.payload["reasoning"]


def test_seller_confirms_the_buyer_accepted_counter_price():
    pricing = PricingStrategy(
        base_price=100.0,
        reservation_price=90.0,
        target_price=100.0,
    )
    agent = SellerAgent(
        agent_id="seller_accept_test",
        company_name="Test Supplier",
        product_catalog={"Widget": pricing},
        use_llm=False,
    )
    proposal = NegotiationMessage(
        message_type="PROPOSE",
        session_id="accept-counter",
        from_agent="buyer",
        to_agent=agent.agent_id,
        timestamp="",
        payload={
            "product": "Widget",
            "quantity": 4,
            "proposed_price_per_unit": 95.0,
        },
    )
    counter = agent.handle_message(proposal)
    accepted_price = counter.payload["counter_price_per_unit"]

    confirmation = agent.handle_message(NegotiationMessage(
        message_type="ACCEPT",
        session_id="accept-counter",
        from_agent="buyer",
        to_agent=agent.agent_id,
        timestamp="",
        payload={
            "product": "Widget",
            "quantity": 4,
            "agreed_price_per_unit": accepted_price,
        },
    ))

    assert confirmation.message_type == "CONFIRM"
    assert confirmation.payload["agreed_price_per_unit"] == accepted_price
    assert confirmation.payload["total_amount"] == accepted_price * 4
    assert agent.active_sessions["accept-counter"]["status"] == "agreed"


def test_first_ranked_supplier_is_chosen_by_lowest_price():
    items = [
        {"company_name": "Expensive Supplier", "estimated_price": 120.0},
        {"company_name": "Cheap Supplier", "estimated_price": 90.0},
        {"company_name": "Mid Supplier", "estimated_price": 105.0},
    ]

    best = min(items, key=lambda item: item["estimated_price"])

    assert best["company_name"] == "Cheap Supplier"
