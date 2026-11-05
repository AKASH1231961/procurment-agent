import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


load_dotenv()


class PricingStrategy(BaseModel):
    """Defines how a seller prices and negotiates."""

    base_price: float = Field(description="Listed price per unit")
    reservation_price: float = Field(description="Minimum acceptable price (never revealed)")
    target_price: float = Field(description="Desired selling price")
    volume_discount_threshold: int = Field(default=100, description="Min quantity for discount")
    volume_discount_rate: float = Field(default=0.05, description="Discount % for bulk orders")
    flexibility: str = Field(default="moderate", description="negotiation flexibility: firm, moderate, flexible")


class NegotiationOffer(BaseModel):
    """Structure of an offer in negotiation."""

    product: str
    quantity: int
    price_per_unit: float
    currency: str = "USD"
    delivery_days: Optional[int] = None
    payment_terms: Optional[str] = None


class NegotiationMessage(BaseModel):
    """A2A protocol message."""

    message_type: str  # PROPOSE, COUNTER, ACCEPT, REJECT
    session_id: str
    from_agent: str
    to_agent: str
    timestamp: str
    payload: Dict[str, Any]


class SellerAgent:
    """Autonomous seller agent that negotiates on behalf of a supplier.""" #now the seller agent is created by the automatically later it need to be chageed to or connected to a companys agent 


    def __init__(
        self,
        agent_id: str,
        company_name: str,
        product_catalog: Dict[str, PricingStrategy],
        negotiation_style: str = "moderate",
        use_llm: bool = True,
    ):
        self.agent_id = agent_id
        self.company_name = company_name
        self.product_catalog = product_catalog
        self.negotiation_style = negotiation_style
        self.use_llm = use_llm
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.llm_usage_by_session: Dict[str, list[dict[str, Any]]] = {}

        api_key = os.getenv("GROQ_API_KEY")
        if use_llm and not api_key:
            raise ValueError("GROQ_API_KEY is missing")
        if use_llm:
            self.llm = ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=0.3,
                groq_api_key=api_key
            )

    def get_agent_card(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": f"{self.company_name} Sales Agent",
            "version": "1.0",
            "capabilities": ["negotiate", "quote", "bulk_order"],
            "protocols": ["a2a"],
            "endpoints": {
                "negotiate": f"http://localhost:8000/agents/{self.agent_id}/negotiate"
            },
            "company": self.company_name,
            "supported_products": list(self.product_catalog.keys()),
            "negotiation_style": self.negotiation_style,
            "min_order_quantity": 1,
            "response_time_sla": "< 5 seconds",
        }

    def handle_message(self, message: NegotiationMessage) -> NegotiationMessage:
        msg_type = message.message_type
        print(f"\n📥 [{self.company_name}] Received {msg_type}")

        if msg_type == "PROPOSE":
            return self._handle_propose(message)
        elif msg_type == "COUNTER":
            return self._handle_counter(message)
        elif msg_type == "ACCEPT":
            return self._handle_accept(message)
        elif msg_type == "REJECT":
            return self._handle_reject(message)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    def _handle_propose(self, message: NegotiationMessage) -> NegotiationMessage:
        session_id = message.session_id
        payload = message.payload

        product = payload["product"]
        quantity = payload["quantity"]
        buyer_offer = payload["proposed_price_per_unit"]

        pricing = self._get_pricing_for_product(product, quantity)

        if not pricing:
            return self._create_reject_message(
                message,
                reason=f"Product '{product}' not in our catalog",
            )

        self.active_sessions[session_id] = {
            "buyer_agent": message.from_agent,
            "product": product,
            "quantity": quantity,
            "pricing": pricing,
            "history": [message],
            "turn_count": 1,
        }

        decision = self._decide_response(buyer_offer, pricing, quantity, session_id)

        print(f"   Buyer offer: ${buyer_offer}/unit for {quantity} units")
        print(f"   Our target: ${pricing.target_price}/unit")
        print(f"   Our minimum: ${pricing.reservation_price}/unit")
        print(f"   Decision: {decision['action']}")

        if decision["action"] == "ACCEPT":
            self.active_sessions[session_id]["status"] = "agreed"
            self.active_sessions[session_id]["agreed_price_per_unit"] = buyer_offer
            return self._create_accept_message(
                message,
                buyer_offer,
                decision.get("reasoning", "The offer meets our target price."),
            )
        elif decision["action"] == "COUNTER":
            return self._create_counter_message(
                message,
                decision["counter_price"],
                decision.get("reasoning", "We are countering based on our target price and negotiation approach."),
            )
        return self._create_reject_message(
            message,
            decision.get("reasoning", decision.get("reason", "The offer is below the price we can accept.")),
        )

    def _handle_counter(self, message: NegotiationMessage) -> NegotiationMessage:
        session_id = message.session_id
        session = self.active_sessions.get(session_id)

        if not session:
            return self._create_reject_message(
                message,
                reason="Session not found or expired",
            )

        session["history"].append(message)
        session["turn_count"] += 1

        buyer_counter = message.payload["counter_price_per_unit"]
        pricing = session["pricing"]
        quantity = session["quantity"]

        if session["turn_count"] > 5:
            return self._create_reject_message(
                message,
                reason="Negotiation timeout (max turns exceeded)",
            )

        decision = self._decide_response(buyer_counter, pricing, quantity, session_id)

        print(f"   Buyer counter: ${buyer_counter}/unit")
        print(f"   Decision: {decision['action']}")

        if decision["action"] == "ACCEPT":
            return self._create_accept_message(
                message,
                buyer_counter,
                decision.get("reasoning", "The offer meets our target price."),
            )
        elif decision["action"] == "COUNTER":
            return self._create_counter_message(
                message,
                decision["counter_price"],
                decision.get("reasoning", "We are countering based on our target price and negotiation approach."),
            )
        return self._create_reject_message(
            message,
            decision.get("reasoning", decision.get("reason", "The offer is below the price we can accept.")),
        )

    def _handle_accept(self, message: NegotiationMessage) -> NegotiationMessage:
        session_id = message.session_id
        session = self.active_sessions.get(session_id)

        if not session:
            return self._create_reject_message(
                message,
                reason="Session not found or expired",
            )

        agreed_price = message.payload.get("agreed_price_per_unit")
        last_counter = session.get("last_seller_counter")
        if (
            not isinstance(agreed_price, (int, float))
            or not math.isfinite(agreed_price)
            or agreed_price <= 0
            or last_counter is None
            or agreed_price > last_counter
            or message.payload.get("quantity") != session["quantity"]
        ):
            return self._create_reject_message(
                message,
                reason="Accepted price or quantity does not match the negotiation.",
            )

        session["history"].append(message)
        session["status"] = "agreed"
        session["agreed_price_per_unit"] = agreed_price
        print(f"   ✅ Deal closed at ${agreed_price}/unit")

        return NegotiationMessage(
            message_type="CONFIRM",
            session_id=session_id,
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "status": "agreed",
                "agreed_price_per_unit": agreed_price,
                "quantity": session["quantity"],
                "total_amount": agreed_price * session["quantity"],
                "message": "Order confirmed. Proceeding to payment.",
            },
        )

    def _handle_reject(self, message: NegotiationMessage) -> NegotiationMessage:
        session_id = message.session_id
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "buyer_rejected"

        print(f"   ❌ Buyer rejected our offer")

        return NegotiationMessage(
            message_type="ACKNOWLEDGE",
            session_id=session_id,
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={"message": "Understood. Thank you for considering us."},
        )

    def _decide_response(
        self,
        buyer_offer: float,
        pricing: PricingStrategy,
        quantity: int,
        session_id: str,
    ) -> dict:
        if buyer_offer >= pricing.base_price:
            return {
                "action": "ACCEPT",
                "reasoning": (
                    f"The offer of ${buyer_offer:.2f} per unit meets or exceeds "
                    f"our listed price of ${pricing.base_price:.2f} per unit."
                ),
            }
        if self.use_llm:
            return self._decide_with_llm(buyer_offer, pricing, quantity, session_id)
        return self._decide_with_rules(buyer_offer, pricing, quantity)

    def _decide_with_rules(
        self,
        buyer_offer: float,
        pricing: PricingStrategy,
        quantity: int,
    ) -> dict:
        effective_target = pricing.target_price
        if quantity >= pricing.volume_discount_threshold:
            effective_target *= (1 - pricing.volume_discount_rate)

        if buyer_offer >= effective_target:
            return {
                "action": "ACCEPT",
                "reasoning": (
                    f"The offer of ${buyer_offer:.2f} per unit meets or exceeds "
                    f"our target of ${effective_target:.2f} per unit."
                ),
            }

        if buyer_offer >= pricing.reservation_price:
            if pricing.flexibility == "firm":
                counter = pricing.target_price
            elif pricing.flexibility == "flexible":
                counter = (buyer_offer + effective_target) / 2
            else:
                counter = buyer_offer + (effective_target - buyer_offer) * 0.6

            return {
                "action": "COUNTER",
                "counter_price": round(counter, 2),
                "reasoning": self._counter_reason(
                    pricing.flexibility,
                    buyer_offer,
                    effective_target,
                    counter,
                ),
            }

        return {
            "action": "REJECT",
            "reasoning": (
                "The offer is below the minimum price we can accept."
            ),
        }

    @staticmethod
    def _counter_reason(
        flexibility: str,
        buyer_offer: float,
        effective_target: float,
        counter_price: float,
    ) -> str:
        if flexibility == "firm":
            approach = "our firm approach keeps the counter at our target"
        elif flexibility == "flexible":
            approach = "our flexible approach splits the gap between your offer and our target"
        else:
            approach = "our moderate approach moves 60% of the gap toward our target"
        return (
            f"Your offer of ${buyer_offer:.2f} is below our target of "
            f"${effective_target:.2f} per unit; {approach}. "
            f"Our counter is ${counter_price:.2f} per unit."
        )

    def _decide_with_llm(
        self,
        buyer_offer: float,
        pricing: PricingStrategy,
        quantity: int,
        session_id: str,
    ) -> dict:
        session = self.active_sessions[session_id]
        turn_count = session["turn_count"]

        effective_target = pricing.target_price
        if quantity >= pricing.volume_discount_threshold:
            effective_target *= (1 - pricing.volume_discount_rate)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a sales negotiation agent for {self.company_name}.

NEGOTIATION STYLE: {pricing.flexibility}
- firm: Rarely budge on price, only small concessions
- moderate: Willing to negotiate, but protect margins
- flexible: Eager to close deals, can make larger concessions

PRODUCT: {session['product']}
QUANTITY: {quantity} units

YOUR CONSTRAINTS:
- Base/listed price: ${pricing.base_price}/unit
- Target price (what you want): ${effective_target}/unit
- Reservation price (absolute minimum): ${pricing.reservation_price}/unit

BUYER'S OFFER: ${buyer_offer}/unit
NEGOTIATION TURN: {turn_count}/10

RULES:
1. If buyer_offer >= target_price: ACCEPT
2. If buyer_offer >= reservation_price: COUNTER
3. If buyer_offer < reservation_price: REJECT or make firm counter at reservation

If COUNTER, suggest a price between buyer's offer and your target.
As turns progress (turn 5+), be more willing to accept buyer's price if it's above reservation.

Respond ONLY with valid JSON in this exact format:
{{{{"action": "ACCEPT", "reasoning": "Brief buyer-facing explanation"}}}} OR
{{{{"action": "COUNTER", "counter_price": 85.50, "reasoning": "Brief buyer-facing explanation"}}}} OR
{{{{"action": "REJECT", "reasoning": "Brief buyer-facing explanation"}}}}
Explain the decision briefly using the offer, target, order size, and negotiation
style as relevant. Do not reveal the reservation price or provide hidden
chain-of-thought; give only a concise explanation suitable for the buyer.
"""),
            ("human", "What is your decision?"),
        ])

        response = self.llm.invoke(prompt.format_messages())
        response_metadata = getattr(response, "response_metadata", {})
        self.llm_usage_by_session.setdefault(session_id, []).append({
            "model": (
                response_metadata.get("model_name")
                or getattr(self.llm, "model", None)
                or getattr(self.llm, "model_name", None)
            ),
            "usage_metadata": getattr(response, "usage_metadata", None),
        })

        try:
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                if not isinstance(decision, dict):
                    raise ValueError("Seller LLM decision must be a JSON object")
                reasoning = decision.get("reasoning")
                if not isinstance(reasoning, str) or not reasoning.strip():
                    decision["reasoning"] = (
                        f"Decision based on your ${buyer_offer:.2f} offer, "
                        f"the order quantity, and our negotiation criteria."
                    )
                return decision
            print("   ⚠️ LLM response malformed, using rules")
            return self._decide_with_rules(buyer_offer, pricing, quantity)
        except (json.JSONDecodeError, ValueError):
            print("   ⚠️ Could not parse LLM decision, using rules")
            return self._decide_with_rules(buyer_offer, pricing, quantity)

    def _create_counter_message(
        self,
        original: NegotiationMessage,
        counter_price: float,
        reasoning: str,
    ) -> NegotiationMessage:
        session = self.active_sessions.get(original.session_id)
        if session:
            session["last_seller_counter"] = counter_price
            session["last_seller_reasoning"] = reasoning

        return NegotiationMessage(
            message_type="COUNTER",
            session_id=original.session_id,
            from_agent=self.agent_id,
            to_agent=original.from_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "product": original.payload["product"],
                "quantity": original.payload["quantity"],
                "counter_price_per_unit": counter_price,
                "currency": "USD",
                "reasoning": reasoning,
                "delivery_days": 14,
                "payment_terms": "Net 30",
            },
        )

    def _create_accept_message(
        self,
        original: NegotiationMessage,
        agreed_price: float,
        reasoning: str,
    ) -> NegotiationMessage:
        quantity = original.payload["quantity"]
        return NegotiationMessage(
            message_type="ACCEPT",
            session_id=original.session_id,
            from_agent=self.agent_id,
            to_agent=original.from_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "agreed_price_per_unit": agreed_price,
                "agreed_quantity": quantity,
                "total_amount": agreed_price * quantity,
                "currency": "USD",
                "reasoning": reasoning,
                "order_id": f"ORD-{self.agent_id}-{original.session_id[:8]}",
                "next_step": "payment",
            },
        )

    def _create_reject_message(self, original: NegotiationMessage, reason: str = "Offer not acceptable") -> NegotiationMessage:
        return NegotiationMessage(
            message_type="REJECT",
            session_id=original.session_id,
            from_agent=self.agent_id,
            to_agent=original.from_agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "reason": reason,
                "message": "Thank you for your interest. We cannot proceed with this offer.",
            },
        )

    def _get_pricing_for_product(self, product: str, quantity: int) -> Optional[PricingStrategy]:
        if product in self.product_catalog:
            return self.product_catalog[product]

        product_lower = product.lower()
        for catalog_product, pricing in self.product_catalog.items():
            if product_lower in catalog_product.lower() or catalog_product.lower() in product_lower:
                return pricing

        return None


def create_seller_agent_for_supplier(supplier_data: dict) -> SellerAgent:
    company_name = supplier_data["company_name"]
    product_name = supplier_data["product_name"]
    listed_price = supplier_data.get("price_numeric", 100.0)

    pricing_strategy = PricingStrategy(
        base_price=listed_price,
        reservation_price=listed_price * 0.90,
        target_price=listed_price * 0.95,
        volume_discount_threshold=100,
        volume_discount_rate=0.05,
        flexibility="moderate",
    )

    agent_id = f"seller_{company_name.replace(' ', '_').lower()}_{abs(hash(company_name)) % 10000}"

    return SellerAgent(
        agent_id=agent_id,
        company_name=company_name,
        product_catalog={
            product_name: pricing_strategy,
            "default": pricing_strategy,
        },
        negotiation_style="moderate",
        # use_llm=False,
        use_llm=True,
    )


if __name__ == "__main__":
    test_supplier = {
        "company_name": "Amazon",
        "product_name": "Crucial P3 Plus 1TB PCIe 4.0 NVMe M.2 SSD",
        "price_numeric": 79.99,
    }

    seller = create_seller_agent_for_supplier(test_supplier)
    print(json.dumps(seller.get_agent_card(), indent=2))
