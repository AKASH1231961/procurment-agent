import json
import uuid
from typing import Any, Dict, List, Optional

from Inventory_db  import clear_candidates, get_conn, get_candidates
from buyer_negotiation import DEFAULT_NEGOTIATION_LOG, log_token_usage
from seller_agent import NegotiationMessage, create_seller_agent_for_supplier

class ProcurementNegotiationManager:
    """MVP manager for one active supplier negotiation per task."""

    def __init__(self):
        self.active_agent_by_task: Dict[int, Any] = {}
        self.active_session_by_task: Dict[int, str] = {}
        self.active_terms_by_task: Dict[int, Dict[str, Any]] = {}
        self.logged_seller_usage_by_task: Dict[int, int] = {}

    def _log_seller_usage(self, task_id: int, seller_agent: Any) -> None:
        session_id = self.active_session_by_task[task_id]
        usage_events = seller_agent.llm_usage_by_session.get(session_id, [])
        logged_count = self.logged_seller_usage_by_task.get(task_id, 0)
        for usage_event in usage_events[logged_count:]:
            log_token_usage(
                DEFAULT_NEGOTIATION_LOG,
                agent=seller_agent.agent_id,
                model=usage_event["model"],
                usage_metadata=usage_event["usage_metadata"],
                session_id=session_id,
            )
        self.logged_seller_usage_by_task[task_id] = len(usage_events)

    def _get_first_candidate(self, task_id: int) -> Optional[Dict[str, Any]]:
        candidates = get_candidates(task_id)
        if not candidates:
            return None
        return sorted(candidates, key=lambda c: c.get("estimated_price") or float("inf"))[0]

    def start_negotiation_for_task(self, task_id: int, product_name: str, quantity: int, buyer_offer: float):
        candidate = self._get_first_candidate(task_id)
        if not candidate:
            return {
                "status": "failed",
                "reason": "No supplier candidates available for this task.",
            }

        supplier_data = {
            "company_name": candidate.get("supplier_name", "Unknown supplier"),
            "product_name": product_name,
            "price_numeric": float(candidate.get("estimated_price") or buyer_offer),
        }

        seller_agent = create_seller_agent_for_supplier(supplier_data)
        self.active_agent_by_task[task_id] = seller_agent

        session_id = str(uuid.uuid4())
        self.active_session_by_task[task_id] = session_id
        self.logged_seller_usage_by_task[task_id] = 0
        self.active_terms_by_task[task_id] = {
            "product": product_name,
            "quantity": quantity,
        }

        message = NegotiationMessage(
            message_type="PROPOSE",
            session_id=session_id,
            from_agent="buyer_agent",
            to_agent=seller_agent.agent_id,
            timestamp="",
            payload={
                "product": product_name,
                "quantity": quantity,
                "proposed_price_per_unit": buyer_offer,
            },
        )

        response = seller_agent.handle_message(message)
        self._log_seller_usage(task_id, seller_agent)
        if response.message_type in {"ACCEPT", "REJECT", "CONFIRM", "ACKNOWLEDGE"}:
            clear_candidates(task_id)
        return {
            "status": "negotiating",
            "task_id": task_id,
            "supplier": supplier_data["company_name"],
            "session_id": session_id,
            "seller_agent_id": seller_agent.agent_id,
            "response": response.model_dump(),
        }

    def continue_negotiation(self, task_id: int, counter_price: float):
        seller_agent = self.active_agent_by_task.get(task_id)
        session_id = self.active_session_by_task.get(task_id)
        terms = self.active_terms_by_task.get(task_id)

        if not seller_agent or not session_id or not terms:
            return {"status": "failed", "reason": "No active negotiation for this task."}

        message = NegotiationMessage(
            message_type="COUNTER",
            session_id=session_id,
            from_agent="buyer_agent",
            to_agent=seller_agent.agent_id,
            timestamp="",
            payload={
                "product": terms["product"],
                "quantity": terms["quantity"],
                "counter_price_per_unit": counter_price,
            },
        )

        response = seller_agent.handle_message(message)
        self._log_seller_usage(task_id, seller_agent)
        if response.message_type in {"ACCEPT", "REJECT", "CONFIRM", "ACKNOWLEDGE"}:
            clear_candidates(task_id)
        return {
            "status": "negotiating",
            "task_id": task_id,
            "response": response.model_dump(),
        }


if __name__ == "__main__":
    manager = ProcurementNegotiationManager()
    result = manager.start_negotiation_for_task(
        task_id=1,
        product_name="Widget",
        quantity=10,
        buyer_offer=95.0,
    )
    print(json.dumps(result, indent=2))
