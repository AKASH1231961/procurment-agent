"""LangChain agent for monitoring inventory shortages."""

import argparse
import importlib.util
import json
import math
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

from websearch import search_google_shopping, search_wholesale_suppliers
from seller_agent import create_seller_agent_for_supplier
from buyer_negotiation import (
    DEFAULT_NEGOTIATION_LOG,
    get_agreement_metrics,
    log_token_usage,
    resolve_buyer_terms,
    run_buyer_negotiation,
)


DATABASE_MODULE_PATH = Path(__file__).with_name("Inventory_db .py")


def _load_database_module():
    spec = importlib.util.spec_from_file_location(
        "inventory_database",
        DATABASE_MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {DATABASE_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@tool
def check_inventory() -> str:
    """Find items at or below their reorder threshold and open restock tasks."""
    database = _load_database_module()

    # Create procurement tasks for shortages that do not already have one.
    database.get_open_restock_needs()

    with database.get_conn() as connection:
        shortages = connection.execute(
            """
            SELECT
                items.*,
                procurement_tasks.task_id
            FROM items
            LEFT JOIN procurement_tasks
                ON procurement_tasks.item_id = items.item_id
                AND procurement_tasks.status NOT IN (
                    'settled',
                    'failed',
                    'cancelled'
                )
            WHERE items.current_stock <= items.reorder_threshold
            ORDER BY items.item_id
            """
        ).fetchall()

    if not shortages:
        return "Inventory check complete: no items are currently short."

    notifications = [
        (
            f"SHORTAGE: {item['name']} ({item['sku']}) has reached its "
            f"reorder threshold. Active procurement task: "
            f"{item['task_id'] or 'none'}."
        )
        for item in shortages
    ]

    return "Inventory alert:\n" + "\n".join(notifications)


@tool
def find_suppliers_for_restock(
    product: str,
    task_id: int,
    max_results: int = 10,
) -> str:
    """Find online wholesale sellers and save them for a restock task.

    Call this after check_inventory reports a shortage. Use the exact product
    name and active procurement task ID returned by check_inventory.
    """
    database = _load_database_module()
    database.clear_candidates(task_id)
    # Inventory descriptions often include the SKU, but product searches are
    # more reliable when the SKU is not sent as part of the query.
    search_product = re.sub(r"\s*\([^)]*\)", "", product).strip()
    raw_results = search_wholesale_suppliers.invoke({
        "product": search_product,
        "max_results": max_results,
    })
    results = json.loads(raw_results)

    if isinstance(results, dict) and results.get("error"):
        return f"Supplier search failed: {results['error']}"
    if not results:
        raw_results = search_google_shopping.invoke({
            "product": search_product,
            "max_results": max_results,
        })
        results = json.loads(raw_results)

    if isinstance(results, dict) and results.get("error"):
        return f"Supplier search failed: {results['error']}"
    if not results:
        return f"No online suppliers found for {search_product}."

    saved = []
    for supplier in results:
        database.log_candidate_supplier(
            task_id=task_id,
            supplier_name=supplier.get("company_name", "Unknown supplier"),
            url=supplier.get("url"),
            estimated_price=supplier.get("price_numeric"),
            currency=supplier.get("currency", "USD"),
            snippet=(
                f"{supplier.get('product_name', product)}; "
                f"price: {supplier.get('price', 'Not available')}; "
                f"source: web search"
            ),
        )
        saved.append(supplier)

    return json.dumps({
        "task_id": task_id,
        "product": search_product,
        "suppliers_found": len(saved),
        "suppliers": saved,
    }, indent=2)


@tool
def negotiate_with_first_supplier(
    product: str,
    task_id: int,
    max_price: float,
    quantity: int = 50,
    initial_offer: float | None = None,
    max_rounds: int = 5,
) -> str:
    """Negotiate with the cheapest candidate within the buyer's authorized limit.

    max_price must come from the user's instruction or an explicit procurement
    policy. When supplied, omitted quantity defaults to 50 and initial_offer
    defaults to 90 percent of max_price.
    """
    quantity, initial_offer = resolve_buyer_terms(
        max_price=max_price,
        quantity=quantity,
        initial_offer=initial_offer,
    )
    database = _load_database_module()
    candidates = database.get_candidates(task_id)

    if not candidates:
        return f"No supplier candidates found for task {task_id}. Run find_suppliers_for_restock first."

    best_candidate = sorted(
        candidates,
        key=lambda candidate: candidate.get("estimated_price") or float("inf"),
    )[0]

    supplier_name = best_candidate.get("supplier_name", "Unknown supplier")
    listed_price = float(best_candidate.get("estimated_price") or initial_offer)
    seller = create_seller_agent_for_supplier({
        "company_name": supplier_name,
        "product_name": product,
        "price_numeric": listed_price,
    })

    session_id = f"task-{task_id}-supplier-{best_candidate.get('candidate_id')}"
    negotiation = run_buyer_negotiation(
        seller=seller,
        product=product,
        quantity=quantity,
        initial_offer=float(initial_offer),
        max_price=float(max_price),
        max_rounds=max_rounds,
        session_id=session_id,
        strategy_model=model,
        benchmark_price=(
            float(best_candidate["estimated_price"])
            if best_candidate.get("estimated_price") is not None
            and math.isfinite(float(best_candidate["estimated_price"]))
            and float(best_candidate["estimated_price"]) > 0
            else None
        ),
    )
    database.clear_candidates(task_id)
    return json.dumps({
        "task_id": task_id,
        "supplier": supplier_name,
        "seller_agent_id": seller.agent_id,
        "session_id": session_id,
        "initial_offer": float(initial_offer),
        "max_price": float(max_price),
        "max_rounds": max_rounds,
        "negotiation": negotiation,
        "agreement_metrics": get_agreement_metrics(),
    }, indent=2)


@tool
def get_negotiation_agreement_metrics() -> str:
    """Report the confirmed agreement rate across completed negotiations."""
    return json.dumps(get_agreement_metrics(), indent=2)


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing")


model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    groq_api_key=api_key,
)

inventory_monitor = create_agent(
    model=model,
    tools=[
        check_inventory,
        find_suppliers_for_restock,
        negotiate_with_first_supplier,
        get_negotiation_agreement_metrics,
    ],
    system_prompt=(
        "You are an inventory monitoring agent. "
        "Always run check_inventory first. For every shortage it reports, "
        "call find_suppliers_for_restock using the shortage item name and "
        "active procurement task ID. After suppliers are found, choose the best "
        "candidate and call negotiate_with_first_supplier only when the user or an "
        "explicit procurement policy provides an authorized maximum price per unit. "
        "Use any supplied opening offer and quantity exactly. If either is omitted, "
        "use the tool's documented defaults: 90% of max_price for the opening offer "
        "and 50 units for quantity. Never invent a maximum price or supplier "
        "information. If the maximum price is missing, do not negotiate. Report the "
        "shortage, supplier list, and full negotiation outcome."
    ),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check inventory and discover suppliers; optionally negotiate."
    )
    parser.add_argument(
        "--max-price",
        type=float,
        help="Authorized maximum price per unit. Required to enable negotiation.",
    )
    parser.add_argument(
        "--initial-offer",
        type=float,
        help="Opening offer per unit (defaults to 90%% of --max-price).",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=50,
        help="Quantity to order if negotiating (default: 50).",
    )
    args = parser.parse_args()

    if args.initial_offer is not None and args.max_price is None:
        parser.error("--initial-offer requires --max-price")

    negotiation_context = ""
    if args.max_price is not None:
        quantity, initial_offer = resolve_buyer_terms(
            max_price=args.max_price,
            quantity=args.quantity,
            initial_offer=args.initial_offer,
        )
        negotiation_context = (
            f" The user explicitly authorized a maximum price of "
            f"${args.max_price:.2f} per unit, an opening offer of "
            f"${initial_offer:.2f} per unit, and a quantity of {quantity}. "
            "Use exactly these negotiation terms."
        )
    else:
        negotiation_context = (
            " Do not start a negotiation because no authorized maximum price "
            "was supplied. Report the shortage and supplier candidates only."
        )

    inventory_run_id = f"inventory-run-{uuid.uuid4()}"
    result = inventory_monitor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check the inventory, identify shortages, and find "
                        "online suppliers for each item that needs restocking."
                        + negotiation_context
                    ),
                }
            ]
        }
    )

    llm_call_number = 0
    for message in result["messages"]:
        if getattr(message, "type", None) != "ai":
            continue
        llm_call_number += 1
        response_metadata = getattr(message, "response_metadata", {})
        log_token_usage(
            DEFAULT_NEGOTIATION_LOG,
            agent="buyer_agent",
            model=(
                response_metadata.get("model_name")
                or getattr(model, "model", None)
                or getattr(model, "model_name", None)
            ),
            usage_metadata=getattr(message, "usage_metadata", None),
            session_id=f"{inventory_run_id}-call-{llm_call_number}",
        )

    # print(result["messages"][-1].content)
    for message in result["messages"]:
        print("\n" + "=" * 80)
        print(message.type.upper())
        print("=" * 80)
        print(message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool calls:")
            for tool_call in message.tool_calls:
                print(tool_call)
