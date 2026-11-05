import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_retrieval import retrieve_procurement_examples


def test_retrieval_filters_to_procurement_and_vendor_examples(tmp_path):
    corpus_path = tmp_path / "examples.jsonl"
    index_path = tmp_path / "retrieval.sqlite3"
    records = [
        {
            "id": "procurement-1",
            "metadata": {
                "negotiation_type": "procurement",
                "context": "bulk_order",
            },
            "conversations": [
                {
                    "from": "human",
                    "value": "A supplier quote is above our budget for a bulk order.",
                },
                {
                    "from": "gpt",
                    "value": "Ask whether a larger volume can lower the unit price.",
                },
            ],
        },
        {
            "id": "vendor-1",
            "metadata": {
                "negotiation_type": "vendor_negotiation",
                "context": "price_gap",
            },
            "conversations": [
                {
                    "from": "human",
                    "value": "The vendor counter is too high and the price gap is not closing.",
                },
                {
                    "from": "gpt",
                    "value": "Explore adjustments to terms before increasing the offer.",
                },
            ],
        },
        {
            "id": "real-estate-1",
            "metadata": {
                "negotiation_type": "real_estate",
                "context": "home_purchase",
            },
            "conversations": [
                {
                    "from": "human",
                    "value": "Negotiate the price of a house.",
                },
                {
                    "from": "gpt",
                    "value": "Discuss the inspection and closing date.",
                },
            ],
        },
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    results = retrieve_procurement_examples(
        "supplier counter price gap bulk order",
        limit=3,
        corpus_path=corpus_path,
        cache_path=index_path,
    )

    assert {example["id"] for example in results} == {
        "procurement-1",
        "vendor-1",
    }
    assert all(
        example["category"] in {"procurement", "vendor_negotiation"}
        for example in results
    )
    assert index_path.is_file()
