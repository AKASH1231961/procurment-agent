"""SQLite FTS retrieval for procurement and vendor-negotiation advice."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_PATH = (
    Path(__file__).with_name("corpus") / "negotiation-strategy-sft-100k.jsonl"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agentic_procurement"
RELEVANT_CATEGORIES = {"procurement", "vendor_negotiation"}


def retrieve_procurement_examples(
    query: str,
    limit: int = 3,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    cache_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Return the most relevant procurement/vendor examples from the SFT corpus."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Negotiation corpus not found: {corpus_path}")
    fts_query = _make_fts_query(query)
    if not fts_query:
        return []

    if cache_path is None:
        corpus_key = hashlib.sha256(
            str(corpus_path.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = DEFAULT_CACHE_DIR / f"negotiation-strategy-{corpus_key}.sqlite3"
    cache_path = Path(cache_path)
    _ensure_index(corpus_path, cache_path)

    with sqlite3.connect(cache_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT example_id, category, context, prompt, advice
            FROM examples
            WHERE examples MATCH ?
              AND category IN (?, ?)
            ORDER BY bm25(examples, 0, 0, 1, 2, 0)
            """,
            (
                fts_query,
                "procurement",
                "vendor_negotiation",
            ),
        )
        results = []
        seen_examples = set()
        for row in rows:
            identity = (row["category"], row["prompt"], row["advice"])
            if identity in seen_examples:
                continue
            seen_examples.add(identity)
            results.append({
                "id": row["example_id"],
                "category": row["category"],
                "context": row["context"],
                "scenario": row["prompt"],
                "advice": row["advice"],
            })
            if len(results) == limit:
                break
    return results


def _make_fts_query(text: str) -> str:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text.lower())
    terms = list(dict.fromkeys(terms))[:40]
    return " OR ".join(f'"{term}"' for term in terms)


def _ensure_index(corpus_path: Path, cache_path: Path) -> None:
    source_stat = corpus_path.stat()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _index_is_current(
        cache_path,
        corpus_path,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        return

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        _build_index(
            corpus_path,
            temporary_path,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _index_is_current(
    cache_path: Path,
    corpus_path: Path,
    source_size: int,
    source_mtime_ns: int,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with sqlite3.connect(cache_path) as connection:
            metadata = dict(
                connection.execute("SELECT key, value FROM corpus_metadata").fetchall()
            )
    except sqlite3.Error:
        return False
    return metadata == {
        "corpus_path": str(corpus_path.resolve()),
        "source_size": str(source_size),
        "source_mtime_ns": str(source_mtime_ns),
    }


def _build_index(
    corpus_path: Path,
    index_path: Path,
    source_size: int,
    source_mtime_ns: int,
) -> None:
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            """
            CREATE VIRTUAL TABLE examples USING fts5(
                category UNINDEXED,
                context UNINDEXED,
                prompt,
                advice,
                example_id UNINDEXED
            )
            """
        )
        connection.execute(
            "CREATE TABLE corpus_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO corpus_metadata (key, value) VALUES (?, ?)",
            [
                ("corpus_path", str(corpus_path.resolve())),
                ("source_size", str(source_size)),
                ("source_mtime_ns", str(source_mtime_ns)),
            ],
        )

        batch: list[tuple[str, str, str, str, str]] = []
        with corpus_path.open("r", encoding="utf-8") as corpus_file:
            for line_number, line in enumerate(corpus_file, start=1):
                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {corpus_path} at line {line_number}"
                    ) from error

                metadata = record.get("metadata") or {}
                category = metadata.get("negotiation_type", "")
                if category not in RELEVANT_CATEGORIES:
                    continue

                conversations = record.get("conversations") or []
                prompt = next(
                    (
                        str(turn.get("value", ""))
                        for turn in conversations
                        if turn.get("from") == "human"
                    ),
                    "",
                )
                advice = next(
                    (
                        str(turn.get("value", ""))
                        for turn in conversations
                        if turn.get("from") in {"gpt", "assistant"}
                    ),
                    "",
                )
                if not prompt or not advice:
                    continue

                batch.append(
                    (
                        category,
                        str(metadata.get("context", "")),
                        prompt,
                        advice,
                        str(record.get("id", "")),
                    )
                )
                if len(batch) >= 500:
                    connection.executemany(
                        """
                        INSERT INTO examples
                            (category, context, prompt, advice, example_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()

        if batch:
            connection.executemany(
                """
                INSERT INTO examples
                    (category, context, prompt, advice, example_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )
