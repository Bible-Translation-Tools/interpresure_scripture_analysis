"""Helpers for loading MACULA verse tokens.

Supports both Greek (``macula_greek`` table) and Hebrew (``macula_hebrew``
table) in the same SQLite database or in separate database files.

Greek MACULA (NT):
    Source: https://github.com/Clear-Bible/macula-greek
    Already available as SQLite in the project.

Hebrew MACULA (OT):
    Source: https://github.com/Clear-Bible/macula-hebrew
    Convert from TSV with:
        python interpresure/artifacts/convert_macula_hebrew_tsv_to_db.py \\
            macula_hebrew.tsv macula_hebrew.db

Both use the same U23003 word-level ref format (e.g. ``GEN 1:1!1``), so the
verse-lookup pattern is identical for both languages.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import click


_LANGUAGE_TABLE: dict[str, str] = {
    "grc": "macula_greek",
    "greek": "macula_greek",
    "heb": "macula_hebrew",
    "hebrew": "macula_hebrew",
}


def _verse_ref_pattern(book: str, chapter: int, verse: int) -> str:
    return f"{book.upper()} {chapter}:{verse}!%"


def _resolve_table(biblical_language: str, db_path: Path) -> str:
    """Return the MACULA table name for *biblical_language*.

    Falls back to auto-detection if the language code is not in the registry —
    useful for single-language databases without needing to pass a language tag.
    """
    lang_lower = biblical_language.strip().lower()
    if lang_lower in _LANGUAGE_TABLE:
        return _LANGUAGE_TABLE[lang_lower]

    # Auto-detect: use whichever MACULA table exists in the database
    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    for candidate in ("macula_greek", "macula_hebrew"):
        if candidate in tables:
            return candidate

    raise click.ClickException(
        f"Could not find a MACULA table in {db_path}. "
        "Expected 'macula_greek' or 'macula_hebrew'."
    )


def load_macula_tokens_for_verse(
    macula_db_path: Path,
    *,
    book: str,
    chapter: int,
    verse: int,
    biblical_language: str = "grc",
) -> list[dict[str, Any]]:
    """Load MACULA word-level tokens for a single verse.

    Selects all columns from the appropriate table so callers receive the full
    metadata available for that language — Greek and Hebrew have different
    morphological fields.
    """
    if not macula_db_path.exists():
        raise click.ClickException(f"MACULA database not found: {macula_db_path}")

    table = _resolve_table(biblical_language, macula_db_path)
    pattern = _verse_ref_pattern(book, chapter, verse)

    query = f"""
        SELECT *
        FROM {table}
        WHERE ref LIKE ?
        ORDER BY rowid
    """

    with sqlite3.connect(str(macula_db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (pattern,)).fetchall()
        return [dict(row) for row in rows]


def enrich_verse_records_with_macula_tokens(
    verse_records: list[dict[str, Any]],
    macula_db_path: Path | None,
    biblical_language: str = "grc",
) -> list[dict[str, Any]]:
    """Attach ``macula_tokens`` to each verse record.

    No-op when *macula_db_path* is ``None``.
    The *biblical_language* parameter selects the correct table.
    """
    if macula_db_path is None:
        return verse_records

    enriched: list[dict[str, Any]] = []
    for verse_record in verse_records:
        record = dict(verse_record)
        record["macula_tokens"] = load_macula_tokens_for_verse(
            macula_db_path,
            book=str(record["book"]),
            chapter=int(record["chapter"]),
            verse=int(record["verse"]),
            biblical_language=biblical_language,
        )
        enriched.append(record)
    return enriched
