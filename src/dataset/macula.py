"""Helpers for loading Macula verse tokens."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import click


def _verse_ref_pattern(book: str, chapter: int, verse: int) -> str:
    return f"{book.upper()} {chapter}:{verse}!%"


def load_macula_tokens_for_verse(
    macula_db_path: Path,
    *,
    book: str,
    chapter: int,
    verse: int,
) -> list[dict[str, Any]]:
    if not macula_db_path.exists():
        raise click.ClickException(f"Macula database not found: {macula_db_path}")

    query = """
        SELECT
            "xml:id" AS xml_id,
            ref,
            role,
            "class",
            "type",
            english,
            mandarin,
            gloss,
            text,
            after,
            lemma,
            normalized,
            strong,
            morph,
            person,
            number,
            gender,
            "case",
            tense,
            voice,
            mood,
            degree,
            domain,
            ln,
            frame,
            subjref,
            referent
        FROM macula_greek
        WHERE ref LIKE ?
        ORDER BY CAST(substr(ref, instr(ref, '!') + 1) AS INTEGER)
    """

    with sqlite3.connect(str(macula_db_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (_verse_ref_pattern(book, chapter, verse),)).fetchall()
        return [dict(row) for row in rows]


def enrich_verse_records_with_macula_tokens(
    verse_records: list[dict[str, Any]],
    macula_db_path: Path | None,
) -> list[dict[str, Any]]:
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
        )
        enriched.append(record)
    return enriched
