"""Helpers for loading BART discourse-analysis annotations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import click


def load_bart_annotations_for_verse(
    bart_db_path: Path,
    *,
    book: str,
    chapter: int,
    verse: int,
) -> list[dict[str, Any]]:
    if not bart_db_path.exists():
        raise click.ClickException(f"BART annotations database not found: {bart_db_path}")

    query = """
        SELECT
            BookCode,
            Chapter,
            Verse,
            Greek_Text,
            Reference,
            OSIS_Ref,
            Annotation_Name,
            Description,
            Type,
            Label
        FROM annotations
        WHERE BookCode = ? AND Chapter = ? AND Verse = ?
        ORDER BY rowid
    """

    with sqlite3.connect(str(bart_db_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (book.upper(), int(chapter), int(verse))).fetchall()
        return [dict(row) for row in rows]


def enrich_verse_records_with_bart_annotations(
    verse_records: list[dict[str, Any]],
    bart_db_path: Path | None,
) -> list[dict[str, Any]]:
    if bart_db_path is None:
        return verse_records

    enriched: list[dict[str, Any]] = []
    for verse_record in verse_records:
        record = dict(verse_record)
        record["bart_annotations"] = load_bart_annotations_for_verse(
            bart_db_path,
            book=str(record["book"]),
            chapter=int(record["chapter"]),
            verse=int(record["verse"]),
        )
        enriched.append(record)
    return enriched
