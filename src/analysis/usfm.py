"""USFM loading helpers for pragmatic analysis commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from dataset.usfm import build_verse_lookup, normalize_biblical_language, parse_usfm_text, resolve_usfm_file

from .constants import DEFAULT_LANG_ROOT


def load_analysis_scripture_data(
    *,
    book: str,
    chapter: int | None,
    translation_language: str,
    biblical_language: str,
    usfm_root: Path = DEFAULT_LANG_ROOT,
) -> tuple[
    dict[tuple[int, int], str],
    dict[tuple[int, int], str],
    list[dict[str, Any]],
    Path,
    Path,
    str,
    str,
]:
    normalized_biblical_language = normalize_biblical_language(biblical_language)

    translation_path = resolve_usfm_file(usfm_root, translation_language, book)
    biblical_path = resolve_usfm_file(usfm_root, normalized_biblical_language, book)

    translation_usfm = translation_path.read_text(encoding="utf-8")
    biblical_usfm = biblical_path.read_text(encoding="utf-8")

    translation_parsed = parse_usfm_text(translation_usfm)
    biblical_parsed = parse_usfm_text(biblical_usfm)

    translation_lookup, translation_records = build_verse_lookup(translation_parsed, book=book, chapter=chapter)
    biblical_lookup, biblical_records = build_verse_lookup(biblical_parsed, book=book, chapter=chapter)

    available_refs = {
        (row["chapter"], row["verse"])
        for row in translation_records
    } & {
        (row["chapter"], row["verse"])
        for row in biblical_records
    }

    verse_records: list[dict[str, Any]] = []
    for chapter_num, verse_num in sorted(available_refs):
        verse_records.append(
            {
                "book": book.upper(),
                "chapter": chapter_num,
                "verse": verse_num,
                "reference": f"{book.upper()} {chapter_num}:{verse_num}",
                "translation_text": translation_lookup.get((chapter_num, verse_num), ""),
                "biblical_text": biblical_lookup.get((chapter_num, verse_num), ""),
            }
        )

    if not verse_records:
        chapter_text = f" chapter {chapter}" if chapter is not None else ""
        raise click.ClickException(f"No overlapping verses found for {book.upper()}{chapter_text}.")

    return (
        translation_lookup,
        biblical_lookup,
        verse_records,
        translation_path,
        biblical_path,
        translation_usfm,
        biblical_usfm,
    )
