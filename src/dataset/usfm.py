"""USFM loading and verse lookup helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click

from .constants import DEFAULT_LANG_ROOT, REFERENCE_RE


def normalize_biblical_language(language: str) -> str:
    lower = language.lower()
    if lower in {"grc", "greek"}:
        return "grc"
    if lower in {"heb", "hebrew"}:
        return "heb"
    raise click.ClickException("Biblical language must be Greek or Hebrew.")


def load_usfm_parser():
    try:
        from usfm2dict import UsfmParser

        return UsfmParser()
    except ImportError:
        return None


def fallback_parse_usfm(usfm_text: str) -> dict[str, str]:
    """Parse a minimal subset of USFM when usfm2dict is unavailable."""

    current_book = None
    current_chapter = None
    current_verse = None
    verses: dict[str, str] = {}
    buffer: list[str] = []

    marker_pattern = re.compile(r"(\\id\s+[^\s]+|\\c\s+\d+|\\v\s+\d+)")

    def flush():
        nonlocal buffer, current_book, current_chapter, current_verse
        if current_book and current_chapter is not None and current_verse is not None:
            text = " ".join(part.strip() for part in buffer if part.strip()).strip()
            verses[f"{current_book} {current_chapter}:{current_verse}"] = text
        buffer = []

    for raw_line in usfm_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matches = list(marker_pattern.finditer(line))
        if not matches:
            if current_verse is not None:
                buffer.append(line)
            continue

        cursor = 0
        for match in matches:
            text = line[cursor:match.start()].strip()
            if text and current_verse is not None:
                buffer.append(text)

            marker = match.group(0)
            if marker.startswith("\\id "):
                parts = marker.split(maxsplit=1)
                current_book = parts[1].split()[0]
            elif marker.startswith("\\c "):
                flush()
                current_chapter = int(marker.split()[1])
                current_verse = None
            elif marker.startswith("\\v "):
                flush()
                current_verse = int(marker.split()[1])

            cursor = match.end()

        tail = line[cursor:].strip()
        if tail and current_verse is not None:
            buffer.append(tail)

    flush()
    return verses


def parse_usfm_text(usfm_text: str) -> dict[str, str]:
    parser = load_usfm_parser()
    if parser is not None:
        return parser.parse(usfm_text)
    return fallback_parse_usfm(usfm_text)


def resolve_usfm_file(usfm_root: Path, language: str, book: str) -> Path:
    language_dir = usfm_root / language
    if not language_dir.exists():
        raise click.ClickException(f"USFM language folder not found: {language_dir}")

    patterns = [
        f"{language}_{book}.usfm",
        f"{book}.usfm",
        f"*-{book}.usfm",
        f"*_{book}.usfm",
    ]

    for pattern in patterns:
        matches = sorted(language_dir.glob(pattern))
        if not matches:
            continue
        if len(matches) == 1:
            return matches[0]

        exact_prefixed = [path for path in matches if path.name.startswith(f"{language}_")]
        if exact_prefixed:
            return exact_prefixed[0]

        exact_hyphen = [path for path in matches if path.name.startswith(f"{language}-")]
        if exact_hyphen:
            return exact_hyphen[0]

        return matches[0]

    raise click.ClickException(f"Could not find a USFM file for {language}/{book} under {language_dir}")


def build_verse_lookup(
    parsed_usfm: dict[str, str],
    *,
    book: str,
    chapter: int | None = None,
) -> tuple[dict[tuple[int, int], str], list[dict[str, Any]]]:
    verse_lookup: dict[tuple[int, int], str] = {}
    verse_records: list[dict[str, Any]] = []

    for reference, text in parsed_usfm.items():
        match = REFERENCE_RE.match(reference)
        if not match:
            continue
        if match.group("book").upper() != book.upper():
            continue

        current_chapter = int(match.group("chapter"))
        current_verse = int(match.group("verse"))
        if chapter is not None and current_chapter != chapter:
            continue

        clean_text = " ".join(str(text).split())
        verse_lookup[(current_chapter, current_verse)] = clean_text
        verse_records.append(
            {
                "book": book.upper(),
                "chapter": current_chapter,
                "verse": current_verse,
                "reference": f"{book.upper()} {current_chapter}:{current_verse}",
                "biblical_text": clean_text,
            }
        )

    verse_records.sort(key=lambda item: (item["chapter"], item["verse"]))
    return verse_lookup, verse_records


def load_scripture_data(
    *,
    book: str,
    chapter: int | None,
    biblical_language: str,
    usfm_root: Path = DEFAULT_LANG_ROOT,
) -> tuple[dict[tuple[int, int], str], list[dict[str, Any]], Path]:
    biblical_path = resolve_usfm_file(usfm_root, biblical_language, book)

    biblical_text = biblical_path.read_text(encoding="utf-8")

    biblical_parsed = parse_usfm_text(biblical_text)

    biblical_lookup, biblical_records = build_verse_lookup(
        biblical_parsed, book=book, chapter=chapter
    )
    return biblical_lookup, biblical_records, biblical_path
