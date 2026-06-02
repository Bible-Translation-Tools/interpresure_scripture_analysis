"""Agent tools for the pragmatic analysis pipeline.

Tools are created via factory functions so they can close over run-specific
configuration (USFM root, language settings) without global mutable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from data.interpresure import Interpresure
from dataset.usfm import (
    build_verse_lookup,
    normalize_biblical_language,
    parse_usfm_text,
    resolve_usfm_file,
)

from .constants import DEFAULT_LANG_ROOT


def make_verse_lookup_tool(
    *,
    usfm_root: Path = DEFAULT_LANG_ROOT,
    translation_language: str,
    biblical_language: str,
):
    """Return a verse-range lookup tool bound to the given run configuration.

    The returned function is decorated with ``@tool`` so Agent Framework can
    infer its JSON schema and register it on an agent.  The tool is grounded
    entirely in the project's local USFM data — it never calls any external
    service — which prevents the model from hallucinating text from other
    manuscripts or translations.

    InterpreSure annotations are returned when available for the requested
    chapter; the tool degrades gracefully to text-only when they are not.
    """
    # Lazy import so the agent_framework dependency is only required at runtime.
    from agent_framework import tool  # type: ignore[import]

    normalized_biblical = normalize_biblical_language(biblical_language)

    @tool
    def lookup_verse_range(
        book: Annotated[str, Field(description="USFM book code, e.g. PHM, PHP, PSA, GEN, ISA.")],
        chapter: Annotated[int, Field(description="Chapter number.", ge=1)],
        verse_start: Annotated[int, Field(description="First verse of the range.", ge=1)],
        verse_end: Annotated[
            int | None,
            Field(description="Last verse of the range (inclusive). Omit for a single verse."),
        ] = None,
    ) -> str:
        """Retrieve translation and biblical-language text for a verse or verse range.

        Returns the exact wording from the project's USFM files for the
        requested translation and biblical language.  Use this tool whenever
        you want to verify a cross-reference to another verse, chapter, or
        book rather than relying on memory.  InterpreSure expert annotations
        are included when available for the requested chapter.
        """
        book_upper = book.strip().upper()
        end = verse_end if verse_end is not None else verse_start
        if end < verse_start:
            end = verse_start

        # --- Load translation verses ---
        try:
            t_path = resolve_usfm_file(usfm_root, translation_language, book_upper)
            t_lookup, _ = build_verse_lookup(
                parse_usfm_text(t_path.read_text(encoding="utf-8")),
                book=book_upper,
                chapter=chapter,
            )
        except Exception as exc:
            return f"[Error loading translation for {book_upper} {chapter}: {exc}]"

        # --- Load biblical-language verses ---
        try:
            b_path = resolve_usfm_file(usfm_root, normalized_biblical, book_upper)
            b_lookup, _ = build_verse_lookup(
                parse_usfm_text(b_path.read_text(encoding="utf-8")),
                book=book_upper,
                chapter=chapter,
            )
        except Exception as exc:
            b_lookup = {}

        # --- Build text output ---
        lines: list[str] = [
            f"# {book_upper} {chapter}:{verse_start}"
            + (f"–{end}" if end != verse_start else "")
        ]

        for v in range(verse_start, end + 1):
            t_text = t_lookup.get((chapter, v), "")
            b_text = b_lookup.get((chapter, v), "")
            if not t_text and not b_text:
                continue
            lines.append(f"\n**{book_upper} {chapter}:{v}**")
            if b_text:
                lines.append(f"Original: {b_text}")
            if t_text:
                lines.append(f"Translation: {t_text}")

        if len(lines) == 1:
            return f"[No verses found for {book_upper} {chapter}:{verse_start}–{end}]"

        # --- Append InterpreSure annotations if available ---
        try:
            interp = Interpresure(book_upper, chapter)
            for v in range(verse_start, end + 1):
                ann = interp.get_annotations_markdown(None, chapter, v)
                if ann and ann.strip():
                    lines.append(ann)
        except (KeyError, Exception):
            pass  # No annotations for this chapter — that's fine

        return "\n".join(lines)

    return lookup_verse_range


def format_verse_records_as_chapter_text(
    verse_records: list[dict[str, Any]],
    *,
    biblical_language: str = "Greek",
) -> str:
    """Format the full chapter verse records into a readable string for context injection."""
    lang_label = biblical_language.capitalize()
    lines: list[str] = []
    for rec in verse_records:
        ref = rec.get("reference", f"{rec.get('book','')} {rec.get('chapter','')}:{rec.get('verse','')}")
        b_text = rec.get("biblical_text", "")
        t_text = rec.get("translation_text", "")
        lines.append(f"\n**{ref}**")
        if b_text:
            lines.append(f"{lang_label}: {b_text}")
        if t_text:
            lines.append(f"Translation: {t_text}")
    return "\n".join(lines)
