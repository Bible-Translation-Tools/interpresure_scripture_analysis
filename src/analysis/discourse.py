"""Global discourse pass for a chapter.

Runs before the verse-by-verse loop.  Produces a ``DiscourseMapObservation``
that anchors all subsequent verse analyses with a shared understanding of the
chapter's QUDs, argument structure, discourse boundaries, and relational dynamics.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from .schemas import DiscourseMapObservation


def _build_discourse_prompt(
    *,
    verse_records: list[dict[str, Any]],
    biblical_language: str,
    interpresure_markdown: str | None = None,
    bart_summary: str | None = None,
) -> str:
    """Build the prompt for the global discourse pass."""
    lang_label = biblical_language.capitalize()
    lines: list[str] = [
        "# Chapter Text\n",
        f"Analyze the following chapter.  The original language is {lang_label}.\n",
    ]

    for rec in verse_records:
        ref = rec.get("reference", "")
        b_text = rec.get("biblical_text", "")
        t_text = rec.get("translation_text", "")
        lines.append(f"\n**{ref}**")
        if b_text:
            lines.append(f"{lang_label}: {b_text}")
        if t_text:
            lines.append(f"Translation: {t_text}")

    if interpresure_markdown and interpresure_markdown.strip():
        lines.extend([
            "\n\n# Expert InterpreSure Annotations (all verses)\n",
            interpresure_markdown,
        ])

    if bart_summary and bart_summary.strip():
        lines.extend([
            "\n\n# BART Discourse Annotations\n",
            bart_summary,
        ])

    lines.extend([
        "\n\n# Task\n",
        dedent("""\
            Produce a structured discourse map for this chapter.  Your map will be
            used to anchor all subsequent verse-level analyses, so be precise and
            comprehensive.

            Identify discourse unit boundaries carefully — these will be used to
            inject structural markers into the verse-by-verse analysis loop.  Each
            boundary should correspond to a genuine shift in topic, QUD, or rhetorical
            function, not simply a paragraph break.

            Return your response as valid JSON matching the required schema.
        """).strip(),
    ])

    return "\n".join(lines)


def _collect_all_annotations_markdown(
    verse_records: list[dict[str, Any]],
) -> str:
    """Collect any pre-attached ``pragmatic_annotations`` fields from verse records."""
    parts: list[str] = []
    for rec in verse_records:
        ann = rec.get("pragmatic_annotations", "")
        if ann and isinstance(ann, str) and ann.strip():
            parts.append(ann.strip())
    return "\n\n".join(parts) if parts else ""


def _collect_bart_summary(verse_records: list[dict[str, Any]]) -> str:
    """Collect BART annotation summaries from verse records if present."""
    parts: list[str] = []
    for rec in verse_records:
        bart = rec.get("bart_annotations")
        if bart:
            ref = rec.get("reference", "")
            import json
            try:
                bart_str = json.dumps(bart, ensure_ascii=False, indent=2, default=str)
            except Exception:
                bart_str = str(bart)
            parts.append(f"**{ref}**\n{bart_str}")
    return "\n\n".join(parts) if parts else ""


async def run_discourse_pass(
    *,
    verse_records: list[dict[str, Any]],
    biblical_language: str,
    discourse_agent,
) -> DiscourseMapObservation:
    """Run the global discourse pass and return a structured ``DiscourseMapObservation``.

    The discourse agent is called statelessly (no session) — this is a one-shot
    structured output request.

    Args:
        verse_records: Enriched verse records for the chapter (may include
            ``pragmatic_annotations``, ``bart_annotations``, ``macula_tokens``).
        biblical_language: ``"grc"`` or ``"heb"``, used for display labels.
        discourse_agent: An Agent Framework ``Agent`` configured for the discourse
            pass (see ``agents.make_discourse_agent``).

    Returns:
        A validated ``DiscourseMapObservation`` Pydantic model.
    """
    annotations_md = _collect_all_annotations_markdown(verse_records)
    bart_summary = _collect_bart_summary(verse_records)

    prompt = _build_discourse_prompt(
        verse_records=verse_records,
        biblical_language=biblical_language,
        interpresure_markdown=annotations_md or None,
        bart_summary=bart_summary or None,
    )

    result = await discourse_agent.run(prompt)
    raw = result.text.strip() if hasattr(result, "text") else str(result).strip()

    # Strip markdown code fences if the model wrapped its JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return DiscourseMapObservation.model_validate_json(raw)
