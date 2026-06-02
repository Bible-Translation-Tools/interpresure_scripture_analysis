"""Analysis generation workflow.

Orchestrates the full pipeline for a single chapter:

  1. Load scripture (USFM) and enrich with InterpreSure / BART / MACULA
     when running in few-shot mode.
  2. Global discourse pass → ``DiscourseMapObservation``.
  3. Inject chapter context into a rolling ``AgentSession``.
  4. Verse-by-verse loop with optional discourse boundary markers.
     Each verse analysis is reviewed by the critic; rejected analyses
     are revised in-session up to ``MAX_CRITIC_ROUNDS``.
  5. Chapter summary via the same session.
  6. Assemble and write run directory in scripture-analysis-api format.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from data.interpresure import Interpresure
from dataset.bart_mcp import enrich_verse_records_with_bart_annotations
from dataset.macula import enrich_verse_records_with_macula_tokens

from .agents import get_analysis_type_config, make_agents, make_discourse_agent
from .constants import (
    DEFAULT_ANALYSIS_TYPE,
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_DISCOURSE_BOUNDARY_MARKERS,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_TRANSLATION_LANGUAGE,
    DEFAULT_TRANSLATION_TITLE,
    MAX_CRITIC_ROUNDS,
    REPO_ID,
    REPO_NAME,
    REPO_ROOT,
)
from .discourse import run_discourse_pass
from .output import RunWriter, build_scope_items
from .schemas import AnalysisRunMetadataObservation, ResourcesUsed, TranslationInfo
from .skills import make_interpresure_skills_provider
from .tools import format_verse_records_as_chapter_text, make_verse_lookup_tool
from .usfm import load_analysis_scripture_data, normalize_biblical_language

MAX_CRITIC_ROUNDS = MAX_CRITIC_ROUNDS  # re-export from constants


# ---------------------------------------------------------------------------
# Verse / chapter analysis helpers
# ---------------------------------------------------------------------------


def _build_verse_prompt(
    verse_record: dict[str, Any],
    biblical_language: str,
) -> str:
    """Build the per-verse analysis prompt."""
    lang_label = biblical_language.capitalize()
    ref = verse_record.get("reference", "")
    b_text = verse_record.get("biblical_text", "")
    t_text = verse_record.get("translation_text", "")
    annotations = verse_record.get("pragmatic_annotations", "")
    macula = verse_record.get("macula_tokens")
    bart = verse_record.get("bart_annotations")

    parts = [f"# Verse Analysis: {ref}\n"]
    parts.append(f"**{lang_label}:** {b_text}")
    parts.append(f"**Translation:** {t_text}")

    if annotations and annotations.strip():
        parts.append(f"\n## Expert InterpreSure Annotations\n{annotations}")

    if macula:
        parts.append(
            "\n## MACULA Syntax Tokens\n"
            + json.dumps(macula, ensure_ascii=False, indent=2, default=str)
        )

    if bart:
        parts.append(
            "\n## BART Discourse Annotations\n"
            + json.dumps(bart, ensure_ascii=False, indent=2, default=str)
        )

    parts.append(
        "\n## Task\n"
        "Analyze this verse for pragmatic fidelity. Work through the analysis "
        "checklist systematically. Cite any cross-references you consult in "
        "the ``cross_references`` field. Return valid JSON matching the required schema."
    )

    return "\n".join(parts)


def _build_chapter_context_prompt(
    verse_records: list[dict[str, Any]],
    discourse_map,
    biblical_language: str,
) -> str:
    """Build the opening message that injects full chapter context into the session."""
    chapter_text = format_verse_records_as_chapter_text(
        verse_records, biblical_language=biblical_language
    )
    discourse_json = discourse_map.model_dump_json(indent=2)

    return dedent(f"""\
        # Chapter Context

        Below is the full chapter text you will be analyzing verse by verse,
        followed by the discourse map for this chapter.  Keep both in mind
        throughout the analysis.

        ## Full Chapter Text

        {chapter_text}

        ## Discourse Map

        ```json
        {discourse_json}
        ```

        The verse-by-verse analysis will now begin.  Await the first verse prompt.
    """)


def _build_chapter_summary_prompt() -> str:
    return dedent("""\
        # Chapter Summary

        You have now analyzed all verses in this chapter.  Produce a chapter-level
        summary observation.

        - Score the chapter as a whole on pragmatic fidelity (1–10).
        - List the verses that need the most translator attention in
          ``verses_to_review``.
        - Summarize the key strengths, weaknesses, and suggestions for the
          chapter overall.
        - ``cross_references`` may remain empty for the summary unless a
          cross-chapter pattern was notable.

        Return valid JSON matching the required schema.
    """)


def _build_boundary_marker(unit) -> str:
    return (
        f"--- Discourse boundary: entering {unit.description} "
        f"(vv. {unit.verse_start}–{unit.verse_end}) ---"
    )


async def _parse_observation(result, schema_cls):
    """Parse an agent result into a Pydantic observation model."""
    raw = result.text.strip() if hasattr(result, "text") else str(result).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return schema_cls.model_validate_json(raw)


async def _analyze_verse_with_critic(
    *,
    analyst,
    critic,
    session,
    verse_record: dict[str, Any],
    biblical_language: str,
    output_schema,
    max_rounds: int,
) -> Any:
    """Run analyst → critic review loop for a single verse.

    Returns the accepted (or last) observation after up to ``max_rounds``
    revision attempts.
    """
    from .schemas import CriticReview

    verse_prompt = _build_verse_prompt(verse_record, biblical_language)

    # Initial analysis (goes into rolling session)
    result = await analyst.run(verse_prompt, session=session)
    observation = await _parse_observation(result, output_schema)

    for round_num in range(max_rounds):
        # Critic reviews statelessly (no session)
        analysis_text = result.text if hasattr(result, "text") else str(result)
        critic_result = await critic.run(
            f"Review the following translation analysis:\n\n{analysis_text}"
        )
        review = await _parse_observation(critic_result, CriticReview)

        if review.accepted:
            print(f"  ✅ Accepted (round {round_num})")
            break

        print(f"  🔄 Revision requested (round {round_num + 1}): {review.reasoning[:80]}…")

        # Revision — stays in session so analyst has full context
        revision_prompt = (
            f"Your analysis was rejected by the Linguistic Critic with the following "
            f"feedback:\n\n{review.reasoning}\n\n"
            "Please revise your analysis accordingly. "
            "Return valid JSON matching the required schema."
        )
        result = await analyst.run(revision_prompt, session=session)
        observation = await _parse_observation(result, output_schema)
    else:
        print(f"  ⚠️  Max revision rounds reached — using last version")

    return observation


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


async def generate_analysis(
    *,
    book: str,
    chapter: int,
    biblical_language: str = DEFAULT_BIBLICAL_LANGUAGE,
    translation_language: str = DEFAULT_TRANSLATION_LANGUAGE,
    translation_title: str = DEFAULT_TRANSLATION_TITLE,
    usfm_root: Path = DEFAULT_LANG_ROOT,
    model: str = DEFAULT_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    macula_db_path: Path | None = None,
    bart_db_path: Path | None = None,
    analysis_mode: str = "zero_shot",
    analysis_type: str = DEFAULT_ANALYSIS_TYPE,
    use_expert_materials: bool = False,
    discourse_boundary_markers: bool = DEFAULT_DISCOURSE_BOUNDARY_MARKERS,
    output_dir: Path,
    repo_id: str = REPO_ID,
    repo_name: str = REPO_NAME,
) -> dict[str, Any]:
    """Run the full pragmatic analysis pipeline for a chapter.

    Returns a dict with keys:
        - ``run_dir``: Path to the written run directory.
        - ``discourse_map``: The ``DiscourseMapObservation`` produced.
        - ``verse_observations``: List of ``(verse_num, observation)`` tuples.
        - ``chapter_summary``: The chapter-level summary observation.
    """
    run_timestamp = datetime.now(tz=timezone.utc)
    book_upper = book.strip().upper()
    normalized_lang = normalize_biblical_language(biblical_language)

    type_config = get_analysis_type_config(analysis_type)
    output_schema = type_config["output_schema"]

    # -----------------------------------------------------------------------
    # 1. Load scripture
    # -----------------------------------------------------------------------
    print(f"[1/6] Loading scripture: {book_upper} {chapter}")
    (
        _translation_lookup,
        _biblical_lookup,
        verse_records,
        _translation_path,
        _biblical_path,
        _translation_usfm,
        _biblical_usfm,
    ) = load_analysis_scripture_data(
        book=book_upper,
        chapter=chapter,
        translation_language=translation_language,
        biblical_language=biblical_language,
        usfm_root=usfm_root,
    )

    # -----------------------------------------------------------------------
    # 2. Enrich (few-shot only)
    # -----------------------------------------------------------------------
    interpresure_loaded = False
    bart_loaded = False
    macula_loaded = False

    if use_expert_materials:
        print("[2/6] Enriching with expert materials")
        try:
            interp = Interpresure(book_upper, chapter)
            for rec in verse_records:
                rec["pragmatic_annotations"] = interp.get_annotations_markdown(
                    None, int(rec["chapter"]), int(rec["verse"])
                )
            interpresure_loaded = True
            print(f"  ✓ InterpreSure annotations loaded")
        except KeyError:
            print(f"  ⚠️  No InterpreSure annotations for {book_upper} {chapter}")

        if normalized_lang == "grc":
            if macula_db_path:
                try:
                    verse_records = enrich_verse_records_with_macula_tokens(
                        verse_records, macula_db_path
                    )
                    macula_loaded = True
                    print(f"  ✓ MACULA tokens loaded")
                except Exception as e:
                    print(f"  ⚠️  MACULA enrichment failed: {e}")

            if bart_db_path:
                try:
                    verse_records = enrich_verse_records_with_bart_annotations(
                        verse_records, bart_db_path
                    )
                    bart_loaded = True
                    print(f"  ✓ BART annotations loaded")
                except Exception as e:
                    print(f"  ⚠️  BART enrichment failed: {e}")
    else:
        print("[2/6] Zero-shot mode — skipping expert enrichment")

    # -----------------------------------------------------------------------
    # 3. Create tools, skills, agents
    # -----------------------------------------------------------------------
    print("[3/6] Initialising agents")
    verse_lookup_tool = make_verse_lookup_tool(
        usfm_root=usfm_root,
        translation_language=translation_language,
        biblical_language=biblical_language,
    )
    skills_provider = make_interpresure_skills_provider()

    analyst, critic = make_agents(
        model=model,
        critic_model=critic_model,
        analysis_type=analysis_type,
        analysis_mode=analysis_mode,
        api_key=api_key,
        base_url=base_url,
        verse_lookup_tool=verse_lookup_tool,
        skills_provider=skills_provider,
    )
    discourse_agent = make_discourse_agent(
        model=model,
        analysis_type=analysis_type,
        api_key=api_key,
        base_url=base_url,
        skills_provider=skills_provider,
    )

    # -----------------------------------------------------------------------
    # 4. Global discourse pass
    # -----------------------------------------------------------------------
    print("[4/6] Running global discourse pass")
    discourse_map = await run_discourse_pass(
        verse_records=verse_records,
        biblical_language=normalized_lang,
        discourse_agent=discourse_agent,
    )
    boundary_starts = discourse_map.boundary_start_verses()
    print(
        f"  ✓ Discourse map produced — "
        f"{len(discourse_map.discourse_boundaries)} unit(s), "
        f"{len(discourse_map.dominant_quds)} QUD(s)"
    )

    # -----------------------------------------------------------------------
    # 5. Verse-by-verse loop
    # -----------------------------------------------------------------------
    print(f"[5/6] Analysing {len(verse_records)} verse(s)")

    session = analyst.create_session()

    # Inject full chapter context + discourse map once
    chapter_ctx = _build_chapter_context_prompt(
        verse_records, discourse_map, normalized_lang
    )
    await analyst.run(chapter_ctx, session=session)

    verse_observations: list[tuple[int, Any]] = []
    max_rounds = MAX_CRITIC_ROUNDS

    for verse_record in verse_records:
        verse_num = int(verse_record["verse"])
        ref = verse_record.get("reference", f"{book_upper} {chapter}:{verse_num}")
        print(f"  → {ref}")

        # Inject discourse boundary marker if applicable
        if discourse_boundary_markers and verse_num in boundary_starts:
            unit = discourse_map.boundary_for_verse(verse_num)
            if unit:
                marker = _build_boundary_marker(unit)
                print(f"    ⇢ {marker}")
                await analyst.run(marker, session=session)

        obs = await _analyze_verse_with_critic(
            analyst=analyst,
            critic=critic,
            session=session,
            verse_record=verse_record,
            biblical_language=normalized_lang,
            output_schema=output_schema,
            max_rounds=max_rounds,
        )
        verse_observations.append((verse_num, obs))

    # -----------------------------------------------------------------------
    # 5b. Chapter summary
    # -----------------------------------------------------------------------
    print("  → Chapter summary")
    summary_result = await analyst.run(_build_chapter_summary_prompt(), session=session)
    chapter_summary = await _parse_observation(summary_result, output_schema)

    # -----------------------------------------------------------------------
    # 6. Write output
    # -----------------------------------------------------------------------
    print("[6/6] Writing run directory")

    norm_mode = "few_shot" if use_expert_materials else "zero_shot"
    run_metadata = AnalysisRunMetadataObservation(
        model=model,
        critic_model=critic_model,
        analysis_mode=norm_mode,
        analysis_type=analysis_type,
        timestamp=run_timestamp.isoformat(),
        translation=TranslationInfo(
            language=translation_language,
            title=translation_title,
        ),
        biblical_language=normalized_lang,
        resources=ResourcesUsed(
            interpresure=interpresure_loaded,
            bart=bart_loaded,
            macula=macula_loaded,
            discourse_boundary_markers=discourse_boundary_markers and bool(boundary_starts),
        ),
    )

    items = build_scope_items(
        book=book_upper,
        chapter=chapter,
        run_metadata_obs=run_metadata,
        discourse_map_obs=discourse_map,
        verse_observations=verse_observations,
        chapter_summary_obs=chapter_summary,
    )

    writer = RunWriter(
        output_dir=output_dir,
        repo_id=repo_id,
        repo_name=repo_name,
    )
    run_dir = writer.write(
        book=book_upper,
        chapter=chapter,
        items=items,
        timestamp=run_timestamp,
    )

    print(f"\n✅ Run written to: {run_dir}")
    return {
        "run_dir": run_dir,
        "discourse_map": discourse_map,
        "verse_observations": verse_observations,
        "chapter_summary": chapter_summary,
    }


def run_analysis_sync(**kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper around ``generate_analysis`` for CLI use."""
    return asyncio.run(generate_analysis(**kwargs))
