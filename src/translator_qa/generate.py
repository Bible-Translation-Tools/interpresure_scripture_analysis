"""Question generation workflow.

Given a book and chapter, reads the original language text and InterpreSure
annotations to produce plain-language diagnostic questions for Mother Tongue
Translators (MTTs).

Outputs:
  1. A plain JSON question file — the input for the answer step.
  2. A run directory in scripture-analysis-api format with
     ``translator_questions`` observations (one per anchor, grouped).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from analysis.output import build_scope_items
from analysis.repo_info import RepoInfo
from analysis.schemas import AnalysisItem, AnalysisRunMetadataObservation, ResourcesUsed, TranslationInfo, TranslatorQuestion, TranslatorQuestionsObservation
from analysis.tools import format_verse_records_as_chapter_text
from data.interpresure import Interpresure
from dataset.bart_mcp import enrich_verse_records_with_bart_annotations
from dataset.macula import enrich_verse_records_with_macula_tokens
from analysis.usfm import load_analysis_scripture_data, normalize_biblical_language

from .agents import make_question_gen_agent
from .constants import DEFAULT_BIBLICAL_LANGUAGE, DEFAULT_LANG_ROOT, DEFAULT_MODEL
from .schemas import GeneratedQuestion, QuestionGenerationResponse, QuestionSet


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_generation_prompt(
    *,
    verse_records: list[dict[str, Any]],
    biblical_language: str,
    book: str,
    chapter: int,
    interpresure_markdown: str,
    bart_summary: str = "",
) -> str:
    lang_label = "Greek" if biblical_language == "grc" else "Hebrew"
    book_upper = book.strip().upper()

    chapter_text = format_verse_records_as_chapter_text(
        verse_records, biblical_language=lang_label
    )

    parts = [
        f"# {book_upper} Chapter {chapter} — {lang_label} Text\n",
        chapter_text,
        "",
        "# InterpreSure Expert Annotations",
        interpresure_markdown,
    ]

    if bart_summary.strip():
        parts += ["", "# BART Discourse Annotations", bart_summary]

    parts += [
        "",
        "# Task",
        dedent(f"""\
            Generate diagnostic questions for Mother Tongue Translators evaluating
            their translation of {book_upper} {chapter}.

            - Only generate questions for verses that have InterpreSure annotations above.
            - Each question must be plain language — no Greek, Hebrew, or jargon.
            - Ground each question in a specific annotation from the expert data.
            - Assign importance scores (1–10) based on how central the feature is to
              the communicative purpose of the passage.
            - Provide a brief rationale (for internal use, may use technical terms).
            - ids must be sequential starting from 1.
            - Return the full question set as JSON.
        """),
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


async def run_question_generation(
    *,
    book: str,
    chapter: int,
    biblical_language: str = DEFAULT_BIBLICAL_LANGUAGE,
    usfm_root: Path = DEFAULT_LANG_ROOT,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    macula_db_path: Path | None = None,
    bart_db_path: Path | None = None,
    output_dir: Path,
    questions_dir: Path,
) -> dict[str, Any]:
    """Generate diagnostic questions for a chapter and write both output formats.

    Args:
        output_dir: Parent directory for the scripture-analysis-api run directory.
        questions_dir: Directory for the plain JSON question file.

    Returns a dict with:
        - ``run_dir``: Path to the scripture-analysis-api run directory.
        - ``questions_file``: Path to the plain JSON question file.
        - ``question_set``: The ``QuestionSet`` object.
    """
    run_timestamp = datetime.now(tz=timezone.utc)
    book_upper = book.strip().upper()
    normalized_lang = normalize_biblical_language(biblical_language)

    # -----------------------------------------------------------------------
    # 1. Load biblical USFM
    # -----------------------------------------------------------------------
    print(f"[1/4] Loading {normalized_lang.upper()} text: {book_upper} {chapter}")
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
        translation_language=normalized_lang,  # load biblical as both
        biblical_language=normalized_lang,
        usfm_root=usfm_root,
    )

    # -----------------------------------------------------------------------
    # 2. Enrich with InterpreSure, MACULA, BART
    # -----------------------------------------------------------------------
    print("[2/4] Loading annotations")
    resources_list: list[str] = []
    interpresure_markdown = ""

    try:
        interp = Interpresure(book_upper, chapter)
        for rec in verse_records:
            rec["pragmatic_annotations"] = interp.get_annotations_markdown(
                None, int(rec["chapter"]), int(rec["verse"])
            )
        # Collect full chapter annotation block
        interpresure_markdown = "\n".join(
            rec.get("pragmatic_annotations", "") for rec in verse_records
            if rec.get("pragmatic_annotations", "").strip()
        )
        resources_list.append("interpresure")
        print(f"  ✓ InterpreSure loaded")
    except (KeyError, Exception) as e:
        print(f"  ⚠️  InterpreSure not available: {e}")

    if macula_db_path:
        try:
            verse_records = enrich_verse_records_with_macula_tokens(verse_records, macula_db_path)
            resources_list.append("macula")
            print(f"  ✓ MACULA tokens loaded ({normalized_lang})")
        except Exception as e:
            print(f"  ⚠️  MACULA enrichment failed: {e}")

    bart_summary = ""
    if bart_db_path:
        if normalized_lang != "grc":
            print(f"  ℹ️  Skipping BART — Greek NT only (language={normalized_lang})")
        else:
            try:
                verse_records = enrich_verse_records_with_bart_annotations(verse_records, bart_db_path)
                resources_list.append("bart_displays")
                from analysis.discourse import _collect_bart_summary
                bart_summary = _collect_bart_summary(verse_records)
                print(f"  ✓ BART annotations loaded")
            except Exception as e:
                print(f"  ⚠️  BART enrichment failed: {e}")

    if not interpresure_markdown.strip():
        print("  ⚠️  No InterpreSure annotations found — questions will be zero-shot")

    # -----------------------------------------------------------------------
    # 3. Generate questions
    # -----------------------------------------------------------------------
    print("[3/4] Generating questions")
    from analysis.skills import make_interpresure_skills_provider
    skills_provider = make_interpresure_skills_provider()

    agent = make_question_gen_agent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        skills_provider=skills_provider,
    )

    prompt = _build_generation_prompt(
        verse_records=verse_records,
        biblical_language=normalized_lang,
        book=book_upper,
        chapter=chapter,
        interpresure_markdown=interpresure_markdown,
        bart_summary=bart_summary,
    )

    result = await agent.run(prompt)
    raw = result.text.strip() if hasattr(result, "text") else str(result).strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    generation_response = QuestionGenerationResponse.model_validate_json(raw)
    questions = generation_response.questions

    print(f"  ✓ Generated {len(questions)} question(s)")

    # -----------------------------------------------------------------------
    # 4. Write outputs
    # -----------------------------------------------------------------------
    print("[4/4] Writing outputs")

    question_set = QuestionSet(
        book=book_upper,
        chapter=chapter,
        generated_at=run_timestamp.isoformat(),
        model=model,
        resources=resources_list,
        biblical_language=normalized_lang,
        questions=questions,
    )

    # 4a. Plain JSON question file
    questions_dir = Path(questions_dir)
    questions_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = run_timestamp.strftime("%Y%m%dT%H%M%S")
    questions_file = questions_dir / f"{book_upper}_{chapter}_questions_{ts_slug}.json"
    questions_file.write_text(
        question_set.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"  ✓ Questions file: {questions_file}")

    # 4b. Scripture-analysis-api run directory
    # Group questions by anchor for translator_questions observations
    by_anchor: dict[str, list[GeneratedQuestion]] = defaultdict(list)
    for q in questions:
        by_anchor[q.anchor].append(q)

    # Build AnalysisItems: run_metadata + one translator_questions per anchor
    run_metadata_obs = AnalysisRunMetadataObservation(
        model=model,
        analysis_mode="few_shot" if "interpresure" in resources_list else "zero_shot",
        analysis_type="translator_questions",
        timestamp=run_timestamp.isoformat(),
        translation=TranslationInfo(language=normalized_lang, title=normalized_lang.upper()),
        biblical_language=normalized_lang,
        resources=ResourcesUsed(
            interpresure="interpresure" in resources_list,
            macula="macula" in resources_list,
            bart="bart_displays" in resources_list,
        ),
    )

    # Discourse map placeholder (not produced in this workflow)
    # We produce translator_questions directly without a discourse pass

    scope_items: list[AnalysisItem] = [
        AnalysisItem.from_observation(
            run_metadata_obs,
            book=book_upper,
            chapter=chapter,
            anchor=f"{book_upper} {chapter}",
            anchor_level="chapter",
        )
    ]

    for anchor, anchor_questions in sorted(by_anchor.items()):
        obs = TranslatorQuestionsObservation(
            questions=[
                TranslatorQuestion(
                    question=q.question,
                    importance=q.importance,
                    rationale=q.rationale or None,
                    annotation_topics=q.annotation_topics,
                )
                for q in anchor_questions
            ]
        )
        scope_items.append(
            AnalysisItem.from_observation(
                obs,
                book=book_upper,
                chapter=chapter,
                anchor=anchor,
                anchor_level="verse",
            )
        )

    # Determine repo_info for the biblical language dir
    try:
        from analysis.repo_info import get_usfm_commit_sha, load_repo_info
        repo_info = load_repo_info(usfm_root, normalized_lang)
        commit_sha = get_usfm_commit_sha(_biblical_path)
    except FileNotFoundError:
        from analysis.repo_info import RepoInfo
        repo_info = RepoInfo(
            repo_id=f"{normalized_lang}-source",
            name=f"{normalized_lang.upper()} Source Text",
            language=normalized_lang,
            git_url="",
        )
        commit_sha = "unknown"

    from analysis.output import RunWriter
    writer = RunWriter(
        output_dir=Path(output_dir),
        repo_info=repo_info,
        commit_sha=commit_sha,
    )
    run_dir = writer.write(
        book=book_upper,
        chapter=chapter,
        items=scope_items,
        timestamp=run_timestamp,
    )
    print(f"  ✓ Run directory: {run_dir}")

    return {
        "run_dir": run_dir,
        "questions_file": questions_file,
        "question_set": question_set,
    }
