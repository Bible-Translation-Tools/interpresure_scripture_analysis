"""QA answering workflow.

Reads a question set JSON file and answers each question about a given
translation, producing ``interpresure_qa`` observations in a run directory.

The answering agent is stateless per question — each Q&A call is independent
so the workflow is straightforward to reason about and can be extended to
parallel execution later.

Resources provided to the answering agent per question:
  - The translation text for the relevant verse(s)
  - The original language text for the same verse(s)
  - The question and its rationale
  - Optionally: InterpreSure annotations, MACULA tokens, BART data
  - The InterpreSure skill (glossary + checklist)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from analysis.output import RunWriter
from analysis.repo_info import get_usfm_commit_sha, load_repo_info
from analysis.schemas import AnalysisItem, AnalysisRunMetadataObservation, ResourcesUsed, TranslationInfo
from analysis.usfm import load_analysis_scripture_data, normalize_biblical_language
from data.interpresure import Interpresure
from dataset.bart_mcp import enrich_verse_records_with_bart_annotations
from dataset.macula import enrich_verse_records_with_macula_tokens

from .agents import make_qa_agent
from .constants import DEFAULT_BIBLICAL_LANGUAGE, DEFAULT_LANG_ROOT, DEFAULT_MODEL, DEFAULT_TRANSLATION_LANGUAGE
from .schemas import GeneratedQuestion, InterpreSureQAObservation, QuestionSet


# ---------------------------------------------------------------------------
# Prompt builder for a single question
# ---------------------------------------------------------------------------


def _build_qa_prompt(
    *,
    question: GeneratedQuestion,
    translation_text: str,
    biblical_text: str,
    biblical_language: str,
    interpresure_markdown: str = "",
    macula_tokens: Any = None,
    bart_annotations: Any = None,
) -> str:
    lang_label = "Greek" if biblical_language == "grc" else "Hebrew"

    parts = [
        f"# Verse Reference: {question.anchor}",
        "",
        f"## {lang_label} Text",
        biblical_text or "(not available)",
        "",
        "## Translation",
        translation_text or "(not available)",
    ]

    if interpresure_markdown.strip():
        parts += ["", "## InterpreSure Expert Annotations", interpresure_markdown]

    if macula_tokens:
        parts += [
            "",
            "## MACULA Token Data",
            json.dumps(macula_tokens, ensure_ascii=False, indent=2, default=str),
        ]

    if bart_annotations:
        parts += [
            "",
            "## BART Discourse Annotations",
            json.dumps(bart_annotations, ensure_ascii=False, indent=2, default=str),
        ]

    parts += [
        "",
        "## Question",
        question.question,
        "",
        "## Rationale (internal — what pragmatic feature is being tested)",
        question.rationale or "(no rationale provided)",
        "",
        "## Task",
        dedent("""\
            Answer the question above based on the translation text.
            Use the original language text and annotations as interpretive context
            but base your verdict on what the *translation* communicates.

            Return a JSON object with:
            - question: (copy the question verbatim)
            - answer: plain-language answer (no source-language terms)
            - model: leave blank — the pipeline fills this in
            - reasoning: technical chain-of-thought (may use source language)
            - result: "pass", "fail", or "na"
            - severity: 0 if pass; 1–10 if fail or na
            - confidence: 0–100
        """),
    ]

    return "\n".join(parts)


def _get_verse_text(
    verse_records: list[dict[str, Any]],
    anchor: str,
) -> tuple[str, str]:
    """Extract biblical and translation text for the given anchor.

    Handles single verses ("PHM 1:10") and ranges ("ROM 3:10-18").
    Returns (biblical_text, translation_text).
    """
    # Try to parse verse range from anchor
    range_match = re.search(r":(\d+)(?:-(\d+))?$", anchor)
    if not range_match:
        return "", ""

    v_start = int(range_match.group(1))
    v_end = int(range_match.group(2)) if range_match.group(2) else v_start

    relevant = [
        r for r in verse_records
        if v_start <= int(r.get("verse", 0)) <= v_end
    ]

    biblical_parts = [r.get("biblical_text", "") for r in relevant if r.get("biblical_text")]
    translation_parts = [r.get("translation_text", "") for r in relevant if r.get("translation_text")]

    return " ".join(biblical_parts), " ".join(translation_parts)


def _get_annotations_for_anchor(
    verse_records: list[dict[str, Any]],
    anchor: str,
) -> tuple[str, Any, Any]:
    """Collect InterpreSure annotations, MACULA tokens, and BART data for the anchor."""
    range_match = re.search(r":(\d+)(?:-(\d+))?$", anchor)
    if not range_match:
        return "", None, None

    v_start = int(range_match.group(1))
    v_end = int(range_match.group(2)) if range_match.group(2) else v_start

    relevant = [
        r for r in verse_records
        if v_start <= int(r.get("verse", 0)) <= v_end
    ]

    annotations = "\n".join(
        r.get("pragmatic_annotations", "")
        for r in relevant
        if r.get("pragmatic_annotations", "").strip()
    )

    macula = [r["macula_tokens"] for r in relevant if r.get("macula_tokens")]
    bart = [r["bart_annotations"] for r in relevant if r.get("bart_annotations")]

    return annotations, macula or None, bart or None


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


async def run_qa_answering(
    *,
    questions_file: Path,
    translation_language: str = DEFAULT_TRANSLATION_LANGUAGE,
    biblical_language: str = DEFAULT_BIBLICAL_LANGUAGE,
    usfm_root: Path = DEFAULT_LANG_ROOT,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    macula_db_path: Path | None = None,
    bart_db_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Answer each question in the question set for the given translation.

    Args:
        questions_file: Path to the JSON question file from the generate step.
        translation_language: Language dir for the translation to evaluate.

    Returns a dict with:
        - ``run_dir``: Path to the scripture-analysis-api run directory.
        - ``answers``: list of ``InterpreSureQAObservation`` objects.
    """
    run_timestamp = datetime.now(tz=timezone.utc)

    # -----------------------------------------------------------------------
    # 1. Load question set
    # -----------------------------------------------------------------------
    print(f"[1/5] Loading question set: {questions_file.name}")
    raw_qs = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    question_set = QuestionSet.model_validate(raw_qs)
    book_upper = question_set.book.upper()
    chapter = question_set.chapter
    normalized_lang = normalize_biblical_language(
        biblical_language or question_set.biblical_language
    )

    print(f"  ✓ {len(question_set.questions)} question(s) for {book_upper} {chapter}")

    # -----------------------------------------------------------------------
    # 2. Load translation + biblical USFM
    # -----------------------------------------------------------------------
    print(f"[2/5] Loading scripture: {book_upper} {chapter}")
    repo_info = load_repo_info(usfm_root, translation_language)
    (
        _tl_lookup,
        _bl_lookup,
        verse_records,
        translation_path,
        _biblical_path,
        _translation_usfm,
        _biblical_usfm,
    ) = load_analysis_scripture_data(
        book=book_upper,
        chapter=chapter,
        translation_language=translation_language,
        biblical_language=normalized_lang,
        usfm_root=usfm_root,
    )
    commit_sha = get_usfm_commit_sha(translation_path)
    print(f"  ✓ Translation: {repo_info.name} ({commit_sha})")

    # -----------------------------------------------------------------------
    # 3. Enrich verse records
    # -----------------------------------------------------------------------
    print("[3/5] Loading annotations")
    resources_list: list[str] = []

    try:
        interp = Interpresure(book_upper, chapter)
        for rec in verse_records:
            rec["pragmatic_annotations"] = interp.get_annotations_markdown(
                None, int(rec["chapter"]), int(rec["verse"])
            )
        resources_list.append("interpresure")
        print(f"  ✓ InterpreSure loaded")
    except Exception as e:
        print(f"  ⚠️  InterpreSure not available: {e}")

    if macula_db_path:
        try:
            verse_records = enrich_verse_records_with_macula_tokens(verse_records, macula_db_path)
            resources_list.append("macula")
            print(f"  ✓ MACULA tokens loaded ({normalized_lang})")
        except Exception as e:
            print(f"  ⚠️  MACULA enrichment failed: {e}")

    if bart_db_path:
        if normalized_lang != "grc":
            print(f"  ℹ️  Skipping BART — Greek NT only")
        else:
            try:
                verse_records = enrich_verse_records_with_bart_annotations(verse_records, bart_db_path)
                resources_list.append("bart_displays")
                print(f"  ✓ BART annotations loaded")
            except Exception as e:
                print(f"  ⚠️  BART enrichment failed: {e}")

    # -----------------------------------------------------------------------
    # 4. Answer questions
    # -----------------------------------------------------------------------
    print(f"[4/5] Answering {len(question_set.questions)} question(s)")
    from analysis.skills import make_interpresure_skills_provider
    skills_provider = make_interpresure_skills_provider()

    agent = make_qa_agent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        skills_provider=skills_provider,
    )

    answers: list[tuple[GeneratedQuestion, InterpreSureQAObservation]] = []

    for q in question_set.questions:
        print(f"  → [{q.importance}/10] {q.anchor}: {q.question[:60]}…")

        biblical_text, translation_text = _get_verse_text(verse_records, q.anchor)
        annotations, macula_tokens, bart_annotations = _get_annotations_for_anchor(
            verse_records, q.anchor
        )

        prompt = _build_qa_prompt(
            question=q,
            translation_text=translation_text,
            biblical_text=biblical_text,
            biblical_language=normalized_lang,
            interpresure_markdown=annotations,
            macula_tokens=macula_tokens,
            bart_annotations=bart_annotations,
        )

        result = await agent.run(prompt)
        raw = result.text.strip() if hasattr(result, "text") else str(result).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

        obs = InterpreSureQAObservation.model_validate_json(raw)
        # Stamp the model name — LLM doesn't know its own identifier
        obs = obs.model_copy(update={"model": model})
        answers.append((q, obs))

        verdict_icon = {"pass": "✅", "fail": "❌", "na": "⚠️ "}[obs.result]
        print(f"    {verdict_icon} {obs.result} (severity={obs.severity}, confidence={obs.confidence})")

    # -----------------------------------------------------------------------
    # 5. Write run directory
    # -----------------------------------------------------------------------
    print("[5/5] Writing run directory")

    run_metadata_obs = AnalysisRunMetadataObservation(
        model=model,
        analysis_mode="few_shot" if "interpresure" in resources_list else "zero_shot",
        analysis_type="interpresure_qa",
        timestamp=run_timestamp.isoformat(),
        translation=TranslationInfo(language=repo_info.language, title=repo_info.name),
        biblical_language=normalized_lang,
        resources=ResourcesUsed(
            interpresure="interpresure" in resources_list,
            macula="macula" in resources_list,
            bart="bart_displays" in resources_list,
        ),
    )

    scope_items: list[AnalysisItem] = [
        AnalysisItem.from_observation(
            run_metadata_obs,
            book=book_upper,
            chapter=chapter,
            anchor=f"{book_upper} {chapter}",
            anchor_level="chapter",
        )
    ]

    for q, obs in answers:
        scope_items.append(
            AnalysisItem.from_observation(
                obs,
                book=book_upper,
                chapter=chapter,
                anchor=q.anchor,
                anchor_level="verse",
            )
        )

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
    print(f"\n✅ Run written to: {run_dir}")

    return {
        "run_dir": run_dir,
        "answers": [obs for _, obs in answers],
        "questions_file": questions_file,
    }
