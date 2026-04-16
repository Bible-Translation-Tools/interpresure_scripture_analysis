"""Analysis generation workflow helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from report import convert_pragmatic_analysis
from teams.pragmatic_analysis import PragmaticAnalysis

from .constants import DEFAULT_CRITIC_MODEL, DEFAULT_MODEL, DEFAULT_TRANSLATION_LANGUAGE, DEFAULT_TRANSLATION_TITLE
from dataset.bart_mcp import enrich_verse_records_with_bart_annotations
from dataset.macula import enrich_verse_records_with_macula_tokens
from data.interpresure import Interpresure

from .usfm import load_analysis_scripture_data, normalize_biblical_language


def finalize(
    outpath: Path,
    translation_language: str,
    translation_title: str,
    translation_usfm: str,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    final = {
        "translation": {
            "title": translation_title,
            "language": translation_language,
            "usfm": translation_usfm,
        },
        "evaluation": evaluations,
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, cls=NumpyBoolEncoder)
    return final


class NumpyBoolEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return str(obj)
        try:
            import numpy as np

            if isinstance(obj, np.bool_):
                return bool(obj)
        except Exception:
            pass
        return super().default(obj)


async def generate_analysis(
    *,
    book: str,
    chapter: int,
    biblical_language: str,
    translation_language: str = DEFAULT_TRANSLATION_LANGUAGE,
    translation_title: str = DEFAULT_TRANSLATION_TITLE,
    usfm_root: Path,
    model: str = DEFAULT_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    output_csv: Path,
    output_json: Path,
    macula_db_path: Path | None = None,
    bart_db_path: Path | None = None,
    analysis_mode: str = "zero-shot",
    use_expert_materials: bool = False,
) -> dict[str, Any]:
    (
        translation_lookup,
        biblical_lookup,
        verse_records,
        translation_path,
        biblical_path,
        translation_usfm,
        _biblical_usfm,
    ) = load_analysis_scripture_data(
        book=book,
        chapter=chapter,
        translation_language=translation_language,
        biblical_language=biblical_language,
        usfm_root=usfm_root,
    )
    normalized_biblical_language = normalize_biblical_language(biblical_language)

    interpresure = None
    if use_expert_materials:
        interpresure = Interpresure(book, chapter)
        if normalized_biblical_language == "grc":
            verse_records = enrich_verse_records_with_macula_tokens(verse_records, macula_db_path)
            verse_records = enrich_verse_records_with_bart_annotations(verse_records, bart_db_path)
        for verse_record in verse_records:
            verse_record["pragmatic_annotations"] = interpresure.get_annotations_markdown(
                "general",
                int(verse_record["chapter"]),
                int(verse_record["verse"]),
            )

    analysis = PragmaticAnalysis(
        model,
        critic_model,
        biblical_language,
        analysis_mode=analysis_mode,
        api_key=api_key,
        base_url=base_url,
        use_expert_materials=use_expert_materials,
    )

    await analysis.run(
        verse_records,
        output_csv_path=output_csv,
    )

    evaluation_path = output_json.with_name(f"{output_json.stem}_general_analysis.json")
    evaluation = convert_pragmatic_analysis.convert_pragmatic(
        individual_path=output_csv,
        output_path=evaluation_path,
        interpresure=interpresure,
        book=book,
    )

    final = finalize(
        output_json,
        translation_language,
        translation_title,
        translation_usfm,
        [evaluation],
    )

    return {
        "final": final,
        "evaluation": evaluation,
        "output_json": output_json,
        "evaluation_json": evaluation_path,
        "output_csv": output_csv,
        "translation_path": translation_path,
        "biblical_path": biblical_path,
        "verse_records": verse_records,
    }


def run_analysis_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(generate_analysis(**kwargs))
