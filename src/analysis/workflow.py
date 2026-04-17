"""Analysis generation workflow helpers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from report import convert_pragmatic_analysis
from teams.pragmatic_analysis import PragmaticAnalysis

from .constants import DEFAULT_CRITIC_MODEL, DEFAULT_MODEL, DEFAULT_TRANSLATION_LANGUAGE, DEFAULT_TRANSLATION_TITLE
from dataset.bart_mcp import enrich_verse_records_with_bart_annotations
from dataset.macula import enrich_verse_records_with_macula_tokens
from data.interpresure import Interpresure

from .usfm import load_analysis_scripture_data, normalize_biblical_language


def _slugify_shot_mode(analysis_mode: str) -> str:
    mode = (analysis_mode or "").strip().lower().replace("-", "_")
    if mode in {"few_shot", "fewshot", "build"}:
        return "few_shot"
    if mode in {"zero_shot", "zeroshot", "test"}:
        return "zero_shot"
    return mode or "analysis"


def _timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def build_analysis_output_paths(
    *,
    output_dir: Path,
    translation_language: str,
    book: str,
    chapter: int,
    analysis_mode: str,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_slug = str(translation_language).strip().lower() or "unknown"
    book_slug = str(book).strip().upper() or "BOOK"
    shot_slug = _slugify_shot_mode(analysis_mode)
    timestamp = _timestamp_slug()
    stem = f"{lang_slug}_{book_slug}_{int(chapter)}_{shot_slug}_{timestamp}"

    output_csv = output_dir / f"{stem}.csv"
    output_json = output_dir / f"{stem}.json"
    evaluation_json = output_dir / f"{stem}_general_analysis.json"
    return output_csv, output_json, evaluation_json


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
    output_dir: Path | None = None,
    output_csv: Path | None = None,
    output_json: Path | None = None,
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
                None,
                int(verse_record["chapter"]),
                int(verse_record["verse"]),
            )

    raise Exception()
    return

    if output_csv is None and output_json is None:
        if output_dir is None:
            raise ValueError("Either output_dir or at least one of output_csv/output_json must be provided.")
        output_csv, output_json, _ = build_analysis_output_paths(
            output_dir=Path(output_dir),
            translation_language=translation_language,
            book=book,
            chapter=chapter,
            analysis_mode=analysis_mode,
        )
    elif output_dir is not None and (output_csv is None or output_json is None):
        derived_output_csv, derived_output_json, _ = build_analysis_output_paths(
            output_dir=Path(output_dir),
            translation_language=translation_language,
            book=book,
            chapter=chapter,
            analysis_mode=analysis_mode,
        )
        if output_csv is None:
            output_csv = derived_output_csv
        if output_json is None:
            output_json = derived_output_json
    else:
        if output_csv is None and output_json is not None:
            output_csv = Path(output_json).with_suffix(".csv")
        if output_json is None and output_csv is not None:
            output_json = Path(output_csv).with_suffix(".json")

    output_csv = Path(output_csv)
    output_json = Path(output_json)
    evaluation_path = output_json.with_name(f"{output_json.stem}_general_analysis.json")

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
