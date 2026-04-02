"""Utilities for converting annotation CSVs into schema-driven AI workflows.

This module supports four CLI commands:

* ``convert`` - turn a CSV into JSON rows plus a JSON Schema sidecar.
* ``build`` - use prose notes plus a schema to generate structured rows.
* ``test`` - use only verse text plus a schema to generate structured rows.
* ``compare`` - compare hand-made and AI-made CSVs grouped by book/chapter/verse.

The commands are intentionally schema-first so the dataset can vary by file
without hard-coding a Pydantic model for each annotation set.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import click
import pandas as pd
from dotenv import load_dotenv

from model.config import get_config_for_model

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANG_ROOT = REPO_ROOT / "lang"
DEFAULT_TRANSLATION_LANGUAGE = "en"
DEFAULT_BIBLICAL_LANGUAGE = "heb"
DEFAULT_CONTEXT_WINDOW = 4
REFERENCE_RE = re.compile(
    r"^(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)(?:-\d+)?$"
)
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:  # pragma: no cover - only needed for AI commands
    AssistantAgent = None
    ModelInfo = None
    OpenAIChatCompletionClient = None


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_GROUP_KEYS = ["book", "chapter", "verse"]
DEFAULT_ROWS_KEY = "rows"
PREFERRED_ROW_KEYS = [
    "token_id",
    "biblical_text",
    "verse_reference",
    "reference",
    "row_id",
    "annotation_id",
    "segment_id",
    "segment",
    "term",
]

INTEGER_LIKE_COLUMNS = {
    "chapter",
    "verse",
    "token_id",
    "row_id",
    "annotation_id",
    "score",
    "index",
    "order",
    "sequence",
    "position",
}

BOOLEAN_LIKE_PREFIXES = ("is_", "has_", "was_", "were_", "should_", "can_", "does_")
BOOLEAN_LIKE_COLUMNS = {
    "accepted",
    "approved",
    "correct",
    "dft_preserved",
    "intervened",
    "missing",
    "present",
    "required",
    "valid",
}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_biblical_language(language: str) -> str:
    lower = language.lower()
    if lower in {"grc", "greek"}:
        return "grc"
    if lower in {"heb", "hebrew"}:
        return "heb"
    raise click.ClickException("Biblical language must be Greek or Hebrew.")


def _load_usfm_parser():
    try:
        from usfm2dict import UsfmParser

        return UsfmParser()
    except ImportError:
        return None


def _fallback_parse_usfm(usfm_text: str) -> dict[str, str]:
    """Parse a minimal subset of USFM when usfm2dict is unavailable.

    This is intentionally conservative: it looks for \id, \c, and \v markers
    and keeps the intervening verse text. It is enough for verse-by-verse
    dataset generation in environments without the external parser installed.
    """

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


def _parse_usfm_text(usfm_text: str) -> dict[str, str]:
    parser = _load_usfm_parser()
    if parser is not None:
        return parser.parse(usfm_text)
    return _fallback_parse_usfm(usfm_text)


def _resolve_usfm_file(usfm_root: Path, language: str, book: str) -> Path:
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


def _build_verse_lookup(
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
                "text": clean_text,
            }
        )

    verse_records.sort(key=lambda item: (item["chapter"], item["verse"]))
    return verse_lookup, verse_records


def load_scripture_data(
    *,
    book: str,
    chapter: int | None,
    translation_language: str,
    biblical_language: str,
    usfm_root: Path = DEFAULT_LANG_ROOT,
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str], list[dict[str, Any]], Path, Path]:
    translation_path = _resolve_usfm_file(usfm_root, translation_language, book)
    biblical_path = _resolve_usfm_file(usfm_root, biblical_language, book)

    translation_text = _read_text_file(translation_path)
    biblical_text = _read_text_file(biblical_path)

    translation_parsed = _parse_usfm_text(translation_text)
    biblical_parsed = _parse_usfm_text(biblical_text)

    translation_lookup, translation_records = _build_verse_lookup(
        translation_parsed, book=book, chapter=chapter
    )
    biblical_lookup, biblical_records = _build_verse_lookup(
        biblical_parsed, book=book, chapter=chapter
    )

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

    return translation_lookup, biblical_lookup, verse_records, translation_path, biblical_path


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple, set)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (dict, list, tuple, set)):
        return value

    if isinstance(value, float) and math.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, str):
        return " ".join(value.split())

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    return value


def _json_safe(value: Any) -> Any:
    value = _normalize_scalar(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def _clean_value_for_comparison(value: Any) -> Any:
    value = _normalize_scalar(value)
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list)):
        return _json_safe(value)
    return value


def _infer_json_type(series: pd.Series, column_name: str) -> str:
    column = column_name.strip().lower()
    non_null = series.dropna()

    if column in INTEGER_LIKE_COLUMNS:
        return "integer"

    if column.startswith(BOOLEAN_LIKE_PREFIXES) or column in BOOLEAN_LIKE_COLUMNS:
        return "boolean"

    if non_null.empty:
        return "string"

    sample_values = []
    for value in non_null.head(50).tolist():
        value = _normalize_scalar(value)
        if value is None:
            continue
        sample_values.append(value)

    if not sample_values:
        return "string"

    if all(isinstance(value, bool) for value in sample_values):
        return "boolean"

    if all(isinstance(value, int) and not isinstance(value, bool) for value in sample_values):
        return "integer"

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in sample_values):
        return "number"

    if all(isinstance(value, str) for value in sample_values):
        lowered = [value.strip().lower() for value in sample_values]
        if lowered and all(item in {"true", "false", "yes", "no", "1", "0"} for item in lowered):
            return "boolean"
        if all(item.replace("-", "").replace(".", "", 1).isdigit() for item in lowered):
            if all("." not in item for item in lowered):
                return "integer"
            return "number"

    return "string"


def _column_description(column_name: str) -> str:
    return column_name.replace("_", " ").strip().capitalize()


def infer_schema_from_dataframe(
    df: pd.DataFrame,
    *,
    title: str = "DatasetRows",
    rows_key: str = DEFAULT_ROWS_KEY,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    normalized_df = _normalize_dataframe_columns(df)

    properties: dict[str, Any] = {}
    required: list[str] = []
    column_types: dict[str, str] = {}

    for column in normalized_df.columns:
        inferred_type = _infer_json_type(normalized_df[column], column)
        column_types[column] = inferred_type
        properties[column] = {
            "type": inferred_type,
            "description": _column_description(column),
        }
        required.append(column)

    row_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": title,
        "type": "object",
        "properties": {
            rows_key: {
                "type": "array",
                "items": row_schema,
            }
        },
        "required": [rows_key],
        "additionalProperties": False,
        "x-column-order": list(normalized_df.columns),
        "x-column-types": column_types,
        "x-group-keys": group_keys or DEFAULT_GROUP_KEYS,
    }
    return schema


def _schema_from_csv(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    schema = infer_schema_from_dataframe(df, title=csv_path.stem)
    return df, schema


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = _normalize_dataframe_columns(df)
    records = normalized.to_dict(orient="records")
    return [{key: _json_safe(value) for key, value in record.items()} for record in records]


def _load_schema(schema_path: Path | None, csv_path: Path | None = None, json_path: Path | None = None) -> dict[str, Any]:
    if schema_path is not None:
        return json.loads(schema_path.read_text(encoding="utf-8"))

    if json_path is not None:
        sidecar = json_path.with_suffix(".schema.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        rows_key = DEFAULT_ROWS_KEY if DEFAULT_ROWS_KEY in loaded else "analysis"
        rows = loaded.get(rows_key, [])
        if rows:
            frame = pd.DataFrame(rows)
            return infer_schema_from_dataframe(frame, title=json_path.stem, rows_key=rows_key)

    if csv_path is not None:
        _, schema = _schema_from_csv(csv_path)
        return schema

    raise click.ClickException("A schema, CSV, or JSON file is required.")


def _schema_properties(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties", {})
    rows_schema = properties.get(DEFAULT_ROWS_KEY, {})
    items = rows_schema.get("items", {})
    row_properties = items.get("properties", {})
    order = schema.get("x-column-order")
    if isinstance(order, list) and order:
        return [str(column) for column in order]
    return list(row_properties.keys())


def _response_format_from_schema(schema: dict[str, Any], *, name: str = "dataset_rows") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _build_agent(
    *,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    schema: dict[str, Any],
    system_message: str,
) -> Any:
    if AssistantAgent is None or ModelInfo is None or OpenAIChatCompletionClient is None:
        raise click.ClickException(
            "autogen_agentchat is not installed in this environment, so the AI generation commands cannot run."
        )

    if api_key is None or base_url is None:
        config = get_config_for_model(model_name)
        api_key = api_key or config.get("key")
        base_url = base_url if base_url is not None else config.get("base_url")

    client_kwargs: dict[str, Any] = {
        "api_type": "openai",
        "model": model_name,
        "api_key": api_key,
        "model_info": ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        "timeout": 120,
        "response_format": _response_format_from_schema(schema),
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIChatCompletionClient(**client_kwargs)
    return AssistantAgent(
        name="DATASET_BUILDER",
        system_message=system_message,
        model_client=client,
    )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _extract_json_payload(raw_content: Any) -> dict[str, Any]:
    if isinstance(raw_content, dict):
        return raw_content

    if not isinstance(raw_content, str):
        raise ValueError("Model response was not text or JSON.")

    cleaned = _strip_code_fences(raw_content)
    return json.loads(cleaned)


def _prompt_for_generation(
    *,
    schema: dict[str, Any],
    notes_text: str,
    source_text: str | None,
    mode_label: str,
    use_expert_notes: bool,
) -> str:
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    column_order = schema.get("x-column-order", _schema_properties(schema))

    prompt_parts = [
        f"ROLE: You are converting biblical annotation material into structured CSV rows for {mode_label}.",
        "You must output a single JSON object that matches the schema exactly.",
        "Do not add any extra keys.",
        "If a value is unavailable, use an empty string, null, false, or 0 as appropriate for the field type.",
        "Preserve the original row order when you can infer it.",
        "If the source material implies multiple rows for the same verse, output multiple row objects.",
        "Every row must include all columns from the schema.",
        "",
        "SCHEMA:",
        schema_text,
        "",
        "COLUMN ORDER:",
        ", ".join(column_order),
    ]

    if source_text:
        prompt_parts.extend(
            [
                "",
                "VERSE OR SOURCE TEXT:",
                source_text,
            ]
        )

    if use_expert_notes:
        prompt_parts.extend(
            [
                "",
                "PROSE NOTES:",
                notes_text,
                "",
                "TASK:",
                "Use the prose notes as the source of truth and map them into the structured rows.",
                "Keep the wording faithful to the notes and do not invent unsupported annotations.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "TASK:",
                "Infer the row values from the verse text alone.",
                "Do not use any expert annotations or prose notes.",
                "If a field cannot be justified from the verse text, leave it blank or use a neutral placeholder.",
            ]
        )

    return "\n".join(prompt_parts)


def _format_context_window(history: list[dict[str, Any]], context_window: int) -> str:
    if not history:
        return "No prior verses have been processed yet."

    selected = history[-context_window:] if context_window > 0 else history
    blocks: list[str] = []

    for item in selected:
        blocks.append(f"Verse {item['reference']}")
        blocks.append(f"Biblical text: {item.get('biblical_text', '')}")
        translation_text = item.get("translation_text")
        if translation_text:
            blocks.append(f"Translation text: {translation_text}")
        generated_rows = item.get("rows", [])
        if generated_rows:
            blocks.append("Generated rows:")
            for row in generated_rows:
                blocks.append(json.dumps(row, ensure_ascii=False))
        blocks.append("")

    return "\n".join(blocks).strip()


def _fill_row_metadata(
    row: dict[str, Any],
    verse_record: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    cleaned = {str(key): _json_safe(value) for key, value in row.items()}
    row_properties = schema.get("properties", {}).get(DEFAULT_ROWS_KEY, {}).get("items", {}).get("properties", {})

    metadata = {
        "book": verse_record.get("book"),
        "chapter": verse_record.get("chapter"),
        "verse": verse_record.get("verse"),
        "verse_reference": verse_record.get("reference"),
        "biblical_text": verse_record.get("biblical_text"),
        "translation": verse_record.get("translation_text"),
    }

    for key, value in metadata.items():
        if key in row_properties:
            current_value = cleaned.get(key)
            if current_value in {None, ""}:
                cleaned[key] = value

    return cleaned


def _build_verse_prompt(
    *,
    schema: dict[str, Any],
    verse_record: dict[str, Any],
    context_history: list[dict[str, Any]],
    context_window: int,
    notes_text: str | None,
    use_expert_notes: bool,
    mode_label: str,
) -> str:
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    column_order = schema.get("x-column-order", _schema_properties(schema))
    context_text = _format_context_window(context_history, context_window)

    prompt_parts = [
        f"ROLE: You are building a structured dataset for {mode_label}.",
        "Work through the verse data in order and keep the terminology consistent with prior verses.",
        "Return exactly one JSON object with a 'rows' array.",
        "Only generate rows for the CURRENT VERSE.",
        "If the schema contains book/chapter/verse or verse_reference fields, fill them consistently for the current verse.",
        "Do not add any keys outside the schema.",
        "",
        "SCHEMA:",
        schema_text,
        "",
        "COLUMN ORDER:",
        ", ".join(column_order),
        "",
        "CURRENT VERSE:",
        f"Reference: {verse_record['reference']}",
        f"Biblical text: {verse_record.get('biblical_text', '')}",
        f"Translation text: {verse_record.get('translation_text', '')}",
        "",
        "PREVIOUS VERSE CONTEXT:",
        context_text,
    ]

    if use_expert_notes:
        prompt_parts.extend(
            [
                "",
                "PROSE NOTES:",
                notes_text or "",
                "",
                "TASK:",
                "Use the prose notes and the current verse context to fill the schema.",
                "Keep the row count appropriate for the verse, even if that differs from neighboring verses.",
                "Make sure the annotations captured for earlier verses remain consistent as the context advances.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "TASK:",
                "Infer the schema values from the verse text alone.",
                "Do not rely on expert annotations or prose notes.",
                "Use the previous verse context only to keep terminology and interpretation consistent.",
            ]
        )

    return "\n".join(prompt_parts)


async def _run_generation(
    *,
    schema: dict[str, Any],
    prompt: str,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    agent = _build_agent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        schema=schema,
        system_message=(
            "You are a careful dataset constructor. "
            "Return valid JSON only and satisfy the provided schema exactly."
        ),
    )
    result = await agent.run(task=prompt)
    raw_content = result.messages[-1].content
    return _extract_json_payload(raw_content)


async def _generate_rows_for_verses(
    *,
    schema: dict[str, Any],
    verse_records: list[dict[str, Any]],
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    notes_text: str | None,
    use_expert_notes: bool,
    context_window: int,
    mode_label: str,
) -> list[dict[str, Any]]:
    agent = _build_agent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        schema=schema,
        system_message=(
            "You are a careful dataset constructor. "
            "Return valid JSON only and satisfy the provided schema exactly."
        ),
    )

    all_rows: list[dict[str, Any]] = []
    context_history: list[dict[str, Any]] = []

    for verse_record in verse_records:
        prompt = _build_verse_prompt(
            schema=schema,
            verse_record=verse_record,
            context_history=context_history,
            context_window=context_window,
            notes_text=notes_text,
            use_expert_notes=use_expert_notes,
            mode_label=mode_label,
        )
        result = await agent.run(task=prompt)
        payload = _extract_json_payload(result.messages[-1].content)
        verse_rows = _rows_from_payload(payload)
        verse_rows = [_fill_row_metadata(row, verse_record, schema) for row in verse_rows]
        all_rows.extend(verse_rows)

        context_history.append(
            {
                "reference": verse_record["reference"],
                "biblical_text": verse_record.get("biblical_text", ""),
                "translation_text": verse_record.get("translation_text", ""),
                "rows": verse_rows,
            }
        )

    return all_rows


def _rows_from_payload(payload: dict[str, Any], rows_key: str = DEFAULT_ROWS_KEY) -> list[dict[str, Any]]:
    rows = payload.get(rows_key)
    if not isinstance(rows, list):
        raise click.ClickException(f"Model response did not contain a '{rows_key}' list.")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException("Each generated row must be a JSON object.")
        normalized_rows.append({str(key): _json_safe(value) for key, value in row.items()})
    return normalized_rows


def _coerce_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    coerced.columns = [str(column).strip() for column in coerced.columns]
    return coerced


def _group_dataframe(df: pd.DataFrame, group_keys: Iterable[str]) -> pd.core.groupby.DataFrameGroupBy:
    missing = [key for key in group_keys if key not in df.columns]
    if missing:
        raise click.ClickException(f"Missing required comparison columns: {', '.join(missing)}")
    return df.groupby(list(group_keys), dropna=False, sort=True)


def _candidate_row_key_columns(df_manual: pd.DataFrame, df_ai: pd.DataFrame, group_keys: list[str]) -> list[str]:
    common = [column for column in df_manual.columns if column in df_ai.columns and column not in group_keys]
    preferred = [column for column in PREFERRED_ROW_KEYS if column in common]
    if preferred:
        return preferred

    return common


def _is_unique_key(df: pd.DataFrame, columns: list[str]) -> bool:
    if not columns:
        return False
    subset = df[columns].copy()
    return not subset.duplicated().any()


def _select_row_key(df_manual: pd.DataFrame, df_ai: pd.DataFrame, group_keys: list[str]) -> list[str] | None:
    candidates = _candidate_row_key_columns(df_manual, df_ai, group_keys)
    for column in PREFERRED_ROW_KEYS:
        if column not in candidates:
            continue
        if _is_unique_key(df_manual, [column]) and _is_unique_key(df_ai, [column]):
            return [column]

    for column in candidates:
        if _is_unique_key(df_manual, [column]) and _is_unique_key(df_ai, [column]):
            return [column]

    # Try small multi-column keys before falling back to a multiset comparison.
    if len(candidates) >= 2:
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                candidate = [candidates[left], candidates[right]]
                if _is_unique_key(df_manual, candidate) and _is_unique_key(df_ai, candidate):
                    return candidate

    return None


def _row_signature(row: pd.Series, columns: list[str]) -> str:
    signature = {column: _clean_value_for_comparison(row[column]) for column in columns}
    return json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)


def _compare_without_key(
    manual_group: pd.DataFrame,
    ai_group: pd.DataFrame,
    compare_columns: list[str],
) -> dict[str, Any]:
    manual_signatures = Counter(_row_signature(row, compare_columns) for _, row in manual_group.iterrows())
    ai_signatures = Counter(_row_signature(row, compare_columns) for _, row in ai_group.iterrows())

    missing = []
    extra = []

    for signature, count in (manual_signatures - ai_signatures).items():
        missing.extend([json.loads(signature)] * count)

    for signature, count in (ai_signatures - manual_signatures).items():
        extra.extend([json.loads(signature)] * count)

    return {
        "comparison_mode": "multiset",
        "matched_rows": sum((manual_signatures & ai_signatures).values()),
        "manual_only_rows": missing,
        "ai_only_rows": extra,
        "row_diffs": [],
    }


def _compare_with_key(
    manual_group: pd.DataFrame,
    ai_group: pd.DataFrame,
    key_columns: list[str],
    compare_columns: list[str],
) -> dict[str, Any]:
    manual_index = manual_group.set_index(key_columns, drop=False)
    ai_index = ai_group.set_index(key_columns, drop=False)

    manual_keys = set(manual_index.index.tolist())
    ai_keys = set(ai_index.index.tolist())

    row_diffs = []
    for key in sorted(manual_keys & ai_keys, key=lambda value: str(value)):
        manual_row = manual_index.loc[key]
        ai_row = ai_index.loc[key]

        if isinstance(manual_row, pd.DataFrame) or isinstance(ai_row, pd.DataFrame):
            continue

        changed_fields = {}
        for column in compare_columns:
            manual_value = _clean_value_for_comparison(manual_row[column])
            ai_value = _clean_value_for_comparison(ai_row[column])
            if manual_value != ai_value:
                changed_fields[column] = {
                    "manual": manual_value,
                    "ai": ai_value,
                }

        if changed_fields:
            row_diffs.append(
                {
                    "key": {column: _clean_value_for_comparison(manual_row[column]) for column in key_columns},
                    "differences": changed_fields,
                }
            )

    return {
        "comparison_mode": "keyed",
        "matched_rows": len(manual_keys & ai_keys),
        "manual_only_rows": [
            {column: _clean_value_for_comparison(row[column]) for column in manual_group.columns}
            for key, row in manual_index.iterrows()
            if key not in ai_keys
        ],
        "ai_only_rows": [
            {column: _clean_value_for_comparison(row[column]) for column in ai_group.columns}
            for key, row in ai_index.iterrows()
            if key not in manual_keys
        ],
        "row_diffs": row_diffs,
        "key_columns": key_columns,
    }


def compare_dataframes(
    manual_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    *,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    group_keys = group_keys or DEFAULT_GROUP_KEYS
    manual_df = _coerce_dataframe_columns(manual_df)
    ai_df = _coerce_dataframe_columns(ai_df)

    common_columns = [column for column in manual_df.columns if column in ai_df.columns]
    manual_only_columns = [column for column in manual_df.columns if column not in ai_df.columns]
    ai_only_columns = [column for column in ai_df.columns if column not in manual_df.columns]

    grouped_manual = _group_dataframe(manual_df, group_keys)
    grouped_ai = _group_dataframe(ai_df, group_keys)

    verse_keys = sorted(
        set(grouped_manual.groups.keys()) | set(grouped_ai.groups.keys()),
        key=lambda value: str(value),
    )
    verse_reports = []

    for verse_key in verse_keys:
        manual_group = grouped_manual.get_group(verse_key) if verse_key in grouped_manual.groups else pd.DataFrame(columns=manual_df.columns)
        ai_group = grouped_ai.get_group(verse_key) if verse_key in grouped_ai.groups else pd.DataFrame(columns=ai_df.columns)

        compare_columns = [column for column in common_columns if column not in group_keys]
        row_key = _select_row_key(manual_group, ai_group, group_keys)

        if row_key:
            comparison = _compare_with_key(manual_group, ai_group, row_key, compare_columns)
        else:
            comparison = _compare_without_key(manual_group, ai_group, compare_columns)

        report = {
            "book": verse_key[0] if len(verse_key) > 0 else None,
            "chapter": verse_key[1] if len(verse_key) > 1 else None,
            "verse": verse_key[2] if len(verse_key) > 2 else None,
            "manual_rows": int(len(manual_group)),
            "ai_rows": int(len(ai_group)),
            "manual_only_columns": manual_only_columns,
            "ai_only_columns": ai_only_columns,
            **comparison,
        }
        verse_reports.append(report)

    summary = {
        "group_keys": group_keys,
        "manual_row_count": int(len(manual_df)),
        "ai_row_count": int(len(ai_df)),
        "manual_only_columns": manual_only_columns,
        "ai_only_columns": ai_only_columns,
        "verse_reports": verse_reports,
    }
    return summary


def _print_comparison_summary(report: dict[str, Any]) -> None:
    mismatched = [
        verse
        for verse in report["verse_reports"]
        if verse["manual_only_rows"] or verse["ai_only_rows"] or verse["row_diffs"]
    ]
    click.echo(f"Compared {len(report['verse_reports'])} verse groups.")
    click.echo(f"Manual rows: {report['manual_row_count']}")
    click.echo(f"AI rows: {report['ai_row_count']}")
    click.echo(f"Verse groups with differences: {len(mismatched)}")
    if report["manual_only_columns"]:
        click.echo(f"Columns only in manual CSV: {', '.join(report['manual_only_columns'])}")
    if report["ai_only_columns"]:
        click.echo(f"Columns only in AI CSV: {', '.join(report['ai_only_columns'])}")


@click.group()
def cli() -> None:
    """Dataset utilities for schema-driven CSV and AI generation."""


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output JSON path.")
@click.option(
    "--schema-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output JSON Schema path.",
)
def convert(csv_path: Path, json_out: Path | None, schema_out: Path | None) -> None:
    """Convert a CSV into JSON rows plus a schema sidecar."""
    df, schema = _schema_from_csv(csv_path)
    records = _records_from_dataframe(df)
    payload = {DEFAULT_ROWS_KEY: records}

    if json_out is None:
        json_out = csv_path.with_suffix(".json")
    if schema_out is None:
        schema_out = csv_path.with_suffix(".schema.json")

    _write_json(json_out, payload)
    _write_json(schema_out, schema)
    click.echo(f"Wrote {json_out}")
    click.echo(f"Wrote {schema_out}")


@cli.command()
@click.option(
    "--template-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV whose columns will be used to infer the schema.",
)
@click.option(
    "--json-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON dataset file with a schema sidecar.",
)
@click.option(
    "--schema-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Pre-generated JSON Schema file from convert.",
)
@click.option("--book", required=True, help="Bible book code, such as PHM or PSA.")
@click.option("--chapter", type=int, default=None, help="Optional chapter number. Omit to process the full book.")
@click.option(
    "--translation-language",
    default=DEFAULT_TRANSLATION_LANGUAGE,
    show_default=True,
    help="Language folder for the translation USFM file.",
)
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
    help="Language folder for the original-language USFM file.",
)
@click.option(
    "--usfm-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_LANG_ROOT,
    show_default=True,
    help="Root folder containing the lang subdirectories.",
)
@click.option(
    "--context-window",
    type=int,
    default=DEFAULT_CONTEXT_WINDOW,
    show_default=True,
    help="How many prior verses to include in the rolling context.",
)
@click.option("--notes", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Prose notes.")
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Output CSV path.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional raw JSON output path.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model name to use.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Base URL override.")
def build(
    template_csv: Path | None,
    json_file: Path | None,
    schema_file: Path | None,
    book: str,
    chapter: int | None,
    translation_language: str,
    biblical_language: str,
    usfm_root: Path,
    context_window: int,
    notes: Path,
    output_csv: Path,
    output_json: Path | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Use prose notes and USFM verse context to generate a structured CSV."""
    if schema_file is None and template_csv is None and json_file is None:
        raise click.ClickException("Provide --template-csv, --json-file, or --schema-file.")

    schema = _load_schema(schema_file, csv_path=template_csv, json_path=json_file)
    normalized_biblical_language = _normalize_biblical_language(biblical_language)
    _, _, verse_records, translation_path, biblical_path = load_scripture_data(
        book=book,
        chapter=chapter,
        translation_language=translation_language,
        biblical_language=normalized_biblical_language,
        usfm_root=usfm_root,
    )

    if not verse_records:
        chapter_text = f" chapter {chapter}" if chapter is not None else ""
        raise click.ClickException(f"No overlapping verses found for {book.upper()}{chapter_text}.")

    rows = asyncio.run(
        _generate_rows_for_verses(
            schema=schema,
            verse_records=verse_records,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            notes_text=_read_text_file(notes),
            use_expert_notes=True,
            context_window=context_window,
            mode_label=f"{book.upper()} from prose notes",
        )
    )

    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=_schema_properties(schema))
    frame.to_csv(output_csv, index=False)
    click.echo(f"Wrote {output_csv}")

    if output_json is not None:
        payload = {
            DEFAULT_ROWS_KEY: rows,
            "book": book.upper(),
            "chapter": chapter,
            "translation_language": translation_language,
            "biblical_language": normalized_biblical_language,
            "translation_usfm": str(translation_path),
            "biblical_usfm": str(biblical_path),
        }
        _write_json(output_json, payload)
        click.echo(f"Wrote {output_json}")


@cli.command(name="test")
@click.option(
    "--template-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV whose columns will be used to infer the schema.",
)
@click.option(
    "--json-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON dataset file with a schema sidecar.",
)
@click.option(
    "--schema-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Pre-generated JSON Schema file from convert.",
)
@click.option("--book", required=True, help="Bible book code, such as PHM or PSA.")
@click.option("--chapter", type=int, default=None, help="Optional chapter number. Omit to process the full book.")
@click.option(
    "--translation-language",
    default=DEFAULT_TRANSLATION_LANGUAGE,
    show_default=True,
    help="Language folder for the translation USFM file.",
)
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
    help="Language folder for the original-language USFM file.",
)
@click.option(
    "--usfm-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_LANG_ROOT,
    show_default=True,
    help="Root folder containing the lang subdirectories.",
)
@click.option(
    "--context-window",
    type=int,
    default=DEFAULT_CONTEXT_WINDOW,
    show_default=True,
    help="How many prior verses to include in the rolling context.",
)
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Output CSV path.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional raw JSON output path.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model name to use.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Base URL override.")
def test(
    template_csv: Path | None,
    json_file: Path | None,
    schema_file: Path | None,
    book: str,
    chapter: int | None,
    translation_language: str,
    biblical_language: str,
    usfm_root: Path,
    context_window: int,
    output_csv: Path,
    output_json: Path | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Generate a test dataset from verse text only, without expert notes."""
    if schema_file is None and template_csv is None and json_file is None:
        raise click.ClickException("Provide --template-csv, --json-file, or --schema-file.")

    schema = _load_schema(schema_file, csv_path=template_csv, json_path=json_file)
    normalized_biblical_language = _normalize_biblical_language(biblical_language)
    _, _, verse_records, translation_path, biblical_path = load_scripture_data(
        book=book,
        chapter=chapter,
        translation_language=translation_language,
        biblical_language=normalized_biblical_language,
        usfm_root=usfm_root,
    )

    if not verse_records:
        chapter_text = f" chapter {chapter}" if chapter is not None else ""
        raise click.ClickException(f"No overlapping verses found for {book.upper()}{chapter_text}.")

    rows = asyncio.run(
        _generate_rows_for_verses(
            schema=schema,
            verse_records=verse_records,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            notes_text=None,
            use_expert_notes=False,
            context_window=context_window,
            mode_label=f"{book.upper()} verse-only test generation",
        )
    )

    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=_schema_properties(schema))
    frame.to_csv(output_csv, index=False)
    click.echo(f"Wrote {output_csv}")

    if output_json is not None:
        payload = {
            DEFAULT_ROWS_KEY: rows,
            "book": book.upper(),
            "chapter": chapter,
            "translation_language": translation_language,
            "biblical_language": normalized_biblical_language,
            "translation_usfm": str(translation_path),
            "biblical_usfm": str(biblical_path),
        }
        _write_json(output_json, payload)
        click.echo(f"Wrote {output_json}")


@cli.command()
@click.argument("manual_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ai_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional detailed report path.")
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional summary CSV path.")
def compare(manual_csv: Path, ai_csv: Path, output_json: Path | None, output_csv: Path | None) -> None:
    """Compare a hand-made CSV to an AI-generated CSV grouped by verse."""
    manual_df = pd.read_csv(manual_csv)
    ai_df = pd.read_csv(ai_csv)
    report = compare_dataframes(manual_df, ai_df)

    _print_comparison_summary(report)

    if output_json is not None:
        _write_json(output_json, report)
        click.echo(f"Wrote {output_json}")

    if output_csv is not None:
        summary_rows = []
        for verse in report["verse_reports"]:
            summary_rows.append(
                {
                    "book": verse["book"],
                    "chapter": verse["chapter"],
                    "verse": verse["verse"],
                    "manual_rows": verse["manual_rows"],
                    "ai_rows": verse["ai_rows"],
                    "comparison_mode": verse["comparison_mode"],
                    "manual_only_rows": len(verse["manual_only_rows"]),
                    "ai_only_rows": len(verse["ai_only_rows"]),
                    "row_diffs": len(verse["row_diffs"]),
                }
            )
        pd.DataFrame(summary_rows).to_csv(output_csv, index=False)
        click.echo(f"Wrote {output_csv}")


if __name__ == "__main__":
    cli()
