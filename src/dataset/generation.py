"""Model-driven dataset generation helpers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import click
import pandas as pd

from model.config import get_config_for_model

from .constants import CSV_METADATA_COLUMNS, DEFAULT_MODEL, DEFAULT_ROWS_KEY, PREFERRED_ROW_KEYS, REFERENCE_RE
from .bart_mcp import enrich_verse_records_with_bart_annotations
from .few_shot import render_few_shot_examples
from .macula import enrich_verse_records_with_macula_tokens
from .schema import json_safe, load_schema, output_columns, response_format_from_schema, schema_properties, write_json

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:  # pragma: no cover - only needed for AI commands
    AssistantAgent = None
    ModelInfo = None
    OpenAIChatCompletionClient = None


def build_agent(
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
        "response_format": response_format_from_schema(schema),
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIChatCompletionClient(**client_kwargs)
    return AssistantAgent(
        name="DATASET_BUILDER",
        system_message=system_message,
        model_client=client,
    )


def build_chapter_system_message(
    *,
    schema: dict[str, Any],
    few_shot_examples_text: str | None,
    notes_text: str | None,
    use_expert_notes: bool,
    bart_access_enabled: bool,
    mode_label: str,
    book: str,
    chapter: int | None,
) -> str:
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    chapter_label = f"{book.upper()} chapter {chapter}" if chapter is not None else f"{book.upper()} the full book"

    parts = [
        f"You are maintaining one ongoing conversation for {chapter_label}.",
        f"The task is to build a structured dataset for {mode_label}.",
        "Keep the entire chapter in one conversational thread.",
        "Do not restart the conversation for each verse.",
        "The schema and any chapter-level guidance are provided here once at the start.",
        "This system message is the first message in the conversation.",
        "A single verse may produce multiple rows.",
        "Create one row per distinct pragmatic inference, span, or annotation that should be captured.",
        "If a verse contains multiple relevant words or phrases, annotate each one separately.",
        "Few-shot example chapters, if provided, appear below and may use different columns from the target schema.",
        "Learn style and annotation granularity from them, but do not assume their columns exist in the target schema.",
        "If Macula token rows are present in the verse prompt, use them directly rather than calling a database tool.",
        "If BART discourse-analysis annotations are provided, inspect them at least once for every verse before finalizing the answer.",
        "Prefer targeted lookups for the current verse and its immediate context.",
        "When answering, return valid JSON only.",
        "Each turn should produce rows for the current verse only.",
        "Do not repeat the schema in your response.",
        "A row may annotate a single word, a phrase, or a larger pragmatic unit.",
        "When a row includes token_id, it must be a comma-separated list of Macula xml:id values for the exact word or phrase being annotated.",
        "Never invent token ids.",
        "Use the Macula token rows provided in the verse prompt to verify the ids for the exact span before writing token_id.",
        "If a span covers multiple words, include all matching Macula ids in surface order, separated by commas.",
        "",
    ]

    if few_shot_examples_text:
        parts.extend(
            [
                "",
                few_shot_examples_text,
                "",
            ]
        )

    parts.extend(
        [
            "SCHEMA:",
            schema_text,
        ]
    )

    if use_expert_notes:
        parts.extend(
            [
                "",
                "PROSE NOTES:",
                notes_text or "",
                "",
                "Use the prose notes as the chapter-level source of truth.",
                "Keep terminology stable across the entire chapter.",
            ]
        )
    elif few_shot_examples_text:
        parts.extend(
            [
                "",
                "This is few-shot generation.",
                "Infer the annotation style from the example chapter files above.",
                "Apply that style to the current chapter and keep terminology consistent across turns.",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "This is zero-shot generation.",
                "Infer the rows from verse text alone.",
                "Use prior turns only to keep terminology and interpretation consistent.",
            ]
        )

    if bart_access_enabled:
        parts.extend(
            [
                "",
                "BART DISCOURSE ANNOTATIONS:",
                "A verse-level BART discourse-analysis lookup is provided below for Greek.",
                "Use it as evidence, and inspect it before producing the final rows for the verse.",
            ]
        )

    return "\n".join(parts)


def emit_system_prompt(system_message: str) -> None:
    click.echo("[SYSTEM] Chapter system prompt:")
    click.echo("```text")
    click.echo(system_message)
    click.echo("```")


def log(message: str, *, level: str = "INFO") -> None:
    click.echo(f"[{level}] {message}")


def _stream_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stream_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_stream_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_stream_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_stream_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_stream_content(content: Any) -> tuple[str, str]:
    if isinstance(content, (dict, list)):
        safe_content = _stream_json_safe(content)
        return "json", json.dumps(safe_content, indent=2, ensure_ascii=False)

    if not isinstance(content, str):
        return "text", str(content)

    stripped = content.strip()
    try:
        parsed = json.loads(strip_code_fences(stripped))
        return "json", json.dumps(_stream_json_safe(parsed), indent=2, ensure_ascii=False)
    except Exception:
        return "text", stripped


def _render_stream_message_text(message: Any) -> tuple[str, str]:
    content = getattr(message, "content", None)
    if hasattr(message, "to_text"):
        text = message.to_text()
        if isinstance(text, str) and text.strip():
            return "text", text.strip()
    if content is None:
        return "text", ""
    return format_stream_content(content)


def print_stream_chunk(message: Any, *, verse_label: str | None = None) -> None:
    source = getattr(message, "source", None) or getattr(message, "name", None) or "MODEL"
    type_name = getattr(message, "type", None) or message.__class__.__name__

    kind, rendered = _render_stream_message_text(message)
    if not rendered:
        return

    label = f"[{source}]"
    if verse_label:
        label = f"{label} {verse_label}"
    label = f"{label} {type_name}"

    click.echo(f"{label} streamed {kind} output:")
    if kind == "json":
        click.echo("```json")
        click.echo(rendered)
        click.echo("```")
    else:
        click.echo("```text")
        click.echo(rendered)
        click.echo("```")


def verse_key_from_row(row: dict[str, Any], default_book: str | None = None) -> tuple[str, int, int] | None:
    book_value = row.get("book") or row.get("Book") or default_book
    chapter_value = row.get("chapter")
    verse_value = row.get("verse")

    if book_value is not None and chapter_value is not None and verse_value is not None:
        try:
            if pd.notna(chapter_value) and pd.notna(verse_value):
                return str(book_value).upper(), int(chapter_value), int(verse_value)
        except Exception:
            pass

    verse_reference = row.get("verse_reference") or row.get("reference")
    if isinstance(verse_reference, str):
        match = REFERENCE_RE.match(verse_reference.strip())
        if match:
            return (
                match.group("book").upper(),
                int(match.group("chapter")),
                int(match.group("verse")),
            )

    return None


def load_checkpoint_rows(output_csv: Path, default_book: str) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return [], set()

    checkpoint_df = pd.read_csv(output_csv)
    if checkpoint_df.empty:
        return [], set()

    if "book" in checkpoint_df.columns:
        observed_books = {
            str(value).upper()
            for value in checkpoint_df["book"].dropna().tolist()
            if str(value).strip()
        }
        if observed_books and observed_books != {default_book.upper()}:
            raise click.ClickException(
                f"Existing output CSV at {output_csv} appears to belong to {', '.join(sorted(observed_books))}, "
                f"not {default_book.upper()}."
            )

    rows = checkpoint_df.to_dict(orient="records")
    completed = set()
    for row in rows:
        verse_key = verse_key_from_row(row, default_book=default_book)
        if verse_key is not None:
            completed.add(verse_key)
    return rows, completed


def build_context_history_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}

    for row in rows:
        verse_key = verse_key_from_row(row)
        if verse_key is None:
            continue

        if verse_key not in grouped:
            grouped[verse_key] = {
                "reference": row.get("verse_reference") or row.get("reference") or f"{verse_key[0]} {verse_key[1]}:{verse_key[2]}",
                "biblical_text": row.get("biblical_text", ""),
                "rows": [],
            }

        grouped[verse_key]["rows"].append(json_safe(dict(row)))

    return [grouped[key] for key in sorted(grouped.keys(), key=lambda value: (value[1], value[2]))]


def write_checkpoint_csv(output_csv: Path, rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=output_columns(schema))
    tmp_path = output_csv.with_name(f".{output_csv.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)


def format_verse_progress(verse_record: dict[str, Any], current: int, total: int) -> str:
    return (
        f"{current}/{total} "
        f"{verse_record['reference']} | "
        f"biblical='{verse_record.get('biblical_text', '')[:60]}'"
    )


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _decode_json_from_text(text: str) -> Any | None:
    cleaned = strip_code_fences(text).strip()
    if not cleaned:
        return None

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Any]] = []
    for match in re.finditer(r"[\{\[]", cleaned):
        start = match.start()
        try:
            payload, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        candidates.append((start, end, payload))

    if not candidates:
        return None

    def score(item: tuple[int, int, Any]) -> tuple[int, int]:
        start, _end, payload = item
        priority = 0
        if isinstance(payload, dict):
            priority += 10 if DEFAULT_ROWS_KEY in payload else 5
        elif isinstance(payload, list):
            priority += 1
        return priority, start

    _, _, best_payload = max(candidates, key=score)
    return best_payload


def extract_json_payload(raw_content: Any) -> dict[str, Any]:
    if isinstance(raw_content, dict):
        return raw_content

    if not isinstance(raw_content, str):
        raise ValueError("Model response was not text or JSON.")

    decoded = _decode_json_from_text(raw_content)
    if decoded is None:
        raise ValueError("Model response did not contain valid JSON.")
    if not isinstance(decoded, dict):
        raise ValueError("Model response JSON was not an object.")
    return decoded


def rows_from_payload(payload: dict[str, Any], rows_key: str = DEFAULT_ROWS_KEY) -> list[dict[str, Any]]:
    rows = payload.get(rows_key)
    if not isinstance(rows, list):
        raise click.ClickException(f"Model response did not contain a '{rows_key}' list.")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException("Each generated row must be a JSON object.")
        normalized_rows.append({str(key): json_safe(value) for key, value in row.items()})
    return normalized_rows


def fill_row_metadata(
    row: dict[str, Any],
    verse_record: dict[str, Any],
) -> dict[str, Any]:
    cleaned = {str(key): json_safe(value) for key, value in row.items()}

    metadata = {
        "book": verse_record.get("book"),
        "chapter": verse_record.get("chapter"),
        "verse": verse_record.get("verse"),
        "verse_reference": verse_record.get("reference"),
        "biblical_text": verse_record.get("biblical_text"),
    }

    for key, value in metadata.items():
        current_value = cleaned.get(key)
        if current_value in {None, ""}:
            cleaned[key] = value

    return cleaned


def format_context_window(history: list[dict[str, Any]], context_window: int) -> str:
    if not history:
        return "No prior verses have been processed yet."

    selected = history[-context_window:] if context_window > 0 else history
    blocks: list[str] = []

    for item in selected:
        blocks.append(f"Verse {item['reference']}")
        blocks.append(f"Biblical text: {item.get('biblical_text', '')}")
        macula_tokens = item.get("macula_tokens", [])
        if macula_tokens:
            blocks.append("Macula tokens:")
            for token in macula_tokens:
                blocks.append(
                    json.dumps(
                        {
                            "xml_id": token.get("xml_id"),
                            "ref": token.get("ref"),
                            "text": token.get("text"),
                            "lemma": token.get("lemma"),
                            "morph": token.get("morph"),
                        },
                        ensure_ascii=False,
                    )
                )
        bart_annotations = item.get("bart_annotations", [])
        if bart_annotations:
            blocks.append("BART annotations:")
            for annotation in bart_annotations:
                blocks.append(json.dumps(annotation, ensure_ascii=False))
        generated_rows = item.get("rows", [])
        if generated_rows:
            blocks.append("Generated rows:")
            for row in generated_rows:
                blocks.append(json.dumps(row, ensure_ascii=False))
        blocks.append("")

    return "\n".join(blocks).strip()


def build_verse_prompt(
    *,
    schema: dict[str, Any],
    verse_record: dict[str, Any],
    context_history: list[dict[str, Any]],
    context_window: int,
    bart_access_enabled: bool,
    mode_label: str,
) -> str:
    column_order = schema.get("x-column-order", schema_properties(schema))
    context_text = format_context_window(context_history, context_window)
    macula_tokens = verse_record.get("macula_tokens", [])

    prompt_parts = [
        f"Continue the chapter conversation for {mode_label}.",
        "Use the schema and chapter notes already provided in the system message.",
        "Return exactly one JSON object with a 'rows' array for the CURRENT VERSE only.",
        "Keep the terminology consistent with earlier turns in this same chapter.",
        "",
        "CURRENT VERSE:",
        f"Reference: {verse_record['reference']}",
        f"Biblical text: {verse_record.get('biblical_text', '')}",
        "",
        "EXPECTED COLUMNS:",
        ", ".join(column_order),
        "",
    ]

    if bart_access_enabled:
        prompt_parts.extend(
            [
                "BART DISCOURSE ANNOTATIONS:",
                "A verse-level BART discourse-analysis lookup is provided below for Greek.",
                "Use it as evidence before answering.",
                "",
            ]
        )

    if macula_tokens:
        token_rows = [
            {
                "xml_id": token.get("xml_id"),
                "ref": token.get("ref"),
                "text": token.get("text"),
                "lemma": token.get("lemma"),
                "morph": token.get("morph"),
                "gloss": token.get("gloss"),
                "role": token.get("role"),
            }
            for token in macula_tokens
        ]
        prompt_parts.extend(
            [
                "MACULA TOKENS FOR THIS VERSE:",
                json.dumps(token_rows, indent=2, ensure_ascii=False),
                "",
                "Use only the xml_id values from this list when filling token_id.",
                "Do not query a database tool.",
                "",
            ]
        )

    bart_annotations = verse_record.get("bart_annotations", [])
    if bart_annotations:
        prompt_parts.extend(
            [
                "BART ANNOTATIONS FOR THIS VERSE:",
                json.dumps(bart_annotations, indent=2, ensure_ascii=False),
                "",
                "Use the BART annotations to identify discourse structure and pragmatic relationships.",
                "",
            ]
        )

    prompt_parts.extend(
        [
            "PREVIOUS TURNS IN THIS CHAPTER:",
            context_text,
        ]
    )

    prompt_parts.extend(
        [
            "",
            "TASK:",
            "Produce the next JSON payload for this verse.",
            "Do not repeat the notes or schema in the response.",
        ]
    )

    return "\n".join(prompt_parts)


async def run_generation(
    *,
    schema: dict[str, Any],
    prompt: str,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    agent = build_agent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        schema=schema,
        system_message=build_chapter_system_message(
            schema=schema,
            few_shot_examples_text=None,
            notes_text=None,
            use_expert_notes=False,
            bart_access_enabled=False,
            mode_label="one-off generation",
            book="UNKNOWN",
            chapter=None,
        ),
    )
    result = await agent.run(task=prompt)
    raw_content = result.messages[-1].content
    return extract_json_payload(raw_content)


async def generate_rows_for_verses(
    *,
    schema: dict[str, Any],
    verse_records: list[dict[str, Any]],
    output_csv: Path,
    book: str,
    chapter: int | None,
    biblical_language: str,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    macula_db_path: Path | None,
    bart_db_path: Path | None,
    max_tool_calls_per_verse: int,
    few_shot_example_paths: Iterable[Path] | None,
    notes_text: str | None,
    use_expert_notes: bool,
    context_window: int,
    mode_label: str,
    stream: bool,
) -> list[dict[str, Any]]:
    verse_records = enrich_verse_records_with_macula_tokens(verse_records, macula_db_path)
    verse_records = enrich_verse_records_with_bart_annotations(verse_records, bart_db_path)
    bart_access_enabled = bart_db_path is not None
    if bart_access_enabled and biblical_language != "grc":
        raise click.ClickException("BART MCP access is only supported for grc runs.")
    few_shot_example_paths = list(few_shot_example_paths or [])
    if few_shot_example_paths:
        log(
            "Loaded "
            f"{len(few_shot_example_paths)} few-shot example file(s): "
            + ", ".join(str(path) for path in few_shot_example_paths)
        )
    few_shot_examples_text = render_few_shot_examples(few_shot_example_paths)
    system_message = build_chapter_system_message(
        schema=schema,
        few_shot_examples_text=few_shot_examples_text,
        notes_text=notes_text,
        use_expert_notes=use_expert_notes,
        bart_access_enabled=bart_access_enabled,
        mode_label=mode_label,
        book=book,
        chapter=chapter,
    )
    emit_system_prompt(system_message)
    agent = build_agent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        schema=schema,
        system_message=system_message,
    )

    existing_rows, completed_keys = load_checkpoint_rows(output_csv, default_book=book)
    all_rows: list[dict[str, Any]] = list(existing_rows)
    context_history: list[dict[str, Any]] = build_context_history_from_rows(existing_rows)
    verse_plan = [
        verse_record
        for verse_record in verse_records
        if (book.upper(), verse_record["chapter"], verse_record["verse"]) not in completed_keys
    ]

    log(
        f"Loaded {len(existing_rows)} existing row(s) from {output_csv} and will resume with {len(verse_plan)} verse(s)."
        if existing_rows
        else f"Starting fresh with {len(verse_plan)} verse(s)."
    )
    if not verse_plan:
        log("All verses are already present in the checkpoint CSV; nothing new to generate.", level="RESUME")

    for verse_record in verse_records:
        verse_key = (book.upper(), verse_record["chapter"], verse_record["verse"])
        if verse_key in completed_keys:
            log(f"Skipping completed verse {verse_record['reference']}", level="RESUME")

    total_pending = len(verse_plan)

    for offset, verse_record in enumerate(verse_plan, start=1):
        verse_label = verse_record["reference"]
        log(f"Generating verse {format_verse_progress(verse_record, offset, total_pending)}")
        if bart_access_enabled:
            bart_annotation_count = len(verse_record.get("bart_annotations", []))
            log(f"Loaded {bart_annotation_count} BART annotation(s) for {verse_label}", level="BART")
        prompt = build_verse_prompt(
            schema=schema,
            verse_record=verse_record,
            context_history=context_history,
            context_window=context_window,
            bart_access_enabled=bart_access_enabled,
            mode_label=mode_label,
        )
        if stream:
            raw_content = None
            click.echo(f"[STREAM] {verse_label} model output follows")
            async for message in agent.run_stream(task=prompt):
                print_stream_chunk(message, verse_label=verse_label)
                content = getattr(message, "content", None)
                if content is not None and (not isinstance(content, str) or content.strip()):
                    raw_content = content
            if raw_content is None:
                raise click.ClickException(f"Model returned no content while generating {verse_label}.")
            payload = extract_json_payload(raw_content)
        else:
            log(f"Calling model for {verse_label}")
            result = await agent.run(task=prompt)
            payload = extract_json_payload(result.messages[-1].content)
        verse_rows = rows_from_payload(payload)
        verse_rows = [fill_row_metadata(row, verse_record) for row in verse_rows]
        all_rows.extend(verse_rows)
        write_checkpoint_csv(output_csv, all_rows, schema)
        log(f"Saved checkpoint after {verse_label} with {len(all_rows)} total row(s).")

        context_history.append(
            {
                "reference": verse_record["reference"],
                "biblical_text": verse_record.get("biblical_text", ""),
                "macula_tokens": verse_record.get("macula_tokens", []),
                "bart_annotations": verse_record.get("bart_annotations", []),
                "rows": verse_rows,
            }
        )

    return all_rows
