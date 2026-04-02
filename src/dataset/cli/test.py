"""`test` command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from .common import (
    build_common_config,
    config_or_current,
    generate_rows_for_verses,
    load_schema,
    load_scripture_data,
    log,
    normalize_biblical_language,
    resolve_model_credentials,
    to_bool,
    to_int,
    write_json,
)
from ..constants import (
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_ROWS_KEY,
)


@click.command(name="test")
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
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing test options.",
)
@click.option("--book", default=None, help="Bible book code, such as PHM or PSA.")
@click.option("--chapter", type=int, default=None, help="Optional chapter number. Omit to process the full book.")
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
    help="Language folder for the original-language USFM file.",
)
@click.option(
    "--macula-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional Macula SQLite database used to preload token ids for each verse.",
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
@click.option(
    "--stream/--no-stream",
    default=True,
    show_default=True,
    help="Stream model outputs to the console while generating.",
)
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output CSV path.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional raw JSON output path.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model name to use.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Base URL override.")
@click.pass_context
def test(
    ctx: click.Context,
    template_csv: Path | None,
    json_file: Path | None,
    schema_file: Path | None,
    config: Path | None,
    book: str,
    chapter: int | None,
    biblical_language: str,
    macula_db_path: Path | None,
    usfm_root: Path,
    context_window: int,
    stream: bool,
    output_csv: Path,
    output_json: Path | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Generate a test dataset from verse text only, without expert notes."""
    config_data = build_common_config(config, "test")
    template_csv = config_or_current(ctx, "template_csv", template_csv, config_data, config_path=config, path_like=True)
    json_file = config_or_current(ctx, "json_file", json_file, config_data, config_path=config, path_like=True)
    schema_file = config_or_current(ctx, "schema_file", schema_file, config_data, config_path=config, path_like=True)
    book = config_or_current(ctx, "book", book, config_data)
    chapter = config_or_current(ctx, "chapter", chapter, config_data, transform=to_int)
    biblical_language = config_or_current(ctx, "biblical_language", biblical_language, config_data)
    macula_db_path = config_or_current(ctx, "macula_db_path", macula_db_path, config_data, config_path=config, path_like=True)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, config_data, config_path=config, path_like=True)
    context_window = config_or_current(ctx, "context_window", context_window, config_data, transform=to_int)
    stream = config_or_current(ctx, "stream", stream, config_data, transform=to_bool)
    output_csv = config_or_current(ctx, "output_csv", output_csv, config_data, config_path=config, path_like=True)
    output_json = config_or_current(ctx, "output_json", output_json, config_data, config_path=config, path_like=True)
    model = config_or_current(ctx, "model", model, config_data)
    api_key = config_or_current(ctx, "api_key", api_key, config_data)
    base_url = config_or_current(ctx, "base_url", base_url, config_data)
    api_key, base_url = resolve_model_credentials(str(model), api_key, base_url, config_data)

    missing = []
    if book is None:
        missing.append("book")
    if output_csv is None:
        missing.append("output_csv")
    if schema_file is None and template_csv is None and json_file is None:
        missing.append("template_csv/json_file/schema_file")
    if missing:
        raise click.ClickException("Provide or configure: " + ", ".join(missing))

    log(f"Test mode started for {str(book).upper()} chapter {chapter if chapter is not None else 'all'}")

    schema = load_schema(schema_file, csv_path=template_csv, json_path=json_file)
    normalized_biblical_language = normalize_biblical_language(str(biblical_language))
    _, verse_records, biblical_path = load_scripture_data(
        book=str(book),
        chapter=chapter,
        biblical_language=normalized_biblical_language,
        usfm_root=Path(usfm_root),
    )
    log(f"Biblical USFM: {biblical_path}")
    log(f"Loaded {len(verse_records)} overlapping verse(s) for generation.")

    if not verse_records:
        chapter_text = f" chapter {chapter}" if chapter is not None else ""
        raise click.ClickException(f"No overlapping verses found for {book.upper()}{chapter_text}.")

    rows = asyncio.run(
        generate_rows_for_verses(
            schema=schema,
            verse_records=verse_records,
            output_csv=Path(output_csv),
            book=str(book),
            chapter=chapter,
            model_name=str(model),
            api_key=api_key,
            base_url=base_url,
            macula_db_path=macula_db_path,
            notes_text=None,
            use_expert_notes=False,
            context_window=context_window,
            mode_label=f"{str(book).upper()} verse-only test generation",
            stream=stream,
        )
    )

    if output_json is not None:
        write_json(
            output_json,
            {
                DEFAULT_ROWS_KEY: rows,
                "book": str(book).upper(),
                "chapter": chapter,
                "biblical_language": normalized_biblical_language,
                "biblical_usfm": str(biblical_path),
            },
        )
        click.echo(f"Wrote {output_json}")
    log(f"Final CSV written to {output_csv} with {len(rows)} row(s).")
