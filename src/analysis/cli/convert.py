"""`convert` command for pragmatic analysis CSVs."""

from __future__ import annotations

from pathlib import Path

import click

from dataset.config import config_or_current
from dataset.usfm import resolve_usfm_file

from report import convert_pragmatic_analysis
from ..constants import DEFAULT_LANG_ROOT, DEFAULT_TRANSLATION_LANGUAGE, DEFAULT_TRANSLATION_TITLE
from ..workflow import finalize

from .common import build_common_config


@click.command(name="convert")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing convert options.",
)
@click.option(
    "--input-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Existing analysis CSV to convert into the final JSON output.",
)
@click.option("--book", default=None, help="Bible book code, such as PHM or PSA.")
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Final JSON output path. Defaults to the CSV stem with .json.",
)
@click.option(
    "--translation-language",
    default=DEFAULT_TRANSLATION_LANGUAGE,
    show_default=True,
    help="Language folder for the translation USFM file.",
)
@click.option(
    "--translation-title",
    default=DEFAULT_TRANSLATION_TITLE,
    show_default=True,
    help="Translation title stored in the final JSON output.",
)
@click.option(
    "--usfm-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_LANG_ROOT,
    show_default=True,
    help="Root folder containing the lang subdirectories.",
)
@click.pass_context
def convert(
    ctx: click.Context,
    config: Path | None,
    input_csv: Path | None,
    book: str,
    output_json: Path | None,
    translation_language: str,
    translation_title: str,
    usfm_root: Path,
) -> None:
    """Convert an existing analysis CSV into the final JSON report."""
    config_data = build_common_config(config, "convert")
    input_csv = config_or_current(ctx, "input_csv", input_csv, config_data, config_path=config, path_like=True)
    book = config_or_current(ctx, "book", book, config_data)
    output_json = config_or_current(ctx, "output_json", output_json, config_data, config_path=config, path_like=True)
    translation_language = config_or_current(ctx, "translation_language", translation_language, config_data)
    translation_title = config_or_current(ctx, "translation_title", translation_title, config_data)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, config_data, config_path=config, path_like=True)

    missing = []
    if input_csv is None:
        missing.append("input_csv")
    if book is None:
        missing.append("book")
    if missing:
        raise click.ClickException("Provide or configure: " + ", ".join(missing))

    if output_json is None:
        output_json = Path(input_csv).with_suffix(".json")

    evaluation_json = Path(output_json).with_name(f"{Path(output_json).stem}_general_analysis.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Converting {input_csv} to chapter evaluation JSON")
    evaluation = convert_pragmatic_analysis.convert_pragmatic(
        individual_path=Path(input_csv),
        output_path=evaluation_json,
        interpresure=None,
        book=str(book),
    )
    translation_path = resolve_usfm_file(Path(usfm_root), str(translation_language), str(book))
    translation_usfm = translation_path.read_text(encoding="utf-8")
    finalize(
        Path(output_json),
        str(translation_language),
        str(translation_title),
        translation_usfm,
        [evaluation],
    )

    click.echo(f"[INFO] Wrote final JSON to {output_json}")
    click.echo(f"[INFO] Wrote chapter evaluation JSON to {evaluation_json}")
