"""`build` command for expert-guided pragmatic analysis."""

from __future__ import annotations

from pathlib import Path

import click

from dataset.config import config_or_current, resolve_model_credentials, to_int

from ..constants import (
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_TRANSLATION_LANGUAGE,
    DEFAULT_TRANSLATION_TITLE,
)
from ..workflow import run_analysis_sync
from .common import build_common_config


@click.command(name="build")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing build options.",
)
@click.option("--book", default=None, help="Bible book code, such as PHM or PSA.")
@click.option("--chapter", type=int, default=None, help="Optional chapter number.")
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
    "--macula-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional Macula SQLite database used to preload token ids for Greek runs.",
)
@click.option(
    "--bart-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional BART discourse-analysis SQLite database used to preload annotations for Greek runs.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory used to generate timestamped CSV and JSON filenames.",
)
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output CSV path.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Final JSON output path.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model name to use.")
@click.option("--critic-model", default=DEFAULT_CRITIC_MODEL, show_default=True, help="Critic model name to use.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Base URL override.")
@click.pass_context
def build(
    ctx: click.Context,
    config: Path | None,
    book: str,
    chapter: int | None,
    translation_language: str,
    translation_title: str,
    biblical_language: str,
    usfm_root: Path,
    macula_db_path: Path | None,
    bart_db_path: Path | None,
    output_dir: Path | None,
    output_csv: Path | None,
    output_json: Path | None,
    model: str,
    critic_model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Run expert-guided pragmatic analysis and write the final JSON output."""
    config_data = build_common_config(config, "build")
    book = config_or_current(ctx, "book", book, config_data)
    chapter = config_or_current(ctx, "chapter", chapter, config_data, transform=to_int)
    translation_language = config_or_current(ctx, "translation_language", translation_language, config_data)
    translation_title = config_or_current(ctx, "translation_title", translation_title, config_data)
    biblical_language = config_or_current(ctx, "biblical_language", biblical_language, config_data)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, config_data, config_path=config, path_like=True)
    macula_db_path = config_or_current(ctx, "macula_db_path", macula_db_path, config_data, config_path=config, path_like=True)
    bart_db_path = config_or_current(ctx, "bart_db_path", bart_db_path, config_data, config_path=config, path_like=True)
    output_dir = config_or_current(ctx, "output_dir", output_dir, config_data, config_path=config, path_like=True)
    output_csv = config_or_current(ctx, "output_csv", output_csv, config_data, config_path=config, path_like=True)
    output_json = config_or_current(ctx, "output_json", output_json, config_data, config_path=config, path_like=True)
    model = config_or_current(ctx, "model", model, config_data)
    critic_model = config_or_current(ctx, "critic_model", critic_model, config_data)
    api_key = config_or_current(ctx, "api_key", api_key, config_data)
    base_url = config_or_current(ctx, "base_url", base_url, config_data)
    api_key, base_url = resolve_model_credentials(str(model), api_key, base_url, config_data)

    missing = []
    if book is None:
        missing.append("book")
    if chapter is None:
        missing.append("chapter")
    if output_dir is None and output_csv is None and output_json is None:
        missing.append("output_dir, output_csv, or output_json")
    if missing:
        raise click.ClickException("Provide or configure: " + ", ".join(missing))
    if (macula_db_path is not None or bart_db_path is not None) and str(biblical_language).lower() not in {"grc", "greek"}:
        raise click.ClickException("macula_db_path and bart_db_path may only be used when biblical_language is grc.")

    if output_csv is None and output_json is not None:
        output_csv = Path(output_json).with_suffix(".csv")
    elif output_json is None and output_csv is not None:
        output_json = Path(output_csv).with_suffix(".json")

    click.echo(f"[INFO] Expert-guided analysis started for {str(book).upper()} chapter {chapter}")

    result = run_analysis_sync(
        book=str(book),
        chapter=int(chapter),
        biblical_language=str(biblical_language),
        translation_language=str(translation_language),
        translation_title=str(translation_title),
        usfm_root=Path(usfm_root),
        model=str(model),
        critic_model=str(critic_model),
        api_key=api_key,
        base_url=base_url,
        output_dir=Path(output_dir) if output_dir is not None else None,
        macula_db_path=Path(macula_db_path) if macula_db_path is not None else None,
        bart_db_path=Path(bart_db_path) if bart_db_path is not None else None,
        output_csv=Path(output_csv) if output_csv is not None else None,
        output_json=Path(output_json) if output_json is not None else None,
        analysis_mode="few-shot",
        use_expert_materials=True,
    )

    click.echo(f"[INFO] Wrote final JSON to {result['output_json']}")
    click.echo(f"[INFO] Wrote intermediate CSV to {result['output_csv']}")
    click.echo(f"[INFO] Wrote chapter evaluation JSON to {result['evaluation_json']}")
