"""`answer` command — answer diagnostic questions for a translation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from dataset.config import config_or_current, resolve_model_credentials

from ..constants import (
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSLATION_LANGUAGE,
)
from ..answer import run_qa_answering
from .common import build_common_config


@click.command(name="answer")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option(
    "--questions-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the question JSON file produced by the `generate` command.",
)
@click.option(
    "--translation-language",
    default=DEFAULT_TRANSLATION_LANGUAGE,
    show_default=True,
    help="Language dir for the translation to evaluate (must have repo_info.yaml).",
)
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
    help="Override biblical language (defaults to value in question file).",
)
@click.option(
    "--usfm-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_LANG_ROOT,
    show_default=True,
)
@click.option(
    "--macula-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--bart-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--api-key", default=None)
@click.option("--base-url", default=None)
@click.pass_context
def answer(
    ctx: click.Context,
    config: Path | None,
    questions_file: Path | None,
    translation_language: str,
    biblical_language: str,
    usfm_root: Path,
    macula_db_path: Path | None,
    bart_db_path: Path | None,
    output_dir: Path,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Answer diagnostic questions for a translation draft.

    Reads the question set produced by `generate`, then answers each question
    about the given translation, producing ``interpresure_qa`` observations in
    a scripture-analysis-api run directory.

    Can be run multiple times against different translations using the same
    question set.
    """
    cfg = build_common_config(config, "answer")

    questions_file = config_or_current(ctx, "questions_file", questions_file, cfg, config_path=config, path_like=True)
    translation_language = config_or_current(ctx, "translation_language", translation_language, cfg)
    biblical_language = config_or_current(ctx, "biblical_language", biblical_language, cfg)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, cfg, config_path=config, path_like=True)
    macula_db_path = config_or_current(ctx, "macula_db_path", macula_db_path, cfg, config_path=config, path_like=True)
    bart_db_path = config_or_current(ctx, "bart_db_path", bart_db_path, cfg, config_path=config, path_like=True)
    output_dir = config_or_current(ctx, "output_dir", output_dir, cfg, config_path=config, path_like=True)
    model = config_or_current(ctx, "model", model, cfg)
    api_key = config_or_current(ctx, "api_key", api_key, cfg)
    base_url = config_or_current(ctx, "base_url", base_url, cfg)
    api_key, base_url = resolve_model_credentials(str(model), api_key, base_url, cfg)

    if bart_db_path and str(biblical_language).lower() not in {"grc", "greek"}:
        click.echo("[INFO] Ignoring bart_db_path — BART is Greek NT only")
        bart_db_path = None

    if questions_file is None:
        raise click.ClickException("Provide --questions-file or set it in the YAML config.")

    click.echo(
        f"[INFO] Answering questions: {questions_file.name} "
        f"→ {translation_language} ({model})"
    )

    result = asyncio.run(
        run_qa_answering(
            questions_file=Path(questions_file),
            translation_language=str(translation_language),
            biblical_language=str(biblical_language),
            usfm_root=Path(usfm_root),
            model=str(model),
            api_key=api_key,
            base_url=base_url,
            macula_db_path=Path(macula_db_path) if macula_db_path else None,
            bart_db_path=Path(bart_db_path) if bart_db_path else None,
            output_dir=Path(output_dir),
        )
    )

    n = len(result["answers"])
    click.echo(f"[INFO] {n} answer(s) written to: {result['run_dir']}")
