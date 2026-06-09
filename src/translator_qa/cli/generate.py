"""`generate` command — produce diagnostic questions from InterpreSure annotations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from dataset.config import config_or_current, resolve_model_credentials, to_int

from ..constants import (
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTIONS_DIR,
)
from ..generate import run_question_generation
from .common import build_common_config


@click.command(name="generate")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option("--book", default=None, help="Bible book code, e.g. PHM, ROM.")
@click.option("--chapter", type=int, default=None, help="Chapter number.")
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
    help="Original language for question generation.",
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
    help="Parent directory for the scripture-analysis-api run directory.",
)
@click.option(
    "--questions-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_QUESTIONS_DIR,
    show_default=True,
    help="Directory for the plain JSON question file.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--api-key", default=None)
@click.option("--base-url", default=None)
@click.pass_context
def generate(
    ctx: click.Context,
    config: Path | None,
    book: str | None,
    chapter: int | None,
    biblical_language: str,
    usfm_root: Path,
    macula_db_path: Path | None,
    bart_db_path: Path | None,
    output_dir: Path,
    questions_dir: Path,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Generate plain-language diagnostic questions for a chapter.

    Reads the original language USFM and InterpreSure annotations to produce
    questions that Mother Tongue Translators can use to evaluate their drafts.

    Outputs:
      - A plain JSON question file (input for the `answer` command).
      - A run directory in scripture-analysis-api format.
    """
    cfg = build_common_config(config, "generate")

    book = config_or_current(ctx, "book", book, cfg)
    chapter = config_or_current(ctx, "chapter", chapter, cfg, transform=to_int)
    biblical_language = config_or_current(ctx, "biblical_language", biblical_language, cfg)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, cfg, config_path=config, path_like=True)
    macula_db_path = config_or_current(ctx, "macula_db_path", macula_db_path, cfg, config_path=config, path_like=True)
    bart_db_path = config_or_current(ctx, "bart_db_path", bart_db_path, cfg, config_path=config, path_like=True)
    output_dir = config_or_current(ctx, "output_dir", output_dir, cfg, config_path=config, path_like=True)
    questions_dir = config_or_current(ctx, "questions_dir", questions_dir, cfg, config_path=config, path_like=True)
    model = config_or_current(ctx, "model", model, cfg)
    api_key = config_or_current(ctx, "api_key", api_key, cfg)
    base_url = config_or_current(ctx, "base_url", base_url, cfg)
    api_key, base_url = resolve_model_credentials(str(model), api_key, base_url, cfg)

    if bart_db_path and str(biblical_language).lower() not in {"grc", "greek"}:
        click.echo(f"[INFO] Ignoring bart_db_path — BART is Greek NT only")
        bart_db_path = None

    missing = []
    if not book:
        missing.append("book")
    if chapter is None:
        missing.append("chapter")
    if missing:
        raise click.ClickException("Provide or configure: " + ", ".join(missing))

    click.echo(f"[INFO] Generating questions: {str(book).upper()} {chapter} ({model})")

    result = asyncio.run(
        run_question_generation(
            book=str(book),
            chapter=int(chapter),
            biblical_language=str(biblical_language),
            usfm_root=Path(usfm_root),
            model=str(model),
            api_key=api_key,
            base_url=base_url,
            macula_db_path=Path(macula_db_path) if macula_db_path else None,
            bart_db_path=Path(bart_db_path) if bart_db_path else None,
            output_dir=Path(output_dir),
            questions_dir=Path(questions_dir),
        )
    )

    qs = result["question_set"]
    click.echo(f"[INFO] {len(qs.questions)} question(s) written to: {result['questions_file']}")
    click.echo(f"[INFO] Run directory: {result['run_dir']}")
