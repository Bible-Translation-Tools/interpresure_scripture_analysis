"""`build` command — expert-guided (few-shot) pragmatic analysis."""

from __future__ import annotations

from pathlib import Path

import click

from dataset.config import config_or_current, resolve_model_credentials, to_bool, to_int

from ..constants import (
    DEFAULT_ANALYSIS_TYPE,
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_DISCOURSE_BOUNDARY_MARKERS,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSLATION_LANGUAGE,
    DEFAULT_TRANSLATION_TITLE,
    REPO_ID,
    REPO_NAME,
)
from ..workflow import run_analysis_sync
from .common import build_common_config


@click.command(name="build")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option("--book", default=None, help="Bible book code, e.g. PHM, PSA.")
@click.option("--chapter", type=int, default=None, help="Chapter number.")
@click.option("--translation-language", default=DEFAULT_TRANSLATION_LANGUAGE, show_default=True)
@click.option("--translation-title", default=DEFAULT_TRANSLATION_TITLE, show_default=True)
@click.option(
    "--biblical-language",
    type=click.Choice(["grc", "heb", "greek", "hebrew"], case_sensitive=False),
    default=DEFAULT_BIBLICAL_LANGUAGE,
    show_default=True,
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
    help="Parent directory for run output. A timestamped subdirectory is created per run.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--critic-model", default=DEFAULT_CRITIC_MODEL, show_default=True)
@click.option("--analysis-type", default=DEFAULT_ANALYSIS_TYPE, show_default=True,
              help="Observation type to produce (e.g. interpresure_suggestions, translator_questions).")
@click.option(
    "--discourse-boundary-markers/--no-discourse-boundary-markers",
    default=DEFAULT_DISCOURSE_BOUNDARY_MARKERS,
    show_default=True,
    help="Inject discourse boundary markers into the verse loop.",
)
@click.option("--repo-id", default=REPO_ID, show_default=True)
@click.option("--repo-name", default=REPO_NAME, show_default=True)
@click.option("--api-key", default=None)
@click.option("--base-url", default=None)
@click.pass_context
def build(
    ctx: click.Context,
    config: Path | None,
    book: str | None,
    chapter: int | None,
    translation_language: str,
    translation_title: str,
    biblical_language: str,
    usfm_root: Path,
    macula_db_path: Path | None,
    bart_db_path: Path | None,
    output_dir: Path,
    model: str,
    critic_model: str,
    analysis_type: str,
    discourse_boundary_markers: bool,
    repo_id: str,
    repo_name: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Run expert-guided (few-shot) pragmatic analysis for a chapter."""
    cfg = build_common_config(config, "build")

    book = config_or_current(ctx, "book", book, cfg)
    chapter = config_or_current(ctx, "chapter", chapter, cfg, transform=to_int)
    translation_language = config_or_current(ctx, "translation_language", translation_language, cfg)
    translation_title = config_or_current(ctx, "translation_title", translation_title, cfg)
    biblical_language = config_or_current(ctx, "biblical_language", biblical_language, cfg)
    usfm_root = config_or_current(ctx, "usfm_root", usfm_root, cfg, config_path=config, path_like=True)
    macula_db_path = config_or_current(ctx, "macula_db_path", macula_db_path, cfg, config_path=config, path_like=True)
    bart_db_path = config_or_current(ctx, "bart_db_path", bart_db_path, cfg, config_path=config, path_like=True)
    output_dir = config_or_current(ctx, "output_dir", output_dir, cfg, config_path=config, path_like=True)
    model = config_or_current(ctx, "model", model, cfg)
    critic_model = config_or_current(ctx, "critic_model", critic_model, cfg)
    analysis_type = config_or_current(ctx, "analysis_type", analysis_type, cfg)
    discourse_boundary_markers = config_or_current(
        ctx, "discourse_boundary_markers", discourse_boundary_markers, cfg, transform=to_bool
    )
    repo_id = config_or_current(ctx, "repo_id", repo_id, cfg)
    repo_name = config_or_current(ctx, "repo_name", repo_name, cfg)
    api_key = config_or_current(ctx, "api_key", api_key, cfg)
    base_url = config_or_current(ctx, "base_url", base_url, cfg)
    api_key, base_url = resolve_model_credentials(str(model), api_key, base_url, cfg)

    missing = []
    if not book:
        missing.append("book")
    if chapter is None:
        missing.append("chapter")
    if missing:
        raise click.ClickException("Provide or configure: " + ", ".join(missing))
    if (macula_db_path or bart_db_path) and str(biblical_language).lower() not in {"grc", "greek"}:
        raise click.ClickException("macula_db_path and bart_db_path require biblical_language=grc.")

    click.echo(f"[INFO] Expert-guided analysis: {str(book).upper()} {chapter} "
               f"({analysis_type}, {model})")

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
        output_dir=Path(output_dir),
        macula_db_path=Path(macula_db_path) if macula_db_path else None,
        bart_db_path=Path(bart_db_path) if bart_db_path else None,
        analysis_mode="few_shot",
        analysis_type=str(analysis_type),
        use_expert_materials=True,
        discourse_boundary_markers=bool(discourse_boundary_markers),
        repo_id=str(repo_id),
        repo_name=str(repo_name),
    )

    click.echo(f"[INFO] Run written to: {result['run_dir']}")
