"""`compare` command."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from .common import (
    build_common_config,
    comparison_summary_rows,
    compare_dataframes,
    config_or_current,
    config_source_is_default,
    log,
    normalize_group_keys,
    print_comparison_summary,
    write_json,
)
from ..constants import DEFAULT_GROUP_KEYS


@click.command(name="compare")
@click.argument("manual_csv", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ai_csv", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing compare options.",
)
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional detailed report path.")
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional summary CSV path.")
@click.option(
    "--group-keys",
    multiple=True,
    default=tuple(DEFAULT_GROUP_KEYS),
    show_default=True,
    help="Group columns used to align rows before comparison.",
)
@click.pass_context
def compare(
    ctx: click.Context,
    manual_csv: Path | None,
    ai_csv: Path | None,
    config: Path | None,
    output_json: Path | None,
    output_csv: Path | None,
    group_keys: tuple[str, ...],
) -> None:
    """Compare a hand-made CSV to an AI-generated CSV grouped by verse."""
    config_data = build_common_config(config, "compare")
    manual_csv = config_or_current(ctx, "manual_csv", manual_csv, config_data, config_path=config, path_like=True)
    ai_csv = config_or_current(ctx, "ai_csv", ai_csv, config_data, config_path=config, path_like=True)
    output_json = config_or_current(ctx, "output_json", output_json, config_data, config_path=config, path_like=True)
    output_csv = config_or_current(ctx, "output_csv", output_csv, config_data, config_path=config, path_like=True)
    if config_source_is_default(ctx, "group_keys") and "group_keys" in config_data:
        group_keys = tuple(normalize_group_keys(config_data["group_keys"]))
    else:
        group_keys = tuple(normalize_group_keys(group_keys))

    if manual_csv is None or ai_csv is None:
        raise click.ClickException("Provide manual_csv and ai_csv or set them in the YAML config.")

    log(f"Compare mode started for {manual_csv} vs {ai_csv}")
    manual_df = pd.read_csv(manual_csv)
    ai_df = pd.read_csv(ai_csv)
    report = compare_dataframes(manual_df, ai_df, group_keys=list(group_keys))

    print_comparison_summary(report)

    if output_json is not None:
        write_json(output_json, report)
        click.echo(f"Wrote {output_json}")

    if output_csv is not None:
        pd.DataFrame(comparison_summary_rows(report)).to_csv(output_csv, index=False)
        click.echo(f"Wrote {output_csv}")
