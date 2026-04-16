"""`compare` command for pragmatic analysis JSON outputs."""

from __future__ import annotations

from pathlib import Path

import click

from dataset.config import config_or_current

from report.compare_pragmatic_analysis import comparison_summary_rows, compare_pragmatic_analysis_files, print_comparison_summary

from .common import build_common_config


@click.command(name="compare")
@click.argument("left_json", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("right_json", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing compare options.",
)
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional detailed report path.")
@click.option("--output-csv", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional summary CSV path.")
@click.pass_context
def compare(
    ctx: click.Context,
    left_json: Path | None,
    right_json: Path | None,
    config: Path | None,
    output_json: Path | None,
    output_csv: Path | None,
) -> None:
    """Compare two final pragmatic analysis JSON files."""
    config_data = build_common_config(config, "compare")
    left_json = config_or_current(ctx, "left_json", left_json, config_data, config_path=config, path_like=True)
    right_json = config_or_current(ctx, "right_json", right_json, config_data, config_path=config, path_like=True)
    output_json = config_or_current(ctx, "output_json", output_json, config_data, config_path=config, path_like=True)
    output_csv = config_or_current(ctx, "output_csv", output_csv, config_data, config_path=config, path_like=True)

    if left_json is None or right_json is None:
        raise click.ClickException("Provide left_json and right_json or set them in the YAML config.")

    click.echo(f"[INFO] Compare mode started for {left_json} vs {right_json}")
    report = compare_pragmatic_analysis_files(Path(left_json), Path(right_json))
    print_comparison_summary(report)

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"Wrote {output_json}")

    if output_csv is not None:
        import pandas as pd

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(comparison_summary_rows(report)).to_csv(output_csv, index=False)
        click.echo(f"Wrote {output_csv}")
