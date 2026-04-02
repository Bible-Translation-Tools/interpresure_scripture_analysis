"""`convert` command."""

from __future__ import annotations

from pathlib import Path

import click

from .common import build_common_config, config_or_current, log, records_from_dataframe, schema_from_csv, write_json
from ..constants import DEFAULT_ROWS_KEY


@click.command(name="convert")
@click.argument("csv_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file containing convert options.",
)
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output JSON path.")
@click.option(
    "--schema-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output JSON Schema path.",
)
@click.pass_context
def convert(
    ctx: click.Context,
    csv_path: Path | None,
    config: Path | None,
    json_out: Path | None,
    schema_out: Path | None,
) -> None:
    """Convert a CSV into JSON rows plus a schema sidecar."""
    config_data = build_common_config(config, "convert")
    csv_path = config_or_current(ctx, "csv_path", csv_path, config_data, config_path=config, path_like=True)
    json_out = config_or_current(ctx, "json_out", json_out, config_data, config_path=config, path_like=True)
    schema_out = config_or_current(ctx, "schema_out", schema_out, config_data, config_path=config, path_like=True)

    if csv_path is None:
        raise click.ClickException("Provide csv_path or set convert.csv_path in the YAML config.")

    log(f"Convert mode started for {csv_path}")
    df, schema = schema_from_csv(csv_path)
    records = records_from_dataframe(df)
    payload = {DEFAULT_ROWS_KEY: records}

    if json_out is None:
        json_out = csv_path.with_suffix(".json")
    if schema_out is None:
        schema_out = csv_path.with_suffix(".schema.json")

    write_json(json_out, payload)
    write_json(schema_out, schema)
    click.echo(f"Wrote {json_out}")
    click.echo(f"Wrote {schema_out}")
