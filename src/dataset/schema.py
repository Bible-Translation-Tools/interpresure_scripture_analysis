"""Schema and JSON helpers for dataset conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import click
import pandas as pd

from .constants import (
    BOOLEAN_LIKE_COLUMNS,
    BOOLEAN_LIKE_PREFIXES,
    CSV_METADATA_COLUMNS,
    DEFAULT_GROUP_KEYS,
    DEFAULT_ROWS_KEY,
    INTEGER_LIKE_COLUMNS,
)


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple, set)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (dict, list, tuple, set)):
        return value

    if isinstance(value, float) and math.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, str):
        return " ".join(value.split())

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    return value


def json_safe(value: Any) -> Any:
    value = normalize_scalar(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def clean_value_for_comparison(value: Any) -> Any:
    value = normalize_scalar(value)
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list)):
        return json_safe(value)
    return value


def infer_json_type(series: pd.Series, column_name: str) -> str:
    column = column_name.strip().lower()
    non_null = series.dropna()

    if column in INTEGER_LIKE_COLUMNS:
        return "integer"

    if column.startswith(BOOLEAN_LIKE_PREFIXES) or column in BOOLEAN_LIKE_COLUMNS:
        return "boolean"

    if non_null.empty:
        return "string"

    sample_values = []
    for value in non_null.head(50).tolist():
        value = normalize_scalar(value)
        if value is None:
            continue
        sample_values.append(value)

    if not sample_values:
        return "string"

    if all(isinstance(value, bool) for value in sample_values):
        return "boolean"

    if all(isinstance(value, int) and not isinstance(value, bool) for value in sample_values):
        return "integer"

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in sample_values):
        return "number"

    if all(isinstance(value, str) for value in sample_values):
        lowered = [value.strip().lower() for value in sample_values]
        if lowered and all(item in {"true", "false", "yes", "no", "1", "0"} for item in lowered):
            return "boolean"
        if all(item.replace("-", "").replace(".", "", 1).isdigit() for item in lowered):
            if all("." not in item for item in lowered):
                return "integer"
            return "number"

    return "string"


def column_description(column_name: str) -> str:
    if column_name.strip().lower() == "token_id":
        return "Comma-separated Macula xml:id values for the snippet"
    return column_name.replace("_", " ").strip().capitalize()


def infer_schema_from_dataframe(
    df: pd.DataFrame,
    *,
    title: str = "DatasetRows",
    rows_key: str = DEFAULT_ROWS_KEY,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    normalized_df = normalize_dataframe_columns(df)

    properties: dict[str, Any] = {}
    required: list[str] = []
    column_types: dict[str, str] = {}

    for column in normalized_df.columns:
        inferred_type = infer_json_type(normalized_df[column], column)
        column_types[column] = inferred_type
        properties[column] = {
            "type": inferred_type,
            "description": column_description(column),
        }
        required.append(column)

    row_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": title,
        "type": "object",
        "properties": {
            rows_key: {
                "type": "array",
                "items": row_schema,
            }
        },
        "required": [rows_key],
        "additionalProperties": False,
        "x-column-order": list(normalized_df.columns),
        "x-column-types": column_types,
        "x-group-keys": group_keys or DEFAULT_GROUP_KEYS,
    }
    return schema


def schema_from_csv(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    schema = infer_schema_from_dataframe(df, title=csv_path.stem)
    return df, schema


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = normalize_dataframe_columns(df)
    records = normalized.to_dict(orient="records")
    return [{key: json_safe(value) for key, value in record.items()} for record in records]


def load_schema(schema_path: Path | None, csv_path: Path | None = None, json_path: Path | None = None) -> dict[str, Any]:
    if schema_path is not None:
        return json.loads(schema_path.read_text(encoding="utf-8"))

    if json_path is not None:
        sidecar = json_path.with_suffix(".schema.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        rows_key = DEFAULT_ROWS_KEY if DEFAULT_ROWS_KEY in loaded else "analysis"
        rows = loaded.get(rows_key, [])
        if rows:
            frame = pd.DataFrame(rows)
            return infer_schema_from_dataframe(frame, title=json_path.stem, rows_key=rows_key)

    if csv_path is not None:
        _, schema = schema_from_csv(csv_path)
        return schema

    raise click.ClickException("A schema, CSV, or JSON file is required.")


def schema_properties(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties", {})
    rows_schema = properties.get(DEFAULT_ROWS_KEY, {})
    items = rows_schema.get("items", {})
    row_properties = items.get("properties", {})
    order = schema.get("x-column-order")
    if isinstance(order, list) and order:
        return [str(column) for column in order]
    return list(row_properties.keys())


def response_format_from_schema(schema: dict[str, Any], *, name: str = "dataset_rows") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def output_columns(schema: dict[str, Any]) -> list[str]:
    schema_columns = schema_properties(schema)
    ordered = list(CSV_METADATA_COLUMNS)
    for column in schema_columns:
        if column not in ordered:
            ordered.append(column)
    return ordered
