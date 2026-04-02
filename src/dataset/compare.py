"""Comparison helpers for manual and AI-generated CSVs."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

import click
import pandas as pd

from .constants import DEFAULT_GROUP_KEYS, PREFERRED_ROW_KEYS
from .schema import clean_value_for_comparison, normalize_dataframe_columns


def coerce_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_dataframe_columns(df)


def group_dataframe(df: pd.DataFrame, group_keys: Iterable[str]) -> pd.core.groupby.DataFrameGroupBy:
    missing = [key for key in group_keys if key not in df.columns]
    if missing:
        raise click.ClickException(f"Missing required comparison columns: {', '.join(missing)}")
    return df.groupby(list(group_keys), dropna=False, sort=True)


def candidate_row_key_columns(df_manual: pd.DataFrame, df_ai: pd.DataFrame, group_keys: list[str]) -> list[str]:
    common = [column for column in df_manual.columns if column in df_ai.columns and column not in group_keys]
    preferred = [column for column in PREFERRED_ROW_KEYS if column in common]
    if preferred:
        return preferred

    return common


def is_unique_key(df: pd.DataFrame, columns: list[str]) -> bool:
    if not columns:
        return False
    subset = df[columns].copy()
    return not subset.duplicated().any()


def select_row_key(df_manual: pd.DataFrame, df_ai: pd.DataFrame, group_keys: list[str]) -> list[str] | None:
    candidates = candidate_row_key_columns(df_manual, df_ai, group_keys)
    for column in PREFERRED_ROW_KEYS:
        if column not in candidates:
            continue
        if is_unique_key(df_manual, [column]) and is_unique_key(df_ai, [column]):
            return [column]

    for column in candidates:
        if is_unique_key(df_manual, [column]) and is_unique_key(df_ai, [column]):
            return [column]

    if len(candidates) >= 2:
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                candidate = [candidates[left], candidates[right]]
                if is_unique_key(df_manual, candidate) and is_unique_key(df_ai, candidate):
                    return candidate

    return None


def row_signature(row: pd.Series, columns: list[str]) -> str:
    signature = {column: clean_value_for_comparison(row[column]) for column in columns}
    return json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)


def compare_without_key(
    manual_group: pd.DataFrame,
    ai_group: pd.DataFrame,
    compare_columns: list[str],
) -> dict[str, Any]:
    manual_signatures = Counter(row_signature(row, compare_columns) for _, row in manual_group.iterrows())
    ai_signatures = Counter(row_signature(row, compare_columns) for _, row in ai_group.iterrows())

    missing = []
    extra = []

    for signature, count in (manual_signatures - ai_signatures).items():
        missing.extend([json.loads(signature)] * count)

    for signature, count in (ai_signatures - manual_signatures).items():
        extra.extend([json.loads(signature)] * count)

    return {
        "comparison_mode": "multiset",
        "matched_rows": sum((manual_signatures & ai_signatures).values()),
        "manual_only_rows": missing,
        "ai_only_rows": extra,
        "row_diffs": [],
    }


def compare_with_key(
    manual_group: pd.DataFrame,
    ai_group: pd.DataFrame,
    key_columns: list[str],
    compare_columns: list[str],
) -> dict[str, Any]:
    manual_index = manual_group.set_index(key_columns, drop=False)
    ai_index = ai_group.set_index(key_columns, drop=False)

    manual_keys = set(manual_index.index.tolist())
    ai_keys = set(ai_index.index.tolist())

    row_diffs = []
    for key in sorted(manual_keys & ai_keys, key=lambda value: str(value)):
        manual_row = manual_index.loc[key]
        ai_row = ai_index.loc[key]

        if isinstance(manual_row, pd.DataFrame) or isinstance(ai_row, pd.DataFrame):
            continue

        changed_fields = {}
        for column in compare_columns:
            manual_value = clean_value_for_comparison(manual_row[column])
            ai_value = clean_value_for_comparison(ai_row[column])
            if manual_value != ai_value:
                changed_fields[column] = {
                    "manual": manual_value,
                    "ai": ai_value,
                }

        if changed_fields:
            row_diffs.append(
                {
                    "key": {column: clean_value_for_comparison(manual_row[column]) for column in key_columns},
                    "differences": changed_fields,
                }
            )

    return {
        "comparison_mode": "keyed",
        "matched_rows": len(manual_keys & ai_keys),
        "manual_only_rows": [
            {column: clean_value_for_comparison(row[column]) for column in manual_group.columns}
            for key, row in manual_index.iterrows()
            if key not in ai_keys
        ],
        "ai_only_rows": [
            {column: clean_value_for_comparison(row[column]) for column in ai_group.columns}
            for key, row in ai_index.iterrows()
            if key not in manual_keys
        ],
        "row_diffs": row_diffs,
        "key_columns": key_columns,
    }


def compare_dataframes(
    manual_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    *,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    group_keys = group_keys or DEFAULT_GROUP_KEYS
    manual_df = coerce_dataframe_columns(manual_df)
    ai_df = coerce_dataframe_columns(ai_df)

    common_columns = [column for column in manual_df.columns if column in ai_df.columns]
    manual_only_columns = [column for column in manual_df.columns if column not in ai_df.columns]
    ai_only_columns = [column for column in ai_df.columns if column not in manual_df.columns]

    grouped_manual = group_dataframe(manual_df, group_keys)
    grouped_ai = group_dataframe(ai_df, group_keys)

    verse_keys = sorted(
        set(grouped_manual.groups.keys()) | set(grouped_ai.groups.keys()),
        key=lambda value: str(value),
    )
    verse_reports = []

    for verse_key in verse_keys:
        manual_group = grouped_manual.get_group(verse_key) if verse_key in grouped_manual.groups else pd.DataFrame(columns=manual_df.columns)
        ai_group = grouped_ai.get_group(verse_key) if verse_key in grouped_ai.groups else pd.DataFrame(columns=ai_df.columns)

        compare_columns = [column for column in common_columns if column not in group_keys]
        row_key = select_row_key(manual_group, ai_group, group_keys)

        if row_key:
            comparison = compare_with_key(manual_group, ai_group, row_key, compare_columns)
        else:
            comparison = compare_without_key(manual_group, ai_group, compare_columns)

        report = {
            "book": verse_key[0] if len(verse_key) > 0 else None,
            "chapter": verse_key[1] if len(verse_key) > 1 else None,
            "verse": verse_key[2] if len(verse_key) > 2 else None,
            "manual_rows": int(len(manual_group)),
            "ai_rows": int(len(ai_group)),
            "manual_only_columns": manual_only_columns,
            "ai_only_columns": ai_only_columns,
            **comparison,
        }
        verse_reports.append(report)

    summary = {
        "group_keys": group_keys,
        "manual_row_count": int(len(manual_df)),
        "ai_row_count": int(len(ai_df)),
        "manual_only_columns": manual_only_columns,
        "ai_only_columns": ai_only_columns,
        "verse_reports": verse_reports,
    }
    return summary


def comparison_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for verse in report["verse_reports"]:
        rows.append(
            {
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "manual_rows": verse["manual_rows"],
                "ai_rows": verse["ai_rows"],
                "comparison_mode": verse["comparison_mode"],
                "manual_only_rows": len(verse["manual_only_rows"]),
                "ai_only_rows": len(verse["ai_only_rows"]),
                "row_diffs": len(verse["row_diffs"]),
            }
        )
    return rows


def print_comparison_summary(report: dict[str, Any]) -> None:
    mismatched = [
        verse
        for verse in report["verse_reports"]
        if verse["manual_only_rows"] or verse["ai_only_rows"] or verse["row_diffs"]
    ]
    click.echo(f"Compared {len(report['verse_reports'])} verse groups.")
    click.echo(f"Manual rows: {report['manual_row_count']}")
    click.echo(f"AI rows: {report['ai_row_count']}")
    click.echo(f"Verse groups with differences: {len(mismatched)}")
    if report["manual_only_columns"]:
        click.echo(f"Columns only in manual CSV: {', '.join(report['manual_only_columns'])}")
    if report["ai_only_columns"]:
        click.echo(f"Columns only in AI CSV: {', '.join(report['ai_only_columns'])}")
