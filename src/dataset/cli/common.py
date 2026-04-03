"""Shared CLI helpers and imports."""

from __future__ import annotations

from pathlib import Path

from ..compare import compare_dataframes, comparison_summary_rows, print_comparison_summary
from ..config import (
    config_or_current,
    config_or_current_many,
    config_source_is_default,
    get_command_config,
    load_yaml_config,
    normalize_group_keys,
    resolve_model_credentials,
    to_bool,
    to_int,
)
from ..constants import (
    DEFAULT_BIBLICAL_LANGUAGE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_GROUP_KEYS,
    DEFAULT_LANG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_ROWS_KEY,
)
from ..generation import generate_rows_for_verses, log
from ..schema import load_schema, records_from_dataframe, schema_from_csv, write_json
from ..usfm import load_scripture_data, normalize_biblical_language


def build_common_config(config: Path | None, command_name: str) -> dict[str, object]:
    return get_command_config(load_yaml_config(config), command_name)
