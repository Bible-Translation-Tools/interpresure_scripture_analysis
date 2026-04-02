"""Config and credential helpers for the dataset CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource
from dotenv import load_dotenv
try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in some envs
    yaml = None

from model.config import get_config_for_model

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if yaml is None:
        raise click.ClickException("PyYAML is required to use --config.")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise click.ClickException(f"YAML config must contain a mapping at the top level: {config_path}")
    return loaded


def get_command_config(config_data: dict[str, Any], command_name: str) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: value
        for key, value in config_data.items()
        if not isinstance(value, dict)
    }

    common = config_data.get("common")
    if isinstance(common, dict):
        merged.update(common)

    command_specific = config_data.get(command_name)
    if isinstance(command_specific, dict):
        merged.update(command_specific)

    return merged


def resolve_config_path(config_path: Path | None, value: Any) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute() or config_path is None:
        return candidate
    return (config_path.parent / candidate).resolve()


def resolve_config_paths(config_path: Path | None, values: dict[str, Any], keys: list[str] | tuple[str, ...]) -> dict[str, Any]:
    resolved = dict(values)
    for key in keys:
        if key in resolved and resolved[key] is not None:
            resolved[key] = resolve_config_path(config_path, resolved[key])
    return resolved


def config_source_is_default(ctx: click.Context, name: str) -> bool:
    return ctx.get_parameter_source(name) in {ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP}


def config_or_current(
    ctx: click.Context,
    name: str,
    current: Any,
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    path_like: bool = False,
    transform: Any = None,
) -> Any:
    if not config_source_is_default(ctx, name):
        return current
    if name not in config:
        return current

    value = config[name]
    if path_like:
        value = resolve_config_path(config_path, value)
    if transform is not None and value is not None:
        value = transform(value)
    return value


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def normalize_group_keys(value: Any) -> list[str]:
    if value is None:
        return list(["book", "chapter", "verse"])
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def resolve_model_credentials(model: str, api_key: Any, base_url: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    resolved_api_key = api_key
    resolved_base_url = base_url

    if resolved_api_key is None:
        if config.get("api_key") is not None:
            resolved_api_key = config.get("api_key")
        elif config.get("api_key_env"):
            resolved_api_key = os.getenv(str(config["api_key_env"]))

    if resolved_base_url is None and config.get("base_url") is not None:
        resolved_base_url = config.get("base_url")

    if resolved_api_key is None or resolved_base_url is None:
        defaults = get_config_for_model(model)
        if resolved_api_key is None:
            resolved_api_key = defaults.get("key")
        if resolved_base_url is None:
            resolved_base_url = defaults.get("base_url")

    return resolved_api_key, resolved_base_url
