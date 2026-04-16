"""Shared CLI helpers for pragmatic analysis commands."""

from __future__ import annotations

from pathlib import Path

from dataset.config import get_command_config, load_yaml_config


def build_common_config(config: Path | None, command_name: str) -> dict[str, object]:
    return get_command_config(load_yaml_config(config), command_name)
