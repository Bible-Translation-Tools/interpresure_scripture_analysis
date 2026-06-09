"""Shared CLI helpers for translator QA commands."""

from __future__ import annotations

from pathlib import Path

from dataset.config import get_command_config, load_yaml_config


def build_common_config(config: Path | None, command_name: str) -> dict:
    return get_command_config(load_yaml_config(config), command_name)
