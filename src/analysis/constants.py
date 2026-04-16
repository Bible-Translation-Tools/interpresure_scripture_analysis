"""Shared constants for the pragmatic analysis CLI."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LANG_ROOT = REPO_ROOT / "lang"
DEFAULT_TRANSLATION_LANGUAGE = "en"
DEFAULT_TRANSLATION_TITLE = "ulb"
DEFAULT_BIBLICAL_LANGUAGE = "heb"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_CRITIC_MODEL = "gpt-5-mini"
