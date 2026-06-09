"""Shared constants for the translator QA pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_LANG_ROOT = REPO_ROOT / "lang"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "runs"
DEFAULT_QUESTIONS_DIR = REPO_ROOT / "out" / "questions"

DEFAULT_TRANSLATION_LANGUAGE = "en"
DEFAULT_BIBLICAL_LANGUAGE = "grc"

DEFAULT_MODEL = "gpt-4o"
DEFAULT_ANALYSIS_TYPE = "translator_questions"
