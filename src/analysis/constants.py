"""Shared constants for the pragmatic analysis pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Path defaults
DEFAULT_LANG_ROOT = REPO_ROOT / "lang"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "runs"

# Language defaults
DEFAULT_TRANSLATION_LANGUAGE = "en"
DEFAULT_BIBLICAL_LANGUAGE = "grc"

# Model defaults
DEFAULT_MODEL = "gpt-4o"
DEFAULT_CRITIC_MODEL = "gpt-4o-mini"

# Analysis defaults
DEFAULT_ANALYSIS_TYPE = "interpresure_suggestions"
DEFAULT_DISCOURSE_BOUNDARY_MARKERS = True

# Critic review loop
MAX_CRITIC_ROUNDS = 3
