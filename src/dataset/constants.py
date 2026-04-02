"""Shared constants for the dataset package."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LANG_ROOT = REPO_ROOT / "lang"
DEFAULT_BIBLICAL_LANGUAGE = "heb"
DEFAULT_CONTEXT_WINDOW = 4
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_GROUP_KEYS = ["book", "chapter", "verse"]
DEFAULT_ROWS_KEY = "rows"
CSV_METADATA_COLUMNS = [
    "book",
    "chapter",
    "verse",
    "verse_reference",
    "biblical_text",
]
REFERENCE_RE = re.compile(
    r"^(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)(?:-\d+)?$"
)
PREFERRED_ROW_KEYS = [
    "token_id",
    "macula_token_id",
    "biblical_text",
    "verse_reference",
    "reference",
    "row_id",
    "annotation_id",
    "segment_id",
    "segment",
    "term",
]
INTEGER_LIKE_COLUMNS = {
    "chapter",
    "verse",
    "row_id",
    "annotation_id",
    "score",
    "index",
    "order",
    "sequence",
    "position",
}
BOOLEAN_LIKE_PREFIXES = ("is_", "has_", "was_", "were_", "should_", "can_", "does_")
BOOLEAN_LIKE_COLUMNS = {
    "accepted",
    "approved",
    "correct",
    "dft_preserved",
    "intervened",
    "missing",
    "present",
    "required",
    "valid",
}
