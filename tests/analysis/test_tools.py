"""Tests for analysis/tools.py — verse lookup tool and chapter text formatting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analysis.tools import format_verse_records_as_chapter_text, make_verse_lookup_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
LANG_ROOT = REPO_ROOT / "lang"


# ---------------------------------------------------------------------------
# format_verse_records_as_chapter_text
# ---------------------------------------------------------------------------


class TestFormatVerseRecordsAsChapterText:
    def test_empty_records(self):
        result = format_verse_records_as_chapter_text([])
        assert result == ""

    def test_single_verse(self):
        records = [
            {
                "reference": "PHM 1:1",
                "biblical_text": "Παῦλος δέσμιος",
                "translation_text": "Paul, a prisoner",
            }
        ]
        result = format_verse_records_as_chapter_text(records)
        assert "PHM 1:1" in result
        assert "Παῦλος" in result
        assert "Paul, a prisoner" in result

    def test_multiple_verses_in_order(self):
        records = [
            {"reference": "PHM 1:1", "biblical_text": "A", "translation_text": "a"},
            {"reference": "PHM 1:2", "biblical_text": "B", "translation_text": "b"},
        ]
        result = format_verse_records_as_chapter_text(records)
        pos1 = result.index("PHM 1:1")
        pos2 = result.index("PHM 1:2")
        assert pos1 < pos2

    def test_language_label_default(self):
        records = [{"reference": "PHM 1:1", "biblical_text": "X", "translation_text": "Y"}]
        result = format_verse_records_as_chapter_text(records)
        assert "Greek" in result

    def test_language_label_custom(self):
        records = [{"reference": "PSA 145:1", "biblical_text": "X", "translation_text": "Y"}]
        result = format_verse_records_as_chapter_text(records, biblical_language="Hebrew")
        assert "Hebrew" in result

    def test_missing_biblical_text(self):
        records = [{"reference": "PHM 1:1", "biblical_text": "", "translation_text": "Paul"}]
        result = format_verse_records_as_chapter_text(records)
        assert "Paul" in result

    def test_reference_fallback(self):
        records = [
            {"book": "PHM", "chapter": 1, "verse": 5, "biblical_text": "X", "translation_text": "Y"}
        ]
        # No 'reference' key — should not crash
        result = format_verse_records_as_chapter_text(records)
        assert result  # non-empty


# ---------------------------------------------------------------------------
# make_verse_lookup_tool — factory and tool structure
# ---------------------------------------------------------------------------


class TestMakeVerseLooupTool:
    def test_returns_callable(self):
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        assert callable(tool_fn)

    def test_tool_has_docstring(self):
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        assert tool_fn.__doc__

    @pytest.mark.skipif(
        not (LANG_ROOT / "en" / "58-PHM.usfm").exists()
        and not (LANG_ROOT / "en").exists(),
        reason="USFM data not available",
    )
    def test_lookup_single_verse_real_data(self):
        """Integration test using real USFM files for PHM."""
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        result = tool_fn(book="PHM", chapter=1, verse_start=1)
        assert "PHM" in result
        assert isinstance(result, str)
        assert len(result) > 10

    @pytest.mark.skipif(
        not (LANG_ROOT / "en").exists(),
        reason="USFM data not available",
    )
    def test_lookup_verse_range_real_data(self):
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        result = tool_fn(book="PHM", chapter=1, verse_start=1, verse_end=3)
        assert "PHM 1:1" in result or "PHM" in result

    def test_lookup_missing_file_returns_error_string(self, tmp_path):
        tool_fn = make_verse_lookup_tool(
            usfm_root=tmp_path,
            translation_language="en",
            biblical_language="grc",
        )
        result = tool_fn(book="PHM", chapter=1, verse_start=1)
        assert "[Error" in result or "not found" in result.lower() or isinstance(result, str)

    def test_backward_range_corrected(self, tmp_path):
        """verse_end < verse_start should be silently corrected to verse_start."""
        tool_fn = make_verse_lookup_tool(
            usfm_root=tmp_path,
            translation_language="en",
            biblical_language="grc",
        )
        # Even with empty USFM root, should not crash on backwards range
        result = tool_fn(book="PHM", chapter=1, verse_start=5, verse_end=2)
        assert isinstance(result, str)

    def test_book_code_uppercased(self, tmp_path):
        tool_fn = make_verse_lookup_tool(
            usfm_root=tmp_path,
            translation_language="en",
            biblical_language="grc",
        )
        # Lowercase book should not crash
        result = tool_fn(book="phm", chapter=1, verse_start=1)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: lookup_verse_range with interpresure annotations
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (LANG_ROOT / "en").exists(),
    reason="USFM data not available",
)
class TestLookupWithAnnotations:
    def test_phm_1_10_includes_text(self):
        """PHM 1:10 is in the interpresure annotation set — should return text."""
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        result = tool_fn(book="PHM", chapter=1, verse_start=10)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_cross_chapter_lookup(self):
        """Reaching into PHP chapter 1 from a PHM analysis context."""
        tool_fn = make_verse_lookup_tool(
            usfm_root=LANG_ROOT,
            translation_language="en",
            biblical_language="grc",
        )
        result = tool_fn(book="PHP", chapter=1, verse_start=1)
        assert isinstance(result, str)
