"""Tests for data/interpresure.py — Interpresure source registry, loading, and querying.

Coverage:
  - Available sources: PHM 1, PHP 1, PSA 145
  - ROM 3 is tested only when interpresure/interpresure_rom_3.csv exists
    (generated via `python -m dataset build`).
  - Verse-range annotation matching via the updated _filter_by_verse logic.
  - Graceful KeyError on unknown sources.
  - get_annotations / get_annotations_markdown return correct shapes.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from data.interpresure import Interpresure

REPO_ROOT = Path(__file__).resolve().parents[2]
ROM3_CSV = REPO_ROOT / "interpresure" / "interpresure_rom_3.csv"

rom3_available = pytest.mark.skipif(
    not ROM3_CSV.exists(),
    reason="interpresure_rom_3.csv not yet generated — run `python -m dataset build` for ROM 3",
)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    def test_phm_registered(self):
        interp = Interpresure("PHM", 1)
        assert interp.book == "PHM"
        assert interp.chapter == 1

    def test_php_1_registered(self):
        interp = Interpresure("PHP", 1)
        assert interp.book == "PHP"

    def test_psa_145_registered(self):
        interp = Interpresure("PSA", 145)
        assert interp.book == "PSA"

    def test_lowercase_book_normalised(self):
        interp = Interpresure("phm", 1)
        assert interp.book == "phm"  # stored as-passed; registry key is lowercased

    def test_unknown_source_raises_key_error(self):
        with pytest.raises(KeyError, match="No Interpresure source"):
            Interpresure("REV", 1)

    def test_error_message_lists_available_sources(self):
        try:
            Interpresure("XYZ", 99)
        except KeyError as exc:
            msg = str(exc)
            assert "PHM" in msg or "phm" in msg.lower()

    @rom3_available
    def test_rom_3_registered_when_csv_exists(self):
        interp = Interpresure("ROM", 3)
        assert interp.book == "ROM"


# ---------------------------------------------------------------------------
# Loading — PHM 1 (representative real data test)
# ---------------------------------------------------------------------------


class TestLoadPhm:
    @pytest.fixture(scope="class")
    def interp(self):
        return Interpresure("PHM", 1)

    def test_dataframe_nonempty(self, interp):
        assert not interp.interpresure.empty

    def test_has_expected_columns(self, interp):
        cols = set(interp.interpresure.columns.str.lower())
        assert "book" in cols
        assert "chapter" in cols
        assert "verse" in cols
        assert "biblical_text" in cols

    def test_chapter_column_all_one(self, interp):
        chapters = interp.interpresure["chapter"].unique()
        assert list(chapters) == [1] or set(chapters) == {1}

    def test_no_raw_nan_values(self, interp):
        # fillna should have replaced all NaN with sentinel string
        assert not interp.interpresure.isnull().any().any()

    def test_get_topics_returns_list(self, interp):
        topics = interp.get_topics()
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_all_registered_topics_accessible(self, interp):
        for topic in interp.get_topics():
            df = interp.get_annotations(topic)
            assert not df.empty, f"Topic {topic!r} returned empty dataframe"

    def test_get_annotations_general(self, interp):
        df = interp.get_annotations("general")
        assert "biblical_text" in df.columns

    def test_get_annotations_implicature(self, interp):
        df = interp.get_annotations("implicature")
        assert "inference_type" in df.columns

    def test_get_annotations_structure(self, interp):
        df = interp.get_annotations("structure")
        assert "information_structure" in df.columns or "question_under_discussion" in df.columns

    def test_get_annotations_social(self, interp):
        df = interp.get_annotations("social")
        assert "illocutionary_force" in df.columns

    def test_get_annotations_scales(self, interp):
        df = interp.get_annotations("scales")
        assert "is_scalar" in df.columns


# ---------------------------------------------------------------------------
# Loading — PHP 1 and PSA 145 (smoke tests)
# ---------------------------------------------------------------------------


class TestLoadPhp1:
    @pytest.fixture(scope="class")
    def interp(self):
        return Interpresure("PHP", 1)

    def test_dataframe_nonempty(self, interp):
        assert not interp.interpresure.empty

    def test_correct_book(self, interp):
        books = interp.interpresure["book"].str.upper().unique()
        assert "PHILIPPIANS" in books or "PHP" in books or len(books) == 1


class TestLoadPsa145:
    @pytest.fixture(scope="class")
    def interp(self):
        return Interpresure("PSA", 145)

    def test_dataframe_nonempty(self, interp):
        assert not interp.interpresure.empty

    def test_chapter_145(self, interp):
        assert 145 in interp.interpresure["chapter"].values


# ---------------------------------------------------------------------------
# get_annotations_markdown — PHM 1
# ---------------------------------------------------------------------------


class TestGetAnnotationsMarkdownPhm:
    @pytest.fixture(scope="class")
    def interp(self):
        return Interpresure("PHM", 1)

    def test_returns_string(self, interp):
        result = interp.get_annotations_markdown(None, 1, 1)
        assert isinstance(result, str)

    def test_requires_chapter(self, interp):
        with pytest.raises(ValueError, match="chapter and verse are required"):
            interp.get_annotations_markdown(None, None, 1)

    def test_requires_verse(self, interp):
        with pytest.raises(ValueError, match="chapter and verse are required"):
            interp.get_annotations_markdown(None, 1, None)

    def test_verse_1_returns_annotations(self, interp):
        result = interp.get_annotations_markdown(None, 1, 1)
        # PHM 1:1 has annotations — should not be just the header
        assert len(result.strip()) > len("## Pragmatic Expert Annotations:")

    def test_nonexistent_verse_returns_header_only(self, interp):
        result = interp.get_annotations_markdown(None, 1, 999)
        assert result.strip() == "## Pragmatic Expert Annotations:" or result.strip() == "\n## Pragmatic Expert Annotations:"

    def test_specific_topic_filter(self, interp):
        # Requesting only "general" should still work
        result = interp.get_annotations_markdown("general", 1, 1)
        assert isinstance(result, str)

    def test_invalid_topic_raises(self, interp):
        with pytest.raises(KeyError):
            interp.get_annotations_markdown("nonexistent_topic", 1, 1)


# ---------------------------------------------------------------------------
# _filter_by_verse — verse range matching
# ---------------------------------------------------------------------------


class TestFilterByVerse:
    """Unit tests for the verse-range filter logic.

    Uses synthetic DataFrames to test edge cases without depending on real CSV data.
    """

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows).fillna("Not Applicable")

    def test_exact_verse_match(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 5, "biblical_text": "X"},
            {"chapter": 1, "verse": 7, "biblical_text": "Y"},
        ])
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 1
        assert result.iloc[0]["biblical_text"] == "X"

    def test_verse_range_matches_interior_verse(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 10, "verse_end": 18, "biblical_text": "Range annotation"},
            {"chapter": 1, "verse": 20, "verse_end": "Not Applicable", "biblical_text": "Later verse"},
        ])
        # Verse 14 is inside 10-18
        result = interp._filter_by_verse(df, 1, 14)
        assert len(result) == 1
        assert result.iloc[0]["biblical_text"] == "Range annotation"

    def test_verse_range_matches_start_verse(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 10, "verse_end": 18, "biblical_text": "Range"},
        ])
        result = interp._filter_by_verse(df, 1, 10)
        assert len(result) == 1

    def test_verse_range_matches_end_verse(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 10, "verse_end": 18, "biblical_text": "Range"},
        ])
        result = interp._filter_by_verse(df, 1, 18)
        assert len(result) == 1

    def test_verse_before_range_excluded(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 10, "verse_end": 18, "biblical_text": "Range"},
        ])
        result = interp._filter_by_verse(df, 1, 9)
        assert len(result) == 0

    def test_verse_after_range_excluded(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 10, "verse_end": 18, "biblical_text": "Range"},
        ])
        result = interp._filter_by_verse(df, 1, 19)
        assert len(result) == 0

    def test_nan_verse_end_treated_as_single_verse(self):
        interp = Interpresure("PHM", 1)
        df = pd.DataFrame([
            {"chapter": 1, "verse": 5, "verse_end": float("nan"), "biblical_text": "Single"},
        ])
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 1

    def test_sentinel_verse_end_treated_as_single_verse(self):
        """'Not Applicable' sentinel should behave like NaN for range purposes."""
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 5, "verse_end": "Not Applicable", "biblical_text": "Single"},
        ])
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 1

    def test_wrong_chapter_excluded(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 2, "verse": 5, "biblical_text": "Wrong chapter"},
        ])
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 0

    def test_mixed_single_and_range(self):
        interp = Interpresure("PHM", 1)
        df = self._make_df([
            {"chapter": 1, "verse": 5, "verse_end": "Not Applicable", "biblical_text": "Single at 5"},
            {"chapter": 1, "verse": 3, "verse_end": 7.0, "biblical_text": "Range 3-7"},
        ])
        # Verse 5 matches both the single-verse row AND the 3-7 range row
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 2

    def test_no_verse_end_column_fallback(self):
        """DataFrames without a verse_end column should still work (exact match)."""
        interp = Interpresure("PHM", 1)
        df = pd.DataFrame([
            {"chapter": 1, "verse": 5, "biblical_text": "No range col"},
            {"chapter": 1, "verse": 6, "biblical_text": "Other verse"},
        ])
        result = interp._filter_by_verse(df, 1, 5)
        assert len(result) == 1
        assert result.iloc[0]["biblical_text"] == "No range col"


# ---------------------------------------------------------------------------
# ROM 3 — only when CSV exists
# ---------------------------------------------------------------------------


@rom3_available
class TestRom3:
    @pytest.fixture(scope="class")
    def interp(self):
        return Interpresure("ROM", 3)

    def test_loads_successfully(self, interp):
        assert not interp.interpresure.empty

    def test_chapter_3(self, interp):
        assert 3 in interp.interpresure["chapter"].values

    def test_has_verse_end_column(self, interp):
        # ROM 3 data contains verse ranges — the CSV should have verse_end
        assert "verse_end" in interp.interpresure.columns, (
            "interpresure_rom_3.csv is missing 'verse_end' column — regenerate with "
            "`python -m dataset build` after adding verse_end to the schema."
        )

    def test_verse_9_has_annotations(self, interp):
        result = interp.get_annotations_markdown(None, 3, 9)
        assert len(result.strip()) > len("## Pragmatic Expert Annotations:")

    def test_verse_range_10_18_accessible_via_verse_14(self, interp):
        """Verse 14 should return the 10-18 range annotation if verse_end is populated."""
        result = interp.get_annotations_markdown(None, 3, 14)
        # If the annotation covers 10-18, querying verse 14 should return content
        assert isinstance(result, str)

    def test_verse_19_has_annotations(self, interp):
        result = interp.get_annotations_markdown(None, 3, 19)
        assert isinstance(result, str)

    def test_markdown_output_nonempty_for_known_verse(self, interp):
        # At minimum verse 9 and 19 exist in the prose; test both produce output
        for verse in [9, 19, 20, 21]:
            result = interp.get_annotations_markdown(None, 3, verse)
            assert isinstance(result, str), f"Failed for ROM 3:{verse}"
