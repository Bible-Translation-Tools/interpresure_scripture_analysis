"""Tests for analysis/discourse.py — prompt building and discourse pass."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from analysis.discourse import (
    _build_discourse_prompt,
    _collect_all_annotations_markdown,
    _collect_bart_summary,
    run_discourse_pass,
)
from analysis.schemas import DiscourseMapObservation, DiscourseUnit


# ---------------------------------------------------------------------------
# _collect_all_annotations_markdown
# ---------------------------------------------------------------------------


class TestCollectAnnotationsMarkdown:
    def test_empty_records(self):
        assert _collect_all_annotations_markdown([]) == ""

    def test_no_annotations_field(self):
        records = [{"verse": 1, "biblical_text": "X"}]
        assert _collect_all_annotations_markdown(records) == ""

    def test_empty_annotation_skipped(self):
        records = [{"pragmatic_annotations": "   "}, {"pragmatic_annotations": ""}]
        assert _collect_all_annotations_markdown(records) == ""

    def test_single_annotation(self):
        records = [{"pragmatic_annotations": "## Some annotation\n- key: val"}]
        result = _collect_all_annotations_markdown(records)
        assert "Some annotation" in result

    def test_multiple_annotations_joined(self):
        records = [
            {"pragmatic_annotations": "Annotation A"},
            {"pragmatic_annotations": "Annotation B"},
        ]
        result = _collect_all_annotations_markdown(records)
        assert "Annotation A" in result
        assert "Annotation B" in result

    def test_non_string_annotation_skipped(self):
        records = [{"pragmatic_annotations": None}, {"pragmatic_annotations": 42}]
        # Should not crash; non-strings treated as missing
        result = _collect_all_annotations_markdown(records)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _collect_bart_summary
# ---------------------------------------------------------------------------


class TestCollectBartSummary:
    def test_empty_records(self):
        assert _collect_bart_summary([]) == ""

    def test_no_bart_field(self):
        records = [{"verse": 1}]
        assert _collect_bart_summary(records) == ""

    def test_none_bart_skipped(self):
        records = [{"bart_annotations": None}]
        assert _collect_bart_summary(records) == ""

    def test_single_bart_entry(self):
        records = [
            {
                "reference": "PHM 1:1",
                "bart_annotations": {"discourse_unit": "narrative", "level": 3},
            }
        ]
        result = _collect_bart_summary(records)
        assert "PHM 1:1" in result
        assert "discourse_unit" in result

    def test_multiple_bart_entries(self):
        records = [
            {"reference": "PHM 1:1", "bart_annotations": {"x": 1}},
            {"reference": "PHM 1:2", "bart_annotations": {"y": 2}},
        ]
        result = _collect_bart_summary(records)
        assert "PHM 1:1" in result
        assert "PHM 1:2" in result

    def test_non_serializable_bart_uses_str(self):
        class Unserializable:
            pass

        records = [{"reference": "PHM 1:1", "bart_annotations": {"obj": Unserializable()}}]
        # Should not raise
        result = _collect_bart_summary(records)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _build_discourse_prompt
# ---------------------------------------------------------------------------


class TestBuildDiscoursePrompt:
    @pytest.fixture
    def verse_records(self):
        return [
            {"reference": "PHM 1:1", "biblical_text": "Παῦλος", "translation_text": "Paul"},
            {"reference": "PHM 1:2", "biblical_text": "καὶ", "translation_text": "and"},
        ]

    def test_includes_all_verse_references(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc"
        )
        assert "PHM 1:1" in prompt
        assert "PHM 1:2" in prompt

    def test_includes_biblical_text(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc"
        )
        assert "Παῦλος" in prompt

    def test_includes_translation_text(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc"
        )
        assert "Paul" in prompt

    def test_language_label_greek(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc"
        )
        assert "Grc" in prompt or "grc" in prompt or "Greek" in prompt.lower()

    def test_includes_annotations_when_provided(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records,
            biblical_language="grc",
            interpresure_markdown="## Implicature: scalar",
        )
        assert "scalar" in prompt

    def test_skips_annotations_when_none(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc", interpresure_markdown=None
        )
        assert "Expert InterpreSure" not in prompt

    def test_skips_annotations_when_empty(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc", interpresure_markdown="  "
        )
        assert "Expert InterpreSure" not in prompt

    def test_includes_bart_when_provided(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records,
            biblical_language="grc",
            bart_summary="BART: unit=narrative",
        )
        assert "BART" in prompt

    def test_ends_with_task_instruction(self, verse_records):
        prompt = _build_discourse_prompt(
            verse_records=verse_records, biblical_language="grc"
        )
        assert "JSON" in prompt or "schema" in prompt.lower()

    def test_empty_records(self):
        prompt = _build_discourse_prompt(verse_records=[], biblical_language="grc")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# run_discourse_pass (async)
# ---------------------------------------------------------------------------


VALID_DISCOURSE_JSON = DiscourseMapObservation(
    argument_structure="Paul appeals to Philemon.",
    genre_notes="Epistle.",
    discourse_boundaries=[
        DiscourseUnit(verse_start=1, verse_end=7, description="Greeting"),
    ],
).model_dump_json()


class TestRunDiscoursePass:
    async def test_returns_discourse_map(self, mock_verse_records):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = VALID_DISCOURSE_JSON
        mock_agent.run = AsyncMock(return_value=mock_result)

        dm = await run_discourse_pass(
            verse_records=mock_verse_records,
            biblical_language="grc",
            discourse_agent=mock_agent,
        )
        assert isinstance(dm, DiscourseMapObservation)
        assert len(dm.discourse_boundaries) == 1

    async def test_strips_markdown_code_fence(self, mock_verse_records):
        wrapped = f"```json\n{VALID_DISCOURSE_JSON}\n```"
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = wrapped
        mock_agent.run = AsyncMock(return_value=mock_result)

        dm = await run_discourse_pass(
            verse_records=mock_verse_records,
            biblical_language="grc",
            discourse_agent=mock_agent,
        )
        assert isinstance(dm, DiscourseMapObservation)

    async def test_strips_code_fence_without_json_label(self, mock_verse_records):
        wrapped = f"```\n{VALID_DISCOURSE_JSON}\n```"
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = wrapped
        mock_agent.run = AsyncMock(return_value=mock_result)

        dm = await run_discourse_pass(
            verse_records=mock_verse_records,
            biblical_language="grc",
            discourse_agent=mock_agent,
        )
        assert isinstance(dm, DiscourseMapObservation)

    async def test_agent_called_once(self, mock_verse_records):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = VALID_DISCOURSE_JSON
        mock_agent.run = AsyncMock(return_value=mock_result)

        await run_discourse_pass(
            verse_records=mock_verse_records,
            biblical_language="grc",
            discourse_agent=mock_agent,
        )
        mock_agent.run.assert_called_once()

    async def test_empty_verse_records(self):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = VALID_DISCOURSE_JSON
        mock_agent.run = AsyncMock(return_value=mock_result)

        dm = await run_discourse_pass(
            verse_records=[],
            biblical_language="grc",
            discourse_agent=mock_agent,
        )
        assert isinstance(dm, DiscourseMapObservation)
