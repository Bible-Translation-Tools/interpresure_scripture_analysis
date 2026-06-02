"""Tests for analysis/workflow.py — helper functions and critic loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from analysis.schemas import (
    CriticReview,
    DiscourseUnit,
    InterpreSureSuggestionsObservation,
)
from analysis.workflow import (
    _analyze_verse_with_critic,
    _build_boundary_marker,
    _build_chapter_context_prompt,
    _build_chapter_summary_prompt,
    _build_verse_prompt,
    _parse_observation,
)


# ---------------------------------------------------------------------------
# _build_verse_prompt
# ---------------------------------------------------------------------------


class TestBuildVersePrompt:
    @pytest.fixture
    def verse_record(self):
        return {
            "reference": "PHM 1:10",
            "biblical_text": "παρακαλῶ σε",
            "translation_text": "I appeal to you",
            "pragmatic_annotations": "## Implicature\n- inference_type: indirect request",
        }

    def test_includes_reference(self, verse_record):
        prompt = _build_verse_prompt(verse_record, "grc")
        assert "PHM 1:10" in prompt

    def test_includes_biblical_text(self, verse_record):
        prompt = _build_verse_prompt(verse_record, "grc")
        assert "παρακαλῶ" in prompt

    def test_includes_translation(self, verse_record):
        prompt = _build_verse_prompt(verse_record, "grc")
        assert "I appeal to you" in prompt

    def test_includes_annotations_when_present(self, verse_record):
        prompt = _build_verse_prompt(verse_record, "grc")
        assert "indirect request" in prompt

    def test_no_annotations_section_when_absent(self):
        record = {"reference": "PHM 1:1", "biblical_text": "X", "translation_text": "Y"}
        prompt = _build_verse_prompt(record, "grc")
        assert "InterpreSure" not in prompt

    def test_includes_macula_when_present(self):
        record = {
            "reference": "PHM 1:1",
            "biblical_text": "X",
            "translation_text": "Y",
            "macula_tokens": [{"lemma": "παρακαλέω", "pos": "verb"}],
        }
        prompt = _build_verse_prompt(record, "grc")
        assert "MACULA" in prompt
        assert "παρακαλέω" in prompt

    def test_includes_bart_when_present(self):
        record = {
            "reference": "PHM 1:1",
            "biblical_text": "X",
            "translation_text": "Y",
            "bart_annotations": {"unit": "narrative"},
        }
        prompt = _build_verse_prompt(record, "grc")
        assert "BART" in prompt

    def test_ends_with_task_instruction(self, verse_record):
        prompt = _build_verse_prompt(verse_record, "grc")
        assert "JSON" in prompt or "schema" in prompt.lower()


# ---------------------------------------------------------------------------
# _build_boundary_marker
# ---------------------------------------------------------------------------


class TestBuildBoundaryMarker:
    def test_contains_verse_range(self):
        unit = DiscourseUnit(verse_start=8, verse_end=16, description="Central appeal")
        marker = _build_boundary_marker(unit)
        assert "8" in marker
        assert "16" in marker

    def test_contains_description(self):
        unit = DiscourseUnit(verse_start=8, verse_end=16, description="Central appeal for Onesimus")
        marker = _build_boundary_marker(unit)
        assert "Central appeal for Onesimus" in marker

    def test_is_string(self):
        unit = DiscourseUnit(verse_start=1, verse_end=7, description="Greeting")
        assert isinstance(_build_boundary_marker(unit), str)


# ---------------------------------------------------------------------------
# _build_chapter_context_prompt
# ---------------------------------------------------------------------------


class TestBuildChapterContextPrompt:
    def test_includes_full_chapter_text(self, mock_verse_records, phm_discourse_map):
        prompt = _build_chapter_context_prompt(mock_verse_records, phm_discourse_map, "grc")
        assert "PHM 1:1" in prompt
        assert "PHM 1:2" in prompt
        assert "PHM 1:3" in prompt

    def test_includes_discourse_map_json(self, mock_verse_records, phm_discourse_map):
        prompt = _build_chapter_context_prompt(mock_verse_records, phm_discourse_map, "grc")
        assert "discourse_map" in prompt
        assert "argument_structure" in prompt

    def test_includes_discourse_boundaries(self, mock_verse_records, phm_discourse_map):
        prompt = _build_chapter_context_prompt(mock_verse_records, phm_discourse_map, "grc")
        assert "Central appeal for Onesimus" in prompt

    def test_includes_await_instruction(self, mock_verse_records, phm_discourse_map):
        prompt = _build_chapter_context_prompt(mock_verse_records, phm_discourse_map, "grc")
        assert "first verse" in prompt.lower() or "await" in prompt.lower()


# ---------------------------------------------------------------------------
# _build_chapter_summary_prompt
# ---------------------------------------------------------------------------


class TestBuildChapterSummaryPrompt:
    def test_returns_string(self):
        assert isinstance(_build_chapter_summary_prompt(), str)

    def test_mentions_score(self):
        prompt = _build_chapter_summary_prompt()
        assert "score" in prompt.lower() or "1–10" in prompt

    def test_mentions_verses_to_review(self):
        prompt = _build_chapter_summary_prompt()
        assert "verses_to_review" in prompt or "verses" in prompt.lower()


# ---------------------------------------------------------------------------
# _parse_observation (async)
# ---------------------------------------------------------------------------


VALID_OBS_JSON = InterpreSureSuggestionsObservation(
    strengths=["Good"], weaknesses=[], suggestions=[], score=8, confidence=90,
    reasoning="Solid.", cross_references=[], verses_to_review=[]
).model_dump_json()


class TestParseObservation:
    async def test_parses_plain_json(self):
        result = MagicMock()
        result.text = VALID_OBS_JSON
        obs = await _parse_observation(result, InterpreSureSuggestionsObservation)
        assert obs.score == 8
        assert obs.confidence == 90

    async def test_strips_json_code_fence(self):
        result = MagicMock()
        result.text = f"```json\n{VALID_OBS_JSON}\n```"
        obs = await _parse_observation(result, InterpreSureSuggestionsObservation)
        assert isinstance(obs, InterpreSureSuggestionsObservation)

    async def test_strips_plain_code_fence(self):
        result = MagicMock()
        result.text = f"```\n{VALID_OBS_JSON}\n```"
        obs = await _parse_observation(result, InterpreSureSuggestionsObservation)
        assert isinstance(obs, InterpreSureSuggestionsObservation)

    async def test_handles_whitespace(self):
        result = MagicMock()
        result.text = f"   {VALID_OBS_JSON}   "
        obs = await _parse_observation(result, InterpreSureSuggestionsObservation)
        assert isinstance(obs, InterpreSureSuggestionsObservation)

    async def test_falls_back_to_str_result(self):
        result = VALID_OBS_JSON  # plain string, no .text attribute
        obs = await _parse_observation(result, InterpreSureSuggestionsObservation)
        assert isinstance(obs, InterpreSureSuggestionsObservation)


# ---------------------------------------------------------------------------
# _analyze_verse_with_critic (async)
# ---------------------------------------------------------------------------


def _make_analyst_result(obs: InterpreSureSuggestionsObservation):
    result = MagicMock()
    result.text = obs.model_dump_json()
    return result


class TestAnalyzeVerseWithCritic:
    @pytest.fixture
    def verse_record(self):
        return {
            "reference": "PHM 1:10",
            "biblical_text": "παρακαλῶ",
            "translation_text": "I appeal",
        }

    @pytest.fixture
    def good_obs(self):
        return InterpreSureSuggestionsObservation(
            strengths=["Good"], weaknesses=[], suggestions=[], score=8, confidence=90,
            reasoning="Fine.", cross_references=[], verses_to_review=[]
        )

    async def test_returns_observation_on_first_accept(self, verse_record, good_obs):
        analyst = MagicMock()
        analyst.run = AsyncMock(return_value=_make_analyst_result(good_obs))

        critic = MagicMock()
        review = CriticReview(accepted=True, reasoning="Sound.")
        critic_result = MagicMock()
        critic_result.text = review.model_dump_json()
        critic.run = AsyncMock(return_value=critic_result)

        session = MagicMock()
        obs = await _analyze_verse_with_critic(
            analyst=analyst,
            critic=critic,
            session=session,
            verse_record=verse_record,
            biblical_language="grc",
            output_schema=InterpreSureSuggestionsObservation,
            max_rounds=3,
        )
        assert isinstance(obs, InterpreSureSuggestionsObservation)
        assert obs.score == 8

    async def test_revision_called_on_rejection(self, verse_record, good_obs):
        analyst = MagicMock()
        analyst.run = AsyncMock(return_value=_make_analyst_result(good_obs))

        # Reject once, then accept
        reject = CriticReview(accepted=False, reasoning="Please revise.")
        accept = CriticReview(accepted=True, reasoning="Now good.")
        critic_results = [
            MagicMock(text=reject.model_dump_json()),
            MagicMock(text=accept.model_dump_json()),
        ]
        call_count = {"n": 0}

        async def critic_run(prompt, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return critic_results[min(idx, len(critic_results) - 1)]

        critic = MagicMock()
        critic.run = critic_run

        session = MagicMock()
        obs = await _analyze_verse_with_critic(
            analyst=analyst,
            critic=critic,
            session=session,
            verse_record=verse_record,
            biblical_language="grc",
            output_schema=InterpreSureSuggestionsObservation,
            max_rounds=3,
        )
        # Analyst should have been called at least twice (initial + revision)
        assert analyst.run.call_count >= 2
        assert isinstance(obs, InterpreSureSuggestionsObservation)

    async def test_returns_last_observation_at_max_rounds(self, verse_record, good_obs):
        analyst = MagicMock()
        analyst.run = AsyncMock(return_value=_make_analyst_result(good_obs))

        # Always reject
        reject = CriticReview(accepted=False, reasoning="Not good enough.")
        critic_result = MagicMock()
        critic_result.text = reject.model_dump_json()
        critic = MagicMock()
        critic.run = AsyncMock(return_value=critic_result)

        session = MagicMock()
        obs = await _analyze_verse_with_critic(
            analyst=analyst,
            critic=critic,
            session=session,
            verse_record=verse_record,
            biblical_language="grc",
            output_schema=InterpreSureSuggestionsObservation,
            max_rounds=2,
        )
        # Should still return an observation, not raise
        assert isinstance(obs, InterpreSureSuggestionsObservation)

    async def test_analyst_called_with_session(self, verse_record, good_obs):
        analyst = MagicMock()
        analyst.run = AsyncMock(return_value=_make_analyst_result(good_obs))

        accept = CriticReview(accepted=True, reasoning="Good.")
        critic_result = MagicMock()
        critic_result.text = accept.model_dump_json()
        critic = MagicMock()
        critic.run = AsyncMock(return_value=critic_result)

        session = MagicMock(spec=["some_session_attr"])
        await _analyze_verse_with_critic(
            analyst=analyst,
            critic=critic,
            session=session,
            verse_record=verse_record,
            biblical_language="grc",
            output_schema=InterpreSureSuggestionsObservation,
            max_rounds=3,
        )
        # Verify session was passed to analyst.run
        call_kwargs = analyst.run.call_args
        assert call_kwargs is not None
        # session should appear in kwargs or positional
        args, kwargs = call_kwargs
        assert kwargs.get("session") is session or session in args
