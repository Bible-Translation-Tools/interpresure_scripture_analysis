"""Tests for analysis/schemas.py — Pydantic models and envelope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from analysis.schemas import (
    AnalysisItem,
    AnalysisRunMetadataObservation,
    CriticReview,
    DiscourseMapObservation,
    DiscourseUnit,
    InterpreSureSuggestionsObservation,
    ResourcesUsed,
    TranslationInfo,
    TranslatorQuestion,
    TranslatorQuestionsObservation,
)


# ---------------------------------------------------------------------------
# DiscourseMapObservation — boundary helpers
# ---------------------------------------------------------------------------


class TestDiscourseMapBoundaries:
    def test_boundary_start_verses_empty(self):
        dm = DiscourseMapObservation(argument_structure="x", genre_notes="y")
        assert dm.boundary_start_verses() == set()

    def test_boundary_start_verses_single(self):
        dm = DiscourseMapObservation(
            argument_structure="x",
            genre_notes="y",
            discourse_boundaries=[DiscourseUnit(verse_start=3, verse_end=7, description="unit")],
        )
        assert dm.boundary_start_verses() == {3}

    def test_boundary_start_verses_multiple(self, phm_discourse_map):
        assert phm_discourse_map.boundary_start_verses() == {1, 8, 17}

    def test_boundary_for_verse_match(self, phm_discourse_map):
        unit = phm_discourse_map.boundary_for_verse(8)
        assert unit is not None
        assert unit.description == "Central appeal for Onesimus"
        assert unit.verse_end == 16

    def test_boundary_for_verse_no_match(self, phm_discourse_map):
        assert phm_discourse_map.boundary_for_verse(5) is None

    def test_boundary_for_verse_empty_boundaries(self):
        dm = DiscourseMapObservation(argument_structure="x", genre_notes="y")
        assert dm.boundary_for_verse(1) is None

    def test_discourse_map_type_version_frozen(self):
        dm = DiscourseMapObservation(argument_structure="x", genre_notes="y")
        assert dm.type == "discourse_map"
        assert dm.version == "1.0"

    def test_discourse_map_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            DiscourseMapObservation(
                argument_structure="x", genre_notes="y", unexpected_field="bad"
            )


# ---------------------------------------------------------------------------
# InterpreSureSuggestionsObservation — validation
# ---------------------------------------------------------------------------


class TestInterpreSureSuggestions:
    def test_minimal_valid(self):
        obs = InterpreSureSuggestionsObservation(
            strengths=[], weaknesses=[], suggestions=[]
        )
        assert obs.type == "interpresure_suggestions"
        assert obs.version == "2.0"
        assert obs.score is None
        assert obs.confidence is None

    def test_full_valid(self, basic_suggestion):
        assert basic_suggestion.score == 7
        assert basic_suggestion.confidence == 85
        assert len(basic_suggestion.strengths) == 1

    def test_score_bounds_too_low(self):
        with pytest.raises(ValidationError):
            InterpreSureSuggestionsObservation(
                strengths=[], weaknesses=[], suggestions=[], score=0
            )

    def test_score_bounds_too_high(self):
        with pytest.raises(ValidationError):
            InterpreSureSuggestionsObservation(
                strengths=[], weaknesses=[], suggestions=[], score=11
            )

    def test_score_boundary_values(self):
        low = InterpreSureSuggestionsObservation(strengths=[], weaknesses=[], suggestions=[], score=1)
        high = InterpreSureSuggestionsObservation(strengths=[], weaknesses=[], suggestions=[], score=10)
        assert low.score == 1
        assert high.score == 10

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            InterpreSureSuggestionsObservation(
                strengths=[], weaknesses=[], suggestions=[], confidence=101
            )
        with pytest.raises(ValidationError):
            InterpreSureSuggestionsObservation(
                strengths=[], weaknesses=[], suggestions=[], confidence=-1
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            InterpreSureSuggestionsObservation(
                strengths=[], weaknesses=[], suggestions=[], unknown="bad"
            )

    def test_cross_references_default_empty(self):
        obs = InterpreSureSuggestionsObservation(strengths=[], weaknesses=[], suggestions=[])
        assert obs.cross_references == []

    def test_verses_to_review_default_empty(self):
        obs = InterpreSureSuggestionsObservation(strengths=[], weaknesses=[], suggestions=[])
        assert obs.verses_to_review == []


# ---------------------------------------------------------------------------
# TranslatorQuestionsObservation — validation
# ---------------------------------------------------------------------------


class TestTranslatorQuestions:
    def test_valid_single_question(self):
        obs = TranslatorQuestionsObservation(
            questions=[TranslatorQuestion(question="Does this convey an appeal?")]
        )
        assert obs.type == "translator_questions"
        assert obs.version == "1.0"
        assert len(obs.questions) == 1

    def test_empty_questions_rejected(self):
        with pytest.raises(ValidationError):
            TranslatorQuestionsObservation(questions=[])

    def test_question_rationale_optional(self):
        q = TranslatorQuestion(question="Is the tone deferential?")
        assert q.rationale is None
        assert q.annotation_topics == []

    def test_question_with_all_fields(self):
        q = TranslatorQuestion(
            question="Does it sound like a request?",
            rationale="The original uses indirect speech act.",
            annotation_topics=["social", "structure"],
        )
        assert q.rationale is not None
        assert "social" in q.annotation_topics


# ---------------------------------------------------------------------------
# AnalysisRunMetadataObservation — validation
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_valid(self, run_metadata):
        assert run_metadata.type == "analysis_run_metadata"
        assert run_metadata.version == "1.0"
        assert run_metadata.analysis_mode == "few_shot"
        assert run_metadata.resources.interpresure is True

    def test_invalid_analysis_mode(self):
        with pytest.raises(ValidationError):
            AnalysisRunMetadataObservation(
                model="gpt-4o",
                analysis_mode="invalid_mode",
                analysis_type="interpresure_suggestions",
                timestamp="2026-01-01T00:00:00+00:00",
                translation=TranslationInfo(language="en", title="ULT"),
                biblical_language="grc",
            )

    def test_invalid_biblical_language(self):
        with pytest.raises(ValidationError):
            AnalysisRunMetadataObservation(
                model="gpt-4o",
                analysis_mode="zero_shot",
                analysis_type="interpresure_suggestions",
                timestamp="2026-01-01T00:00:00+00:00",
                translation=TranslationInfo(language="en", title="ULT"),
                biblical_language="latin",
            )

    def test_resources_defaults_false(self):
        r = ResourcesUsed()
        assert not r.interpresure
        assert not r.bart
        assert not r.macula
        assert not r.discourse_boundary_markers


# ---------------------------------------------------------------------------
# AnalysisItem.from_observation — envelope factory
# ---------------------------------------------------------------------------


class TestAnalysisItemFromObservation:
    def test_verse_level(self, basic_suggestion):
        item = AnalysisItem.from_observation(
            basic_suggestion,
            book="PHM",
            chapter=1,
            anchor="PHM 1:10",
            anchor_level="verse",
        )
        assert item.type == "interpresure_suggestions"
        assert item.version == "2.0"
        assert item.anchor == "PHM 1:10"
        assert item.anchor_level == "verse"
        assert item.book == "PHM"
        assert item.chapter == 1
        assert item.observation["type"] == "interpresure_suggestions"

    def test_chapter_level(self, phm_discourse_map):
        item = AnalysisItem.from_observation(
            phm_discourse_map,
            book="PHM",
            chapter=1,
            anchor="PHM 1",
            anchor_level="chapter",
        )
        assert item.type == "discourse_map"
        assert item.anchor_level == "chapter"

    def test_observation_contains_all_fields(self, basic_suggestion):
        item = AnalysisItem.from_observation(
            basic_suggestion, book="PHM", chapter=1, anchor="PHM 1:1", anchor_level="verse"
        )
        obs = item.observation
        assert obs["strengths"] == ["Preserves indirect appeal force"]
        assert obs["score"] == 7
        assert obs["confidence"] == 85

    def test_none_anchor_allowed(self, run_metadata):
        item = AnalysisItem.from_observation(
            run_metadata, book="PHM", chapter=1, anchor=None, anchor_level="chapter"
        )
        assert item.anchor is None

    def test_invalid_anchor_level(self, basic_suggestion):
        with pytest.raises(ValidationError):
            AnalysisItem.from_observation(
                basic_suggestion,
                book="PHM",
                chapter=1,
                anchor="PHM 1:1",
                anchor_level="invalid_level",
            )


# ---------------------------------------------------------------------------
# CriticReview
# ---------------------------------------------------------------------------


class TestCriticReview:
    def test_accepted(self):
        r = CriticReview(accepted=True, reasoning="Sound analysis.")
        assert r.accepted is True

    def test_rejected(self):
        r = CriticReview(accepted=False, reasoning="Please revise X.")
        assert r.accepted is False

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            CriticReview(accepted=True, reasoning="ok", extra="bad")

    def test_roundtrip_json(self):
        r = CriticReview(accepted=True, reasoning="Good.")
        restored = CriticReview.model_validate_json(r.model_dump_json())
        assert restored.accepted == r.accepted
        assert restored.reasoning == r.reasoning
