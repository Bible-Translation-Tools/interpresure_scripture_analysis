"""Tests for analysis/agents.py — registry, response format, and agent factories."""

from __future__ import annotations

import pytest

from analysis.agents import (
    DEFAULT_ANALYSIS_TYPE,
    _response_format_for,
    get_analysis_type_config,
)
from analysis.schemas import (
    CriticReview,
    DiscourseMapObservation,
    InterpreSureSuggestionsObservation,
    TranslatorQuestionsObservation,
)


# ---------------------------------------------------------------------------
# get_analysis_type_config
# ---------------------------------------------------------------------------


class TestGetAnalysisTypeConfig:
    def test_interpresure_suggestions_registered(self):
        cfg = get_analysis_type_config("interpresure_suggestions")
        assert "output_schema" in cfg
        assert "discourse_schema" in cfg
        assert "few_shot_instructions" in cfg
        assert "zero_shot_instructions" in cfg

    def test_translator_questions_registered(self):
        cfg = get_analysis_type_config("translator_questions")
        assert "output_schema" in cfg
        assert cfg["output_schema"] is TranslatorQuestionsObservation

    def test_output_schema_correct_type(self):
        cfg = get_analysis_type_config("interpresure_suggestions")
        assert cfg["output_schema"] is InterpreSureSuggestionsObservation

    def test_discourse_schema_is_discourse_map(self):
        for atype in ["interpresure_suggestions", "translator_questions"]:
            cfg = get_analysis_type_config(atype)
            assert cfg["discourse_schema"] is DiscourseMapObservation

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            get_analysis_type_config("nonexistent_type")

    def test_error_message_lists_available(self):
        try:
            get_analysis_type_config("bad_type")
        except ValueError as exc:
            msg = str(exc)
            assert "interpresure_suggestions" in msg
            assert "translator_questions" in msg

    def test_default_analysis_type_is_valid(self):
        cfg = get_analysis_type_config(DEFAULT_ANALYSIS_TYPE)
        assert cfg is not None


# ---------------------------------------------------------------------------
# _response_format_for
# ---------------------------------------------------------------------------


class TestResponseFormatFor:
    def test_type_is_json_schema(self):
        fmt = _response_format_for(CriticReview)
        assert fmt["type"] == "json_schema"

    def test_has_json_schema_key(self):
        fmt = _response_format_for(CriticReview)
        assert "json_schema" in fmt

    def test_name_matches_class(self):
        fmt = _response_format_for(CriticReview)
        assert fmt["json_schema"]["name"] == "CriticReview"

    def test_strict_is_true(self):
        fmt = _response_format_for(CriticReview)
        assert fmt["json_schema"]["strict"] is True

    def test_schema_is_dict(self):
        fmt = _response_format_for(CriticReview)
        assert isinstance(fmt["json_schema"]["schema"], dict)

    def test_schema_contains_properties(self):
        fmt = _response_format_for(CriticReview)
        schema = fmt["json_schema"]["schema"]
        assert "properties" in schema
        assert "accepted" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_works_for_observation_schema(self):
        fmt = _response_format_for(InterpreSureSuggestionsObservation)
        assert fmt["json_schema"]["name"] == "InterpreSureSuggestionsObservation"
        schema = fmt["json_schema"]["schema"]
        assert "strengths" in schema.get("properties", {})

    def test_works_for_discourse_map(self):
        fmt = _response_format_for(DiscourseMapObservation)
        assert fmt["json_schema"]["name"] == "DiscourseMapObservation"


# ---------------------------------------------------------------------------
# Instructions content — spot checks
# ---------------------------------------------------------------------------


class TestInstructionsContent:
    def test_few_shot_references_expert_materials(self):
        cfg = get_analysis_type_config("interpresure_suggestions")
        instructions = cfg["few_shot_instructions"]
        assert "InterpreSure" in instructions or "expert" in instructions.lower()

    def test_zero_shot_does_not_mandate_annotations(self):
        cfg = get_analysis_type_config("interpresure_suggestions")
        instructions = cfg["zero_shot_instructions"]
        # Zero-shot should not say "MUST engage with annotations"
        assert "MUST engage" not in instructions

    def test_few_shot_has_strict_resource_rule(self):
        cfg = get_analysis_type_config("interpresure_suggestions")
        instructions = cfg["few_shot_instructions"]
        assert "Strict" in instructions or "MUST" in instructions

    def test_translator_questions_few_shot_mentions_questions(self):
        cfg = get_analysis_type_config("translator_questions")
        instructions = cfg["few_shot_instructions"]
        assert "question" in instructions.lower()

    def test_instructions_are_nonempty_strings(self):
        for atype in ["interpresure_suggestions", "translator_questions"]:
            cfg = get_analysis_type_config(atype)
            assert isinstance(cfg["few_shot_instructions"], str)
            assert len(cfg["few_shot_instructions"]) > 100
            assert isinstance(cfg["zero_shot_instructions"], str)
            assert len(cfg["zero_shot_instructions"]) > 100


# ---------------------------------------------------------------------------
# make_agents + make_discourse_agent — structure (with mocked AF)
# ---------------------------------------------------------------------------


class TestMakeAgents:
    @pytest.fixture(autouse=True)
    def mock_af(self, monkeypatch):
        """Mock agent_framework so we don't need real API keys."""
        import sys
        from unittest.mock import MagicMock

        mock_af = MagicMock()
        mock_af.Agent = MagicMock(return_value=MagicMock(name="MockAgent"))
        mock_openai = MagicMock()
        mock_openai.OpenAIChatCompletionClient = MagicMock(return_value=MagicMock())

        monkeypatch.setitem(sys.modules, "agent_framework", mock_af)
        monkeypatch.setitem(sys.modules, "agent_framework.openai", mock_openai)

        return mock_af, mock_openai

    def test_make_agents_returns_tuple(self, mock_af):
        from unittest.mock import MagicMock
        from analysis.agents import make_agents

        af, openai = mock_af
        analyst, critic = make_agents(
            model="gpt-4o",
            critic_model="gpt-4o-mini",
            analysis_type="interpresure_suggestions",
            analysis_mode="few_shot",
            api_key="test-key",
            base_url=None,
            verse_lookup_tool=MagicMock(),
            skills_provider=MagicMock(),
        )
        assert analyst is not None
        assert critic is not None

    def test_make_agents_creates_two_clients(self, mock_af):
        from unittest.mock import MagicMock
        from analysis.agents import make_agents

        af, openai = mock_af
        make_agents(
            model="gpt-4o",
            critic_model="gpt-4o-mini",
            analysis_type="interpresure_suggestions",
            analysis_mode="zero_shot",
            api_key=None,
            base_url=None,
            verse_lookup_tool=MagicMock(),
            skills_provider=MagicMock(),
        )
        # Two OpenAIChatCompletionClient calls: analyst + critic
        assert openai.OpenAIChatCompletionClient.call_count == 2

    def test_make_discourse_agent_returns_agent(self, mock_af):
        from unittest.mock import MagicMock
        from analysis.agents import make_discourse_agent

        af, openai = mock_af
        agent = make_discourse_agent(
            model="gpt-4o",
            analysis_type="interpresure_suggestions",
            api_key="key",
            base_url=None,
            skills_provider=MagicMock(),
        )
        assert agent is not None
