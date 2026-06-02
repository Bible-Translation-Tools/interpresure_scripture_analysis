"""Shared fixtures for analysis test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from analysis.schemas import (
    AnalysisRunMetadataObservation,
    CriticReview,
    DiscourseMapObservation,
    DiscourseUnit,
    InterpreSureSuggestionsObservation,
    ResourcesUsed,
    TranslationInfo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LANG_ROOT = REPO_ROOT / "lang"


# ---------------------------------------------------------------------------
# Common observation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phm_discourse_map():
    return DiscourseMapObservation(
        argument_structure="Paul appeals to Philemon to receive Onesimus as a brother.",
        genre_notes="Personal letter; persuasive epistle.",
        dominant_quds=["Will Philemon reconcile with Onesimus?"],
        discourse_boundaries=[
            DiscourseUnit(verse_start=1, verse_end=7, description="Greeting and thanksgiving"),
            DiscourseUnit(verse_start=8, verse_end=16, description="Central appeal for Onesimus"),
            DiscourseUnit(verse_start=17, verse_end=25, description="Closing requests and greetings"),
        ],
        relational_dynamics="Paul as apostolic authority appealing as friend and father.",
        active_scales=["authority/appeal", "debt/gift", "slave/brother"],
        key_presuppositions=[
            "Philemon owes Paul a spiritual debt",
            "Onesimus has become useful to Paul",
        ],
    )


@pytest.fixture
def basic_suggestion():
    return InterpreSureSuggestionsObservation(
        strengths=["Preserves indirect appeal force"],
        weaknesses=["Loses scalar implicature of 'even'"],
        suggestions=["Consider using 'even more' to preserve the scale"],
        score=7,
        confidence=85,
        reasoning="The word *kai* carries scalar emphasis here.",
        cross_references=["PHM 1:10"],
        verses_to_review=[],
    )


@pytest.fixture
def run_metadata():
    return AnalysisRunMetadataObservation(
        model="gpt-4o",
        critic_model="gpt-4o-mini",
        analysis_mode="few_shot",
        analysis_type="interpresure_suggestions",
        timestamp="2026-06-01T12:00:00+00:00",
        translation=TranslationInfo(language="en", title="ULT"),
        biblical_language="grc",
        resources=ResourcesUsed(
            interpresure=True, bart=True, macula=True, discourse_boundary_markers=True
        ),
    )


@pytest.fixture
def mock_verse_records():
    """Minimal verse records for PHM 1:1-3 without real USFM loading."""
    return [
        {
            "book": "PHM",
            "chapter": 1,
            "verse": 1,
            "reference": "PHM 1:1",
            "biblical_text": "Παῦλος δέσμιος Χριστοῦ Ἰησοῦ",
            "translation_text": "Paul, a prisoner of Christ Jesus,",
        },
        {
            "book": "PHM",
            "chapter": 1,
            "verse": 2,
            "reference": "PHM 1:2",
            "biblical_text": "καὶ Ἀπφίᾳ τῇ ἀδελφῇ",
            "translation_text": "and to Apphia our sister,",
        },
        {
            "book": "PHM",
            "chapter": 1,
            "verse": 3,
            "reference": "PHM 1:3",
            "biblical_text": "χάρις ὑμῖν καὶ εἰρήνη",
            "translation_text": "Grace to you and peace",
        },
    ]


# ---------------------------------------------------------------------------
# Mock agent helpers
# ---------------------------------------------------------------------------


def make_mock_agent(response_text: str):
    """Return a mock Agent whose run() returns a response with .text."""
    agent = MagicMock()
    result = MagicMock()
    result.text = response_text
    agent.run = AsyncMock(return_value=result)
    agent.create_session = MagicMock(return_value=MagicMock())
    return agent


def make_accepted_critic():
    """Return a mock critic agent that always accepts."""
    review = CriticReview(accepted=True, reasoning="Linguistically sound.")
    return make_mock_agent(review.model_dump_json())


def make_rejecting_critic(accept_on_round: int = 1):
    """Return a mock critic that rejects until accept_on_round, then accepts."""
    call_count = {"n": 0}
    agent = MagicMock()

    async def _run(prompt, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > accept_on_round:
            review = CriticReview(accepted=True, reasoning="Now acceptable.")
        else:
            review = CriticReview(accepted=False, reasoning="Revise your analysis.")
        result = MagicMock()
        result.text = review.model_dump_json()
        return result

    agent.run = _run
    return agent
