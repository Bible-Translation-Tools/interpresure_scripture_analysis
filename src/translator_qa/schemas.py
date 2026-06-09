"""Pydantic models for the translator QA pipeline.

Two output types:
  - ``GeneratedQuestion`` / ``QuestionSet``: the plain JSON question file
    that bridges the generate → answer steps.
  - ``InterpreSureQAObservation``: the scripture-analysis-api observation
    produced by the answer step (one per Q&A pair).

The ``QuestionGenerationResponse`` is used as a structured LLM output schema
during question generation — it wraps a list of ``GeneratedQuestion`` objects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Question generation schemas
# ---------------------------------------------------------------------------


class GeneratedQuestion(BaseModel):
    """A single question produced by the question generation step."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., description="Sequential 1-based identifier within the question set.")
    anchor: str = Field(
        ...,
        description=(
            "U23003 scripture reference for the verse or verse range this question "
            "pertains to (e.g. 'PHM 1:10', 'ROM 3:10-18')."
        ),
    )
    anchor_level: str = Field(
        default="verse",
        description="Anchor granularity level for the scripture-analysis-api envelope.",
    )
    question: str = Field(
        ...,
        description=(
            "Plain-language question for the translator. No source-language terms "
            "or jargon. Answerable from the translation alone."
        ),
    )
    importance: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "Importance of this question for meaning preservation "
            "(1 = minor nuance, 10 = critical for the main communicative point)."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Brief explanation of what pragmatic feature is at stake and why it matters. "
            "Helps the answering model know what to look for."
        ),
    )
    annotation_topics: list[str] = Field(
        default_factory=list,
        description="InterpreSure annotation topics this question draws from.",
    )


class QuestionGenerationResponse(BaseModel):
    """Structured LLM output for the question generation call."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GeneratedQuestion] = Field(
        ...,
        min_length=1,
        description="All questions generated for this chapter, ordered by anchor.",
    )


class QuestionSet(BaseModel):
    """The full question file written by the generate step.

    This is both the plain JSON output (``questions.json``) and the source
    read by the answer step.
    """

    model_config = ConfigDict(extra="forbid")

    book: str
    chapter: int
    generated_at: str  # ISO 8601 timestamp
    model: str
    resources: list[str]
    biblical_language: str
    questions: list[GeneratedQuestion]


# ---------------------------------------------------------------------------
# QA answer observation schema
# ---------------------------------------------------------------------------


class InterpreSureQAObservation(BaseModel):
    """scripture-analysis-api observation for one Q&A pair (``interpresure_qa`` v1.0)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["interpresure_qa"] = "interpresure_qa"
    version: Literal["1.0"] = "1.0"
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    reasoning: str = Field(
        default="",
        description="Chain-of-thought or explanation before the verdict.",
    )
    result: Literal["pass", "fail", "na"] = Field(
        ...,
        description=(
            "pass = pragmatic goal achieved; "
            "fail = not achieved; "
            "na = requires human judgment."
        ),
    )
    severity: int = Field(
        ...,
        ge=0,
        le=10,
        description=(
            "For fail: issue severity (0 trivial → 10 critical). "
            "For na: importance for human review. "
            "For pass: 0."
        ),
    )
    confidence: int = Field(..., ge=0, le=100)
