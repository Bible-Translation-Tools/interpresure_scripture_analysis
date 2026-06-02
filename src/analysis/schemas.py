"""Pydantic models for all observation types, analysis data structures, and the
analysis item envelope. These mirror the JSON schemas in /schemas/ and are the
single source of truth for structured LLM output and serialization to run directories."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared / internal structures
# ---------------------------------------------------------------------------


class CriticReview(BaseModel):
    """Structured output schema for the critic's accept/reject decision."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(
        ...,
        description="True if the analysis is linguistically sound; False if revision is needed.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Explanation for the decision. If accepted=False, must contain "
            "specific revision instructions."
        ),
    )


class ResourcesUsed(BaseModel):
    """Flags indicating which enrichment resources were active during a run."""

    model_config = ConfigDict(extra="forbid")

    interpresure: bool = False
    bart: bool = False
    macula: bool = False
    discourse_boundary_markers: bool = False


class TranslationInfo(BaseModel):
    """Metadata identifying the translation being analyzed."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="BCP 47 language tag, e.g. 'en', 'es-419'.")
    title: str = Field(..., description="Human-readable title, e.g. 'ULT', 'UST'.")


# ---------------------------------------------------------------------------
# Discourse map
# ---------------------------------------------------------------------------


class DiscourseUnit(BaseModel):
    """A single discourse unit within a chapter."""

    model_config = ConfigDict(extra="forbid")

    verse_start: int = Field(..., ge=1)
    verse_end: int = Field(..., ge=1)
    description: str = Field(
        ...,
        description="What this unit is doing — its function in the chapter's argument or narrative.",
    )


class DiscourseMapObservation(BaseModel):
    """Output of the global discourse pass. Anchored at chapter level."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["discourse_map"] = "discourse_map"
    version: Literal["1.0"] = "1.0"
    dominant_quds: list[str] = Field(
        default_factory=list,
        description="Dominant Questions Under Discussion active across this chapter.",
    )
    argument_structure: str = Field(
        ...,
        description="Prose description of the argumentative, narrative, or poetic arc.",
    )
    discourse_boundaries: list[DiscourseUnit] = Field(
        default_factory=list,
        description="Discourse units with verse ranges and functional descriptions.",
    )
    relational_dynamics: str = Field(
        default="",
        description="Social and relational context operative across the chapter.",
    )
    active_scales: list[str] = Field(
        default_factory=list,
        description="Scalar dimensions active in this chapter.",
    )
    key_presuppositions: list[str] = Field(
        default_factory=list,
        description="Background assumptions the chapter takes for granted.",
    )
    genre_notes: str = Field(
        ...,
        description="Genre and register observations affecting interpretation.",
    )

    def boundary_start_verses(self) -> set[int]:
        """Return the set of verse numbers that open a new discourse unit."""
        return {unit.verse_start for unit in self.discourse_boundaries}

    def boundary_for_verse(self, verse: int) -> DiscourseUnit | None:
        """Return the discourse unit that starts at the given verse, if any."""
        for unit in self.discourse_boundaries:
            if unit.verse_start == verse:
                return unit
        return None


# ---------------------------------------------------------------------------
# InterpreSure suggestions (v2.0)
# ---------------------------------------------------------------------------


class InterpreSureSuggestionsObservation(BaseModel):
    """Pragmatic analysis of a translation passage. v2.0 extends v1.0 with richer
    analytical fields; all additions are optional so consumers use what they need."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["interpresure_suggestions"] = "interpresure_suggestions"
    version: Literal["2.0"] = "2.0"
    strengths: list[str] = Field(
        default_factory=list,
        description="What the translation does well pragmatically.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Areas where pragmatic meaning is lost or distorted.",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete actionable improvements in plain language.",
    )
    score: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Overall pragmatic fidelity score (1–10).",
    )
    confidence: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Model confidence in this analysis (0–100).",
    )
    reasoning: str | None = Field(
        default=None,
        description="Markdown-formatted technical reasoning.",
    )
    cross_references: list[str] = Field(
        default_factory=list,
        description="U23003 references consulted during analysis.",
    )
    verses_to_review: list[int] = Field(
        default_factory=list,
        description="For chapter-level items: verse numbers needing most attention.",
    )


# ---------------------------------------------------------------------------
# Translator questions (v1.0)
# ---------------------------------------------------------------------------


class TranslatorQuestion(BaseModel):
    """A single diagnostic question for a translator."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="Plain-language question answerable from the translation alone.",
    )
    rationale: str | None = Field(
        default=None,
        description="Why this matters — the pragmatic feature at stake.",
    )
    annotation_topics: list[str] = Field(
        default_factory=list,
        description="InterpreSure annotation topics this question draws from.",
    )


class TranslatorQuestionsObservation(BaseModel):
    """Set of diagnostic questions for a translator to evaluate their rendering."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["translator_questions"] = "translator_questions"
    version: Literal["1.0"] = "1.0"
    questions: list[TranslatorQuestion] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Analysis run metadata (v1.0)
# ---------------------------------------------------------------------------


class AnalysisRunMetadataObservation(BaseModel):
    """Provenance and configuration for a single analysis run.
    Anchored at chapter level — one per scope file."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["analysis_run_metadata"] = "analysis_run_metadata"
    version: Literal["1.0"] = "1.0"
    model: str = Field(..., description="Primary analyst model.")
    critic_model: str | None = Field(default=None, description="Critic model, if used.")
    analysis_mode: Literal["few_shot", "zero_shot"] = Field(
        ..., description="Whether expert materials were provided."
    )
    analysis_type: str = Field(
        ..., description="Primary observation type produced by this run."
    )
    timestamp: str = Field(..., description="ISO 8601 run timestamp.")
    translation: TranslationInfo
    biblical_language: Literal["grc", "heb"]
    resources: ResourcesUsed = Field(default_factory=ResourcesUsed)


# ---------------------------------------------------------------------------
# Analysis item envelope (mirrors scripture-analysis-api AnalysisItem)
# ---------------------------------------------------------------------------

AnchorLevel = Literal["repo", "book", "chapter", "verse", "word", "character", "non_verse"]


class AnalysisItem(BaseModel):
    """The envelope that wraps every observation submitted to the scripture-analysis-api."""

    model_config = ConfigDict(extra="forbid")

    book: str | None = None
    chapter: int | None = None
    anchor: str | None = Field(
        default=None,
        description="U23003 scripture reference string.",
    )
    anchor_level: AnchorLevel
    type: str
    version: str
    observation: dict[str, Any]

    @classmethod
    def from_observation(
        cls,
        obs: BaseModel,
        *,
        book: str | None,
        chapter: int | None,
        anchor: str | None,
        anchor_level: AnchorLevel,
    ) -> "AnalysisItem":
        """Convenience constructor: wraps a typed observation model into the envelope."""
        data = obs.model_dump(exclude_none=False)
        return cls(
            book=book,
            chapter=chapter,
            anchor=anchor,
            anchor_level=anchor_level,
            type=data["type"],
            version=data["version"],
            observation=data,
        )
