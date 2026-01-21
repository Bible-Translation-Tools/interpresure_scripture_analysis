from __future__ import annotations
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

# --- Debate Transcript Components ---

class LinguistTranscriptTurn(BaseModel):
    role: Literal["linguist"]
    agent: str
    argument: str
    proposed_score: int

class ModeratorTranscriptTurn(BaseModel):
    role: Literal["moderator"]
    agent: str = "Moderator"
    intervened: bool
    feedback: Optional[str] = None
    violators: Optional[List[str]] = Field(default_factory=list)

# Union for the transcript list
TranscriptItem = Union[LinguistTranscriptTurn, ModeratorTranscriptTurn]

class ClosingStatement(BaseModel):
    agent: str
    statement: str
    score: int

# --- Analysis Variants ---

class IndividualAnalysis(BaseModel):
    type: Literal["individual"]
    model: str
    score: int = Field(..., ge=1, le=10)
    reasoning: str

class DebateAnalysis(BaseModel):
    type: Literal["debate"]
    score: int = Field(..., description="Final consensus score")
    debate_transcript: List[TranscriptItem]
    closing_statements: List[ClosingStatement]

# Union for the analysis list
AnalysisType = Union[IndividualAnalysis, DebateAnalysis]

# --- Hierarchy ---

class VerseEntry(BaseModel):
    verse: int
    greek: str
    translation: str
    annotation: str
    notes: Optional[str] = None
    analysis: List[AnalysisType]

class BookAnalysis(BaseModel):
    book: str
    chapter: int
    category: str
    analysis: List[VerseEntry]