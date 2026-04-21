"""
Pydantic schemas for Prompt Diff Visualizer.
All data structures used across core modules are defined here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ShiftLevel(str, Enum):
    TRIVIAL = "trivial"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    FUNDAMENTAL = "fundamental"


class AnatomyType(str, Enum):
    ROLE = "role"
    INSTRUCTION = "instruction"
    CONSTRAINT = "constraint"
    EXAMPLE = "example"
    FORMAT = "format"
    CONTEXT = "context"
    UNKNOWN = "unknown"


class DiffTag(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


# ---------------------------------------------------------------------------
# Diff models
# ---------------------------------------------------------------------------

class DiffSegment(BaseModel):
    text: str
    tag: DiffTag
    anatomy: AnatomyType = AnatomyType.UNKNOWN
    model_config = {"use_enum_values": True}


class PromptDiff(BaseModel):
    segments_v1: list[DiffSegment]
    segments_v2: list[DiffSegment]
    anatomy_changes: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedding & scoring models
# ---------------------------------------------------------------------------

class EmbeddingResult(BaseModel):
    text: str
    embedding: list[float]
    model: str


class SemanticScore(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0)
    shift_score: float = Field(ge=0.0, le=1.0)
    level: ShiftLevel
    label: str

    @field_validator("similarity", "shift_score", mode="before")
    @classmethod
    def clamp_float(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class PromptSemanticResult(BaseModel):
    prompt_score: SemanticScore
    model_used_for_embedding: str


class OutputSemanticResult(BaseModel):
    model_name: str
    output_v1: str
    output_v2: str
    score: SemanticScore
    latency_v1_ms: float
    latency_v2_ms: float
    tokens_v1: Optional[int] = None
    tokens_v2: Optional[int] = None


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    intent_change: str
    gained: str
    lost: str
    recommendation: str
    raw_verdict: str
    judge_model: str


# ---------------------------------------------------------------------------
# Full diff report
# ---------------------------------------------------------------------------

class DiffReport(BaseModel):
    prompt_v1: str
    prompt_v2: str
    structural_diff: PromptDiff
    prompt_semantic: PromptSemanticResult
    output_results: list[OutputSemanticResult]
    verdict: Optional[JudgeVerdict] = None
    embedding_model: str
    selected_models: list[str]

    @property
    def best_model_for_v2(self) -> Optional[str]:
        if not self.output_results:
            return None
        return max(self.output_results, key=lambda r: r.score.shift_score).model_name
