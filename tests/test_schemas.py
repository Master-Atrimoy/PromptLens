"""Tests for Pydantic schema validation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from core.schemas import SemanticScore, ShiftLevel


def test_semantic_score_clamps_similarity():
    score = SemanticScore(similarity=1.1, shift_score=0.0, level=ShiftLevel.TRIVIAL, label="test")
    assert score.similarity == 1.0

def test_semantic_score_clamps_negative():
    score = SemanticScore(similarity=-0.1, shift_score=1.0, level=ShiftLevel.FUNDAMENTAL, label="test")
    assert score.similarity == 0.0

def test_shift_level_enum_values():
    assert ShiftLevel.TRIVIAL == "trivial"
    assert ShiftLevel.FUNDAMENTAL == "fundamental"
