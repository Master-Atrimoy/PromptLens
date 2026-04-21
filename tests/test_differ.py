"""Tests for the structural differ and anatomy tagger."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.differ import build_diff, classify_anatomy
from core.schemas import AnatomyType, DiffTag


def test_classify_role():
    assert classify_anatomy("You are a senior backend engineer") == AnatomyType.ROLE

def test_classify_constraint():
    assert classify_anatomy("Do not exceed 5 bullet points") == AnatomyType.CONSTRAINT

def test_classify_format():
    assert classify_anatomy("Respond in bullet points formatted as JSON") == AnatomyType.FORMAT

def test_classify_instruction():
    assert classify_anatomy("Summarize the following article") == AnatomyType.INSTRUCTION

def test_diff_added_segments():
    v1 = "Summarize the article."
    v2 = "Summarize the article. Respond in bullet points. Do not exceed 5 items."
    diff = build_diff(v1, v2)
    tags_v2 = [s.tag for s in diff.segments_v2]
    assert DiffTag.ADDED in tags_v2 or DiffTag.CHANGED in tags_v2

def test_diff_unchanged():
    v1 = "Tell me about Python."
    v2 = "Tell me about Python."
    diff = build_diff(v1, v2)
    for seg in diff.segments_v1:
        assert seg.tag == DiffTag.UNCHANGED
    for seg in diff.segments_v2:
        assert seg.tag == DiffTag.UNCHANGED

def test_anatomy_changes_populated():
    v1 = "Explain machine learning."
    v2 = "You are an ML expert. Explain machine learning. Use simple language. Respond in bullet points."
    diff = build_diff(v1, v2)
    assert len(diff.anatomy_changes) > 0
