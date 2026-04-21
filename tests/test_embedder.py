"""Tests for the embedder (TF-IDF fallback path, no Ollama needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.embedder import tfidf_similarity, cosine_similarity
from core.schemas import ShiftLevel


def test_identical_texts_similarity():
    sim = tfidf_similarity("Hello world this is a test", "Hello world this is a test")
    assert sim >= 0.99

def test_different_texts_lower_similarity():
    sim = tfidf_similarity("You are a Python expert.", "Tell me about climate change policy.")
    assert sim < 0.5

def test_cosine_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == 0.0

def test_cosine_identical():
    a = [0.5, 0.5, 0.5]
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

def test_cosine_empty():
    assert cosine_similarity([], []) == 0.0
