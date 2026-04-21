"""
Embedding wrapper using nomic-embed-text via local Ollama.
Falls back to TF-IDF cosine if embedding model is unavailable.
"""

from __future__ import annotations

import math
import re
from typing import Optional
from omegaconf import DictConfig

from core.schemas import EmbeddingResult, SemanticScore, ShiftLevel
from core.config import get_config


# ---------------------------------------------------------------------------
# Cosine similarity (no scipy dependency)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# TF-IDF fallback (used when nomic-embed-text is not installed)
# ---------------------------------------------------------------------------

def _tfidf_vector(text: str, vocab: list[str]) -> list[float]:
    tokens = re.findall(r'\w+', text.lower())
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    return [tf.get(w, 0) / total for w in vocab]


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Fallback similarity using simple TF-IDF cosine."""
    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))
    vocab = sorted(words_a | words_b)
    if not vocab:
        return 1.0
    va = _tfidf_vector(text_a, vocab)
    vb = _tfidf_vector(text_b, vocab)
    return cosine_similarity(va, vb)


# ---------------------------------------------------------------------------
# Main embedder class
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, client, cfg: Optional[DictConfig] = None):
        self.client = client
        self.cfg = cfg or get_config()
        self._embed_model = self.cfg.ollama.embedding_model
        self._use_fallback = False

    def _check_embedding_model(self) -> bool:
        """Check if the embedding model is available locally."""
        available = self.client.list_local_models()
        base = self._embed_model.split(":")[0]
        return any(base in m for m in available)

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text. Falls back to TF-IDF if model unavailable."""
        if not self._use_fallback:
            if not self._check_embedding_model():
                self._use_fallback = True
            else:
                try:
                    vec = self.client.embed(text, model=self._embed_model)
                    return EmbeddingResult(text=text, embedding=vec, model=self._embed_model)
                except Exception:
                    self._use_fallback = True

        # Fallback: return a simple bag-of-words vector
        tokens = re.findall(r'\w+', text.lower())
        vocab = sorted(set(tokens))
        vec = _tfidf_vector(text, vocab)
        return EmbeddingResult(text=text, embedding=vec, model="tfidf-fallback")

    def compute_semantic_score(self, text_a: str, text_b: str) -> SemanticScore:
        """
        Compute semantic similarity between two texts.
        Returns a SemanticScore with similarity, shift_score, and level.
        """
        if not self._use_fallback and self._check_embedding_model():
            try:
                emb_a = self.embed_text(text_a)
                emb_b = self.embed_text(text_b)

                # If vocab mismatch (fallback), align vectors
                if len(emb_a.embedding) != len(emb_b.embedding):
                    sim = tfidf_similarity(text_a, text_b)
                else:
                    sim = cosine_similarity(emb_a.embedding, emb_b.embedding)
            except Exception:
                sim = tfidf_similarity(text_a, text_b)
        else:
            sim = tfidf_similarity(text_a, text_b)

        sim = max(0.0, min(1.0, sim))
        shift = round(1.0 - sim, 4)
        sim = round(sim, 4)

        # Classify level
        t = self.cfg.scoring.thresholds
        if sim >= t.trivial:
            level, label = ShiftLevel.TRIVIAL, "Wording only — meaning unchanged"
        elif sim >= t.minor:
            level, label = ShiftLevel.MINOR, "Light rewording — minor intent shift"
        elif sim >= t.moderate:
            level, label = ShiftLevel.MODERATE, "Notable intent shift — review carefully"
        elif sim >= t.major:
            level, label = ShiftLevel.MAJOR, "Significant restructure — different approach"
        else:
            level, label = ShiftLevel.FUNDAMENTAL, "Fundamental rewrite — entirely different prompt"

        return SemanticScore(similarity=sim, shift_score=shift, level=level, label=label)
