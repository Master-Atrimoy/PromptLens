"""
Orchestration pipeline — wires together all core modules.
Call run_diff() from CLI or Streamlit.
"""

from __future__ import annotations

from typing import Optional
from omegaconf import DictConfig

from core.config import get_config
from core.ollama_client import OllamaClient
from core.differ import build_diff
from core.embedder import Embedder
from core.runner import InferenceRunner
from core.judge import Judge
from core.schemas import DiffReport, PromptSemanticResult


def run_diff(
    prompt_v1: str,
    prompt_v2: str,
    models: list[str],
    run_judge: bool = True,
    cfg: Optional[DictConfig] = None,
) -> DiffReport:
    """
    Full pipeline:
      1. Structural diff + anatomy tagging
      2. Prompt-level semantic score
      3. Parallel inference on all models
      4. Output-level semantic scores
      5. LLM judge verdict (optional)
    """
    cfg = cfg or get_config()

    # Initialise shared services
    client = OllamaClient(cfg)
    embedder = Embedder(client, cfg)
    runner = InferenceRunner(client, embedder, cfg)
    judge = Judge(client, cfg)

    # 1. Structural diff
    structural_diff = build_diff(prompt_v1, prompt_v2)

    # 2. Prompt semantic score
    prompt_score = embedder.compute_semantic_score(prompt_v1, prompt_v2)
    prompt_semantic = PromptSemanticResult(
        prompt_score=prompt_score,
        model_used_for_embedding=embedder._embed_model
        if not embedder._use_fallback
        else "tfidf-fallback",
    )

    # 3 + 4. Inference + output semantic scores
    output_results = runner.run(prompt_v1, prompt_v2, models)

    # 5. Judge
    verdict = None
    if run_judge and output_results:
        verdict = judge.evaluate(prompt_v1, prompt_v2, output_results)

    return DiffReport(
        prompt_v1=prompt_v1,
        prompt_v2=prompt_v2,
        structural_diff=structural_diff,
        prompt_semantic=prompt_semantic,
        output_results=output_results,
        verdict=verdict,
        embedding_model=prompt_semantic.model_used_for_embedding,
        selected_models=models,
    )
