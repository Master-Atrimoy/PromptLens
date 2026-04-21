"""
Inference runner — sends both prompt versions to all selected models in parallel.
"""

from __future__ import annotations

from typing import Optional
from omegaconf import DictConfig

from core.schemas import OutputSemanticResult
from core.ollama_client import OllamaClient
from core.embedder import Embedder
from core.config import get_config


class InferenceRunner:
    def __init__(
        self,
        client: OllamaClient,
        embedder: Embedder,
        cfg: Optional[DictConfig] = None,
    ):
        self.client = client
        self.embedder = embedder
        self.cfg = cfg or get_config()

    def run(
        self,
        prompt_v1: str,
        prompt_v2: str,
        models: list[str],
    ) -> list[OutputSemanticResult]:
        """
        Run both prompts against all models (parallel if cfg.inference.parallel).
        Returns a list of OutputSemanticResult, one per model.
        """
        if self.cfg.inference.parallel and len(models) > 1:
            raw = self.client.generate_parallel(models, prompt_v1, prompt_v2)
        else:
            raw = {}
            for model in models:
                r1 = self.client.generate(model, prompt_v1)
                r2 = self.client.generate(model, prompt_v2)
                raw[model] = {"v1": r1, "v2": r2}

        results: list[OutputSemanticResult] = []
        for model in models:
            data = raw.get(model, {})
            text_v1, lat_v1, tok_v1 = data.get("v1", ("[No output]", 0.0, None))
            text_v2, lat_v2, tok_v2 = data.get("v2", ("[No output]", 0.0, None))

            score = self.embedder.compute_semantic_score(text_v1, text_v2)

            results.append(
                OutputSemanticResult(
                    model_name=model,
                    output_v1=text_v1,
                    output_v2=text_v2,
                    score=score,
                    latency_v1_ms=round(lat_v1, 1),
                    latency_v2_ms=round(lat_v2, 1),
                    tokens_v1=tok_v1,
                    tokens_v2=tok_v2,
                )
            )

        return results
