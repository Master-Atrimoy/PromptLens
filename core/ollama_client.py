"""
Ollama client wrapper.
Handles model discovery, inference, and embedding calls.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx
from omegaconf import DictConfig

from core.config import get_config


class OllamaClient:
    """Sync wrapper around the Ollama REST API."""

    def __init__(self, cfg: Optional[DictConfig] = None):
        self.cfg = cfg or get_config()
        self.base_url = self.cfg.ollama.base_url.rstrip("/")
        # Explicit httpx.Timeout — plain int only sets connect timeout, not read.
        # LLM generation can take 60-180s; read timeout must cover that.
        _t = float(self.cfg.ollama.timeout)
        self.timeout = httpx.Timeout(connect=10.0, read=_t, write=15.0, pool=10.0)

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[str]:
        """Return exact model names as reported by Ollama (e.g. 'llama3.1:8b')."""
        try:
            resp = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            )
            resp.raise_for_status()
            data = resp.json()
            # Keep full names including tags — strip :latest only
            return [
                m["name"].replace(":latest", "")
                for m in data.get("models", [])
            ]
        except Exception:
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if model_name (or its base name) exists locally."""
        available = self.list_local_models()
        # Exact match first, then base-name prefix match
        if model_name in available:
            return True
        base = model_name.split(":")[0]
        return any(m.split(":")[0] == base for m in available)

    def resolve_model_name(self, model_name: str) -> str:
        """
        Return the exact model name string Ollama expects.
        If 'mistral' is given but 'mistral:7b-instruct' is what's installed,
        returns the installed name. Falls back to the input unchanged.
        """
        available = self.list_local_models()
        if model_name in available:
            return model_name
        base = model_name.split(":")[0]
        matches = [m for m in available if m.split(":")[0] == base]
        return matches[0] if matches else model_name

    def health_check(self) -> dict:
        """Check Ollama is running and return status + model list."""
        try:
            models = self.list_local_models()
            return {
                "status": "online",
                "base_url": self.base_url,
                "model_count": len(models),
                "models": models,
            }
        except Exception as e:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "error": str(e),
                "models": [],
            }

    # ------------------------------------------------------------------
    # Synchronous inference — single call
    # ------------------------------------------------------------------

    def generate(
        self, model: str, prompt: str, system: str = ""
    ) -> tuple[str, float, Optional[int]]:
        """
        Single blocking inference call.
        Returns (response_text, latency_ms, token_count).
        Errors are returned as text starting with '[ERROR:' — never raised.
        """
        # Resolve to exact installed name to avoid Ollama 404s
        resolved = self.resolve_model_name(model)

        payload: dict = {
            "model": resolved,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self.cfg.inference.temperature),
                "num_predict": int(self.cfg.inference.max_tokens),
            },
        }
        if system:
            payload["system"] = system

        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000
            return data.get("response", "").strip(), latency, data.get("eval_count")
        except httpx.ReadTimeout:
            latency = (time.time() - t0) * 1000
            return (
                f"[ERROR: Read timeout after {self.timeout.read:.0f}s — "
                f"model '{resolved}' is too slow or still loading. "
                f"Try running: ollama run {resolved}]",
                latency,
                None,
            )
        except httpx.ConnectError:
            latency = (time.time() - t0) * 1000
            return "[ERROR: Cannot connect to Ollama. Is it running? Try: ollama serve]", latency, None
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return f"[ERROR: {type(e).__name__}: {e}]", latency, None

    # ------------------------------------------------------------------
    # Parallel inference — threaded, sequential-per-model-pair
    # ------------------------------------------------------------------

    def generate_parallel(
        self, models: list[str], prompt_v1: str, prompt_v2: str
    ) -> dict[str, dict]:
        """
        Run v1+v2 for each model. Strategy:
        - Each model gets its own thread (model-level parallelism)
        - v1 and v2 for the SAME model run sequentially within that thread
          so Ollama doesn't context-switch mid-generation on one GPU slot.
        
        This avoids the two main failure modes:
        1. asyncio.run() crash inside Streamlit's event loop
        2. GPU thrashing when all 6 calls fire simultaneously
        """

        def run_model_pair(model: str) -> tuple[str, dict]:
            r1 = self.generate(model, prompt_v1)
            r2 = self.generate(model, prompt_v2)
            return model, {"v1": r1, "v2": r2}

        output: dict[str, dict] = {}

        # One thread per model — v1/v2 sequential within each thread
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
            futures = {executor.submit(run_model_pair, m): m for m in models}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    name, result = future.result()
                    output[name] = result
                except Exception as e:
                    output[model] = {
                        "v1": (f"[ERROR: {e}]", 0.0, None),
                        "v2": (f"[ERROR: {e}]", 0.0, None),
                    }

        return output

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """Generate an embedding vector. Uses /api/embeddings endpoint."""
        embed_model = model or self.cfg.ollama.embedding_model
        resolved = self.resolve_model_name(embed_model)
        payload = {"model": resolved, "prompt": text}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            raise RuntimeError(
                f"Embedding failed for '{resolved}': {type(e).__name__}: {e}"
            ) from e
