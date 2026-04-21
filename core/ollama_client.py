"""
Ollama client wrapper.
Handles model discovery, inference, and embedding calls.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
import httpx
from omegaconf import DictConfig

from core.config import get_config


class OllamaClient:
    """Thin async wrapper around the Ollama REST API."""

    def __init__(self, cfg: Optional[DictConfig] = None):
        self.cfg = cfg or get_config()
        self.base_url = self.cfg.ollama.base_url.rstrip("/")
        self.timeout = self.cfg.ollama.timeout

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[str]:
        """Return names of all models currently pulled in local Ollama."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            # Normalise: strip ':latest' suffix for display
            return [m.replace(":latest", "") for m in models]
        except Exception as e:
            return []

    def is_model_available(self, model_name: str) -> bool:
        available = self.list_local_models()
        # Flexible match: "llama3" matches "llama3:latest" etc.
        return any(model_name.split(":")[0] in m for m in available)

    def health_check(self) -> dict:
        """Check if Ollama is running and return status info."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
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
    # Synchronous inference
    # ------------------------------------------------------------------

    def generate(self, model: str, prompt: str, system: str = "") -> tuple[str, float, Optional[int]]:
        """
        Run a single inference call.
        Returns (response_text, latency_ms, token_count).
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.inference.temperature,
                "num_predict": self.cfg.inference.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        t0 = time.time()
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = (time.time() - t0) * 1000
            text = data.get("response", "")
            tokens = data.get("eval_count")
            return text, latency, tokens
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return f"[ERROR: {e}]", latency, None

    # ------------------------------------------------------------------
    # Async parallel inference
    # ------------------------------------------------------------------

    async def _async_generate(self, model: str, prompt: str) -> tuple[str, float, Optional[int]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.inference.temperature,
                "num_predict": self.cfg.inference.max_tokens,
            },
        }
        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = (time.time() - t0) * 1000
        return data.get("response", ""), latency, data.get("eval_count")

    def generate_parallel(
        self, models: list[str], prompt_v1: str, prompt_v2: str
    ) -> dict[str, dict]:
        """
        Run both prompts against all models in parallel using threads.
        Uses ThreadPoolExecutor instead of asyncio.run() to avoid the
        'cannot run nested event loop' error inside Streamlit.
        Returns {model_name: {v1: (text, latency, tokens), v2: ...}}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks = {}
        for model in models:
            tasks[f"{model}__v1"] = (model, prompt_v1)
            tasks[f"{model}__v2"] = (model, prompt_v2)

        raw: dict[str, tuple] = {}

        def _call(key: str, model: str, prompt: str):
            return key, self.generate(model, prompt)

        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as executor:
            futures = {
                executor.submit(_call, key, model, prompt): key
                for key, (model, prompt) in tasks.items()
            }
            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    raw[key] = result
                except Exception as e:
                    key = futures[future]
                    raw[key] = (f"[ERROR: {e}]", 0.0, None)

        output: dict[str, dict] = {}
        for model in models:
            output[model] = {
                "v1": raw.get(f"{model}__v1", ("[ERROR: no result]", 0.0, None)),
                "v2": raw.get(f"{model}__v2", ("[ERROR: no result]", 0.0, None)),
            }
        return output

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """Generate an embedding vector for the given text."""
        embed_model = model or self.cfg.ollama.embedding_model
        payload = {"model": embed_model, "prompt": text}
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Embedding failed for model '{embed_model}': {e}") from e
