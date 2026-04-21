"""
LLM judge — sends both prompt versions + sampled outputs to a local model
and extracts a structured verdict using Pydantic.
"""

from __future__ import annotations

import json
import re
from typing import Optional
from omegaconf import DictConfig

from core.schemas import JudgeVerdict, OutputSemanticResult
from core.ollama_client import OllamaClient
from core.config import get_config

JUDGE_SYSTEM = """You are an expert prompt engineer evaluating two versions of an LLM prompt.
Your job is to analyse what semantically changed between them and what effect it had on outputs.
Respond ONLY with valid JSON — no preamble, no explanation outside the JSON."""

JUDGE_TEMPLATE = """
## Prompt v1
{prompt_v1}

## Prompt v2
{prompt_v2}

## Sample output from v1 (model: {model})
{output_v1}

## Sample output from v2 (model: {model})
{output_v2}

Analyse the above and return JSON with exactly these keys:
{{
  "intent_change": "<one sentence: what changed in the core intent of the prompt>",
  "gained": "<what v2 gains over v1>",
  "lost": "<what v2 loses compared to v1>",
  "recommendation": "<one of: use_v1 | use_v2 | context_dependent>"
}}
"""


class Judge:
    def __init__(self, client: OllamaClient, cfg: Optional[DictConfig] = None):
        self.client = client
        self.cfg = cfg or get_config()
        self.judge_model = self.cfg.ollama.judge_model

    def _extract_json(self, raw: str) -> dict:
        """Extract JSON from judge response, handling markdown fences."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Find first { ... }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in judge response:\n{raw}")

    def evaluate(
        self,
        prompt_v1: str,
        prompt_v2: str,
        output_results: list[OutputSemanticResult],
    ) -> JudgeVerdict:
        """
        Run the judge against the best-shifting model's outputs.
        Falls back gracefully if the judge model is unavailable.
        """
        # Pick the model with the highest output shift for the judge input
        if output_results:
            ref = max(output_results, key=lambda r: r.score.shift_score)
        else:
            return self._fallback_verdict()

        prompt = JUDGE_TEMPLATE.format(
            prompt_v1=prompt_v1,
            prompt_v2=prompt_v2,
            model=ref.model_name,
            output_v1=ref.output_v1[:600],
            output_v2=ref.output_v2[:600],
        )

        # Check model availability
        if not self.client.is_model_available(self.judge_model):
            # Try first available model
            available = self.client.list_local_models()
            if available:
                self.judge_model = available[0]
            else:
                return self._fallback_verdict()

        raw, _, _ = self.client.generate(
            model=self.judge_model,
            prompt=prompt,
            system=JUDGE_SYSTEM,
        )

        try:
            data = self._extract_json(raw)
            return JudgeVerdict(
                intent_change=data.get("intent_change", "Unable to determine."),
                gained=data.get("gained", "N/A"),
                lost=data.get("lost", "N/A"),
                recommendation=data.get("recommendation", "context_dependent"),
                raw_verdict=raw,
                judge_model=self.judge_model,
            )
        except Exception as e:
            return JudgeVerdict(
                intent_change="Could not parse judge response.",
                gained="N/A",
                lost="N/A",
                recommendation="context_dependent",
                raw_verdict=raw,
                judge_model=self.judge_model,
            )

    def _fallback_verdict(self) -> JudgeVerdict:
        return JudgeVerdict(
            intent_change="Judge unavailable — no local models found.",
            gained="N/A",
            lost="N/A",
            recommendation="context_dependent",
            raw_verdict="",
            judge_model="none",
        )
