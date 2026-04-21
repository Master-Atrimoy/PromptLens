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

# Keep system prompt tight — models that don't follow instructions well
# get confused by long system prompts asking for JSON
JUDGE_SYSTEM = (
    "You are an expert prompt engineer. "
    "You MUST respond with ONLY a JSON object. "
    "No markdown. No explanation. No preamble. Raw JSON only."
)

# Compact template — shorter context = faster response = less timeout risk
JUDGE_TEMPLATE = """\
Compare these two prompt versions and their outputs.

PROMPT V1:
{prompt_v1}

PROMPT V2:
{prompt_v2}

OUTPUT FROM V1 (model: {model}):
{output_v1}

OUTPUT FROM V2 (model: {model}):
{output_v2}

Return ONLY this JSON (no other text):
{{"intent_change": "one sentence describing what changed in core intent", "gained": "what v2 gains over v1", "lost": "what v2 loses vs v1", "recommendation": "use_v1 or use_v2 or context_dependent"}}"""


class Judge:
    def __init__(self, client: OllamaClient, cfg: Optional[DictConfig] = None):
        self.client = client
        self.cfg = cfg or get_config()

    def _pick_judge_model(self) -> Optional[str]:
        """
        Pick the best available judge model.
        Priority: config judge_model → mistral → llama → gemma → first available.
        Resolves exact installed name via client.
        """
        available = self.client.list_local_models()
        if not available:
            return None

        # Build priority list
        preferred = [
            self.cfg.ollama.judge_model,  # from config
            "mistral", "llama3", "llama3.1", "gemma3", "gemma",
        ]

        for candidate in preferred:
            base = candidate.split(":")[0].lower()
            for model in available:
                if model.split(":")[0].lower() == base:
                    return model  # exact installed name

        # Fallback: just use whatever is first
        return available[0]

    def _extract_json(self, raw: str) -> dict:
        """
        Robustly extract JSON from model response.
        Handles: plain JSON, ```json fences, leading/trailing text.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find the first { ... } block (greedy — gets outermost object)
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract key-value pairs manually
        result = {}
        for key in ("intent_change", "gained", "lost", "recommendation"):
            pattern = rf'"{key}"\s*:\s*"([^"]*)"'
            m = re.search(pattern, cleaned)
            if m:
                result[key] = m.group(1)

        if result:
            return result

        raise ValueError(f"Could not extract JSON from:\n{raw[:300]}")

    def evaluate(
        self,
        prompt_v1: str,
        prompt_v2: str,
        output_results: list[OutputSemanticResult],
    ) -> JudgeVerdict:
        """
        Run the judge. Uses the model with the highest output shift as reference.
        Always falls back gracefully — never raises.
        """
        judge_model = self._pick_judge_model()
        if not judge_model:
            return self._fallback("No local models available for judge.")

        # Use the model whose outputs shifted most — most informative for the judge
        valid_results = [
            r for r in output_results
            if not r.output_v1.startswith("[ERROR") and not r.output_v2.startswith("[ERROR")
        ]

        if not valid_results:
            # All inference errored — judge on prompts alone without output context
            ref_model = judge_model
            out_v1 = "(inference failed — judging prompts only)"
            out_v2 = "(inference failed — judging prompts only)"
        else:
            ref = max(valid_results, key=lambda r: r.score.shift_score)
            ref_model = ref.model_name
            # Trim outputs — long outputs bloat the context and slow the judge
            out_v1 = ref.output_v1[:400].strip()
            out_v2 = ref.output_v2[:400].strip()

        judge_prompt = JUDGE_TEMPLATE.format(
            prompt_v1=prompt_v1[:500],
            prompt_v2=prompt_v2[:500],
            model=ref_model,
            output_v1=out_v1,
            output_v2=out_v2,
        )

        raw, latency, _ = self.client.generate(
            model=judge_model,
            prompt=judge_prompt,
            system=JUDGE_SYSTEM,
        )

        if raw.startswith("[ERROR"):
            return self._fallback(f"Judge model error: {raw}", judge_model)

        try:
            data = self._extract_json(raw)
            return JudgeVerdict(
                intent_change=data.get("intent_change") or "Unable to determine.",
                gained=data.get("gained") or "N/A",
                lost=data.get("lost") or "N/A",
                recommendation=data.get("recommendation") or "context_dependent",
                raw_verdict=raw,
                judge_model=judge_model,
            )
        except Exception as e:
            # JSON parse failed but we have a raw response — surface it
            return JudgeVerdict(
                intent_change=f"Judge responded but JSON parse failed: {e}",
                gained="See raw verdict",
                lost="See raw verdict",
                recommendation="context_dependent",
                raw_verdict=raw[:600],
                judge_model=judge_model,
            )

    def _fallback(self, reason: str, model: str = "none") -> JudgeVerdict:
        return JudgeVerdict(
            intent_change=reason,
            gained="N/A",
            lost="N/A",
            recommendation="context_dependent",
            raw_verdict="",
            judge_model=model,
        )
