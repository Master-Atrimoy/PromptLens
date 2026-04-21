"""
Hydra-based configuration loader.
Provides a typed config object used across all modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    default_models: list = field(default_factory=lambda: ["llama3", "mistral", "gemma"])
    embedding_model: str = "nomic-embed-text"
    judge_model: str = "mistral"


@dataclass
class ScoringThresholds:
    trivial: float = 0.95
    minor: float = 0.85
    moderate: float = 0.65
    major: float = 0.40


@dataclass
class ScoringConfig:
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)


@dataclass
class InferenceConfig:
    temperature: float = 0.7
    max_tokens: int = 1024
    parallel: bool = True


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_config(config_dir: str | None = None) -> DictConfig:
    """Load config from conf/config.yaml using Hydra."""
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "conf")
    config_dir = os.path.abspath(config_dir)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    return cfg


def get_config() -> DictConfig:
    """Cached config loader — safe to call multiple times."""
    try:
        return load_config()
    except Exception:
        # Fallback: return default structured config as DictConfig
        defaults = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 120,
                "default_models": ["llama3", "mistral", "gemma"],
                "embedding_model": "nomic-embed-text",
                "judge_model": "mistral",
            },
            "scoring": {
                "thresholds": {
                    "trivial": 0.95,
                    "minor": 0.85,
                    "moderate": 0.65,
                    "major": 0.40,
                }
            },
            "inference": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "parallel": True,
            },
        }
        return OmegaConf.create(defaults)
