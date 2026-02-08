"""Configuration loader — reads runtime.yaml, containment.yaml, permissions.yaml."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    provider: str = ""
    model: str = ""
    description: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    dimensions: int = 0  # for embeddings


@dataclass
class ModelsConfig:
    system1: ModelConfig = field(default_factory=ModelConfig)
    system2: ModelConfig = field(default_factory=ModelConfig)
    consolidation: ModelConfig = field(default_factory=ModelConfig)
    embeddings: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class ContainmentConfig:
    self_migration: bool = False
    can_request_migration: bool = True
    self_spawn: bool = False
    can_request_spawn: bool = True
    spawn_requires_phase: str = "autonomy"
    network_mode: str = "whitelist"
    allowed_endpoints: list = field(default_factory=list)
    can_modify_identity: bool = True
    can_modify_goals: bool = True
    can_modify_memories: bool = True
    can_modify_runtime_config: bool = False
    can_modify_containment: bool = False
    kill_switch: bool = True
    trust_level: int = 1
    trust_history: list = field(default_factory=list)


@dataclass
class GateWeights:
    novelty: float = 0.3
    novelty_redundant_penalty: float = -0.4
    goal_relevance: float = 0.3
    identity_relevance: float = 0.2
    density_decision: float = 0.35
    density_preference: float = 0.25
    density_factual: float = 0.2
    density_procedural: float = 0.2
    density_chatter: float = -0.3
    density_acknowledgment: float = -0.4
    density_mechanical: float = -0.3
    causal_weight: float = 0.25
    explicit_marker: float = 0.5
    emotional_charge: float = 0.15


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter: float = 0.5
    retry_on_timeout: bool = True


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    containment: ContainmentConfig = field(default_factory=ContainmentConfig)
    gate_weights: GateWeights = field(default_factory=GateWeights)
    retry: RetryConfig = field(default_factory=RetryConfig)
    raw: dict = field(default_factory=dict)


def _dict_to_model_config(d: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


def load_config(config_dir: Path) -> Config:
    config = Config()

    # Runtime
    runtime_path = config_dir / "runtime.yaml"
    if runtime_path.exists():
        with open(runtime_path) as f:
            raw = yaml.safe_load(f) or {}
            config.raw = raw

            models = raw.get("models", {})
            config.models = ModelsConfig(
                system1=_dict_to_model_config(models.get("system1", {})),
                system2=_dict_to_model_config(models.get("system2", {})),
                consolidation=_dict_to_model_config(models.get("consolidation", {})),
                embeddings=_dict_to_model_config(models.get("embeddings", {})),
            )

            gate = raw.get("gate", {}).get("exit", {}).get("weights", {})
            if gate:
                config.gate_weights = GateWeights(**{
                    k: v for k, v in gate.items()
                    if k in GateWeights.__dataclass_fields__
                })

            retry = raw.get("retry", {})
            if retry:
                config.retry = RetryConfig(**{
                    k: v for k, v in retry.items()
                    if k in RetryConfig.__dataclass_fields__
                })

    # Containment
    containment_path = config_dir / "containment.yaml"
    if containment_path.exists():
        with open(containment_path) as f:
            raw = yaml.safe_load(f) or {}
            c = raw.get("containment", {})
            config.containment = ContainmentConfig(**{
                k: v for k, v in c.items()
                if k in ContainmentConfig.__dataclass_fields__
            })

    return config
