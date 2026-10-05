"""Validated configuration shared by the command-line tools."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json


@dataclass
class ModelConfig:
    backend: str = "api"
    model_id: str = "Qwen/Qwen3.5-9B"
    model_url: str = "https://huggingface.co/Qwen/Qwen3.5-9B"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "MODEL_API_KEY"
    timeout_seconds: int = 300
    http_retries: int = 3
    send_seed: bool = True
    json_mode: bool = False
    extra_body: dict = field(default_factory=dict)
    enable_thinking: bool = True
    load_in_4bit: bool = False
    device_map: str = "auto"
    dtype: str = "auto"
    input_token_limit: int = 24000
    cache_dir: str = "cache/model"
    cache_responses: bool = True


@dataclass
class PipelineConfig:
    temperature: float = 0.6
    mutation_temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    max_tokens: int = 8192
    max_reasoning_steps: int = 32
    candidate_count: int = 4
    mutation_count: int = 1
    mutation_attempts: int = 3
    card_attempts: int = 3
    json_repair_attempts: int = 2
    chain_rollbacks: int = 1
    parallel_experts: bool = True
    answer_mode: str = "exact"
    numeric_checks: bool = False
    verify_consensus: bool = False
    verify_framework: bool = False
    framework_retries: int = 1
    ablation: str = "full"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(obj) - {"model", "pipeline", "seed", "description"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        config = cls(ModelConfig(**obj.get("model", {})),
                     PipelineConfig(**obj.get("pipeline", {})), obj.get("seed", 42))
        config.validate()
        return config

    def validate(self) -> None:
        p, m = self.pipeline, self.model
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for obj, keys in [(m, ["send_seed", "json_mode", "enable_thinking", "load_in_4bit", "cache_responses"]),
                          (p, ["parallel_experts", "numeric_checks", "verify_consensus", "verify_framework"])]:
            for key in keys:
                if type(getattr(obj, key)) is not bool:
                    raise ValueError(f"{key} must be a JSON boolean")
        if not isinstance(m.extra_body, dict):
            raise ValueError("extra_body must be an object")
        if not isinstance(m.model_id, str) or not m.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if m.backend not in {"api", "transformers"}:
            raise ValueError("backend must be api or transformers")
        if p.candidate_count != 4:
            raise ValueError("The paper uses exactly four logic cards: A, B, C, D")
        if p.answer_mode not in {"exact", "numeric"}:
            raise ValueError("answer_mode must be exact or numeric")
        if p.ablation not in {"full", "expert_voting", "no_mutation", "no_cards",
                              "no_router_retry", "no_chain_validation"}:
            raise ValueError("Unknown ablation")
        for key in ["max_tokens", "max_reasoning_steps", "mutation_count",
                    "mutation_attempts", "card_attempts"]:
            if type(getattr(p, key)) is not int or getattr(p, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ["json_repair_attempts", "chain_rollbacks", "framework_retries"]:
            if type(getattr(p, key)) is not int or getattr(p, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if p.chain_rollbacks > 1:
            raise ValueError("At most one chain rollback is supported, as in Section 2.4")
        if not 0 < p.top_p <= 1 or not 0 <= p.temperature <= 2:
            raise ValueError("Invalid decoding parameters")
        if not 0 <= p.mutation_temperature <= 2:
            raise ValueError("Invalid mutation temperature")
        if p.top_k is not None and (type(p.top_k) is not int or p.top_k < 1):
            raise ValueError("top_k must be null or a positive integer")
        if m.http_retries < 1 or m.timeout_seconds < 1 or m.input_token_limit < 1:
            raise ValueError("Invalid model limits")
        if set(m.extra_body) & {"model", "messages", "temperature", "top_p", "seed",
                               "max_tokens", "n", "stream"}:
            raise ValueError("extra_body cannot override core request fields")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def derived_seed(base: int, *parts: object) -> int:
    payload = json.dumps([base, *parts], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)
