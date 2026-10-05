"""API and optional local-model backends with reproducible request caching."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request

from .config import ModelConfig


class BackendError(RuntimeError):
    pass


@dataclass
class Response:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False


class APIBackend:
    parallel_safe = True

    def __init__(self, config: ModelConfig, cache_path: Path):
        self.config = config
        self.lock = threading.Lock()
        self.database = None
        if config.cache_responses:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.database = sqlite3.connect(cache_path, check_same_thread=False)
            self.database.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
            self.database.commit()

    def generate(self, stage: str, messages: list[dict], *, temperature: float,
                 top_p: float, max_tokens: int, seed: int, top_k: int | None = None) -> Response:
        cfg = self.config
        request = {"model": cfg.model_id, "messages": messages, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens, "stream": False}
        if cfg.send_seed:
            request["seed"] = seed
        if cfg.json_mode:
            request["response_format"] = {"type": "json_object"}
        if top_k is not None:
            request["top_k"] = top_k
        request.update(cfg.extra_body)
        # Include the seed even when a provider does not accept it, separating independent trials.
        namespace = {"base_url": cfg.base_url, "stage": stage, "seed": seed, "request": request}
        key = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
        if self.database is not None:
            with self.lock:
                row = self.database.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
            if row:
                result = json.loads(row[0])
                result["cached"] = True
                result["latency_seconds"] = 0.0
                return Response(**result)

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(cfg.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        endpoint = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        for attempt in range(cfg.http_retries):
            try:
                req = urllib.request.Request(endpoint, payload, headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as handle:
                    obj = json.load(handle)
                choice = obj["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise BackendError("No final response content. Increase max_tokens or configure the reasoning parser.")
                usage = obj.get("usage") or {}
                response = Response(content, int(usage.get("prompt_tokens", 0) or 0),
                                    int(usage.get("completion_tokens", 0) or 0),
                                    time.perf_counter() - started,
                                    choice.get("finish_reason") or "stop")
                if self.database is not None:
                    with self.lock:
                        self.database.execute("INSERT OR REPLACE INTO responses VALUES (?,?)",
                                              (key, json.dumps(asdict(response))))
                        self.database.commit()
                return response
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == cfg.http_retries:
                    raise BackendError(f"HTTP {exc.code} at {endpoint}. Check model_id, credentials, and supported request fields.") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 == cfg.http_retries:
                    raise BackendError(f"Cannot reach {endpoint}: {type(exc).__name__}") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise BackendError("Invalid API response structure") from exc
            time.sleep(min(2 ** attempt, 8))
        raise BackendError("API retry limit exceeded")

    def close(self) -> None:
        if self.database is not None:
            self.database.close()


class TransformersBackend:
    parallel_safe = False

    def __init__(self, config: ModelConfig, project_root: Path):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise BackendError("Install requirements-local.txt in a separate Python environment") from exc
        if "qwen3.5" in config.model_id.lower():
            raise BackendError("Use the API backend for Qwen3.5. This local adapter supports text-only causal models such as Qwen3-4B.")
        self.torch, self.config = torch, config
        cache_dir = Path(config.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = project_root / cache_dir
        args = {"device_map": config.device_map, "trust_remote_code": False,
                "cache_dir": str(cache_dir), "torch_dtype": config.dtype}
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise BackendError("The 4-bit local preset requires a CUDA-enabled PyTorch installation")
            args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=str(cache_dir),
                                                      trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **args).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, stage: str, messages: list[dict], *, temperature: float,
                 top_p: float, max_tokens: int, seed: int, top_k: int | None = None) -> Response:
        torch, cfg = self.torch, self.config
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=cfg.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = inputs["input_ids"].shape[1]
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if input_length > cfg.input_token_limit or (context_limit and input_length + max_tokens > context_limit):
            raise BackendError("Context budget exceeded. Use a larger-context API model or reduce generation limits; prompts are never silently truncated.")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        args = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True}
        if temperature > 0:
            args.update(temperature=temperature, top_p=top_p)
            if top_k is not None:
                args["top_k"] = top_k
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **args)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise BackendError("GPU memory exhausted. Reduce max_tokens/input_token_limit or use an API backend.") from exc
        generated = output[0, input_length:]
        result = Response(self.tokenizer.decode(generated, skip_special_tokens=True), input_length,
                          len(generated), time.perf_counter() - started,
                          "length" if len(generated) >= max_tokens else "stop")
        del inputs, output, generated
        return result

    def close(self) -> None:
        pass


def create_backend(config: ModelConfig, root: Path):
    if config.backend == "api":
        return APIBackend(config, root / "cache" / "responses.sqlite3")
    return TransformersBackend(config, root)
