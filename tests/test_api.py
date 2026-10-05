"""Exercise the real HTTP adapter against a local scripted API server."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from tbi.backends import APIBackend, BackendError
from tbi.config import Config, ModelConfig
from tbi.pipeline import ThinkBeyondInstances
from tbi.prompts import INSTRUCTIONS
from .fixtures import QUESTION, paper_script


@contextmanager
def server_fixture(script: dict | None = None, status: int = 200):
    requests = []
    queue = {key: list(values) for key, values in (script or {}).items()}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            system = request["messages"][0]["content"]
            stage = next((name for name, instruction in INSTRUCTIONS.items() if system.startswith(instruction)), None)
            response = queue[stage].pop(0) if stage in queue else {"valid": True, "reason": "Test server response"}
            payload = {"choices": [{"message": {"content": json.dumps(response)}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 12, "completion_tokens": 8}}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class APITests(unittest.TestCase):
    def test_real_http_adapter_cache_and_trial_seed_separation(self):
        with server_fixture() as (url, requests), tempfile.TemporaryDirectory() as folder:
            backend = APIBackend(ModelConfig(base_url=url), Path(folder) / "responses.sqlite3")
            messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
            kwargs = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 100, "seed": 42}
            first = backend.generate("verify", messages, **kwargs)
            second = backend.generate("verify", messages, **kwargs)
            third = backend.generate("verify", messages, **{**kwargs, "seed": 43})
            backend.close()
            self.assertEqual(len(requests), 2)
            self.assertEqual((first.cached, second.cached, third.cached), (False, True, False))
            self.assertEqual(first.completion_tokens, 8)
            self.assertEqual(requests[0]["temperature"], 0.6)
            self.assertEqual(requests[1]["seed"], 43)

    def test_unsupported_seed_can_be_disabled_without_merging_trials(self):
        with server_fixture() as (url, requests), tempfile.TemporaryDirectory() as folder:
            backend = APIBackend(ModelConfig(base_url=url, send_seed=False), Path(folder) / "cache.sqlite3")
            for seed in [42, 43]:
                backend.generate("verify", [{"role": "user", "content": "test"}],
                                 temperature=0.6, top_p=0.95, max_tokens=100, seed=seed)
            backend.close()
            self.assertEqual(len(requests), 2)
            self.assertNotIn("seed", requests[0])

    def test_http_errors_are_not_converted_to_empty_answers(self):
        with server_fixture(status=400) as (url, requests), tempfile.TemporaryDirectory() as folder:
            backend = APIBackend(ModelConfig(base_url=url), Path(folder) / "cache.sqlite3")
            with self.assertRaises(BackendError):
                backend.generate("verify", [{"role": "user", "content": "test"}],
                                 temperature=0.6, top_p=0.95, max_tokens=100, seed=42)
            backend.close()
            self.assertEqual(len(requests), 1)

    def test_end_to_end_pipeline_through_http(self):
        with server_fixture(paper_script()) as (url, requests), tempfile.TemporaryDirectory() as folder:
            config = Config()
            config.model.base_url = url
            config.pipeline.parallel_experts = False
            backend = APIBackend(config.model, Path(folder) / "cache.sqlite3")
            result = ThinkBeyondInstances(backend, config).solve(QUESTION)
            backend.close()
            self.assertEqual((result["answer"], result["status"]), ("58", "ok"))
            self.assertEqual(len(requests), 14)
            self.assertEqual(result["usage"]["completion_tokens"], 14 * 8)
            self.assertTrue(all(r["model"] == config.model.model_id for r in requests))


if __name__ == "__main__":
    unittest.main()
