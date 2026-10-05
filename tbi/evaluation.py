"""Resumable evaluation with explicit scoring rules and distinct trial metrics."""

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import time

from .config import Config, derived_seed
from .pipeline import InferenceError, ThinkBeyondInstances, solve_baseline
from .scoring import require_scorer, score_answer


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted((root / "tbi").glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def summarize(records: list[dict], repeats: int, scorer: str) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["problem_id"]].append(record)
    correct = sum(bool(record["correct"]) for record in records)
    total = len(records)
    per_run = {}
    for run in range(repeats):
        subset = [r for r in records if r["run"] == run]
        per_run[str(run)] = {"count": len(subset),
                             "accuracy": sum(r["correct"] for r in subset) / len(subset) if subset else None}
    group_accuracy = [sum(r["correct"] for r in group) / len(group) for group in groups.values()]
    error_groups = sum(any(r["status"] == "error" for r in group) for group in groups.values())
    std = None
    if len(group_accuracy) > 1:
        mean = sum(group_accuracy) / len(group_accuracy)
        variance = sum((a - mean)**2 for a in group_accuracy) / (len(group_accuracy) - 1)
        std = math.sqrt(variance / len(group_accuracy))
    usage_keys = ["total_requests", "model_calls", "cache_hits", "prompt_tokens",
                  "completion_tokens", "fresh_completion_tokens", "schema_failures", "backend_errors"]
    metrics = {"scorer": scorer, "problems": len(groups), "trials": total, "repeats": repeats,
               "correct_trials": correct, "accuracy": correct / total if total else None,
               "avg_at_n": sum(group_accuracy) / len(group_accuracy) if group_accuracy else None,
               "per_run": per_run,
               "pass_at_1_mean": correct / total if total else None,
               "any_success_at_n": sum(any(r["correct"] for r in group) for group in groups.values()) / len(groups) if groups else None,
               "problem_mean_standard_error": std,
               "error_trials": sum(r["status"] == "error" for r in records), "problems_with_errors": error_groups,
               "fallback_trials": sum(r["status"] == "fallback" for r in records),
               "decision_counts": dict(Counter(r.get("decision", "error") for r in records)),
               "grading_status_counts": dict(Counter(r.get("grading", {}).get("status", "unknown") for r in records)),
               "total_usage": {key: sum(r.get("usage", {}).get(key, 0) for r in records) for key in usage_keys},
               "mean_latency_seconds": sum(r.get("latency_seconds", 0) for r in records) / total if total else None,
               "note": "avg_at_n is mean per-trial accuracy, not majority voting or pass@N. Errors count as incorrect. Exact matching is a transparent proxy; the paper does not specify an executable official grader."}
    return metrics


def evaluate(backend, config: Config, records: list[dict], out: Path, *, repeats: int,
             scorer: str, method: str, metadata: dict, resume: bool = False) -> dict:
    require_scorer(scorer)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "traces"
    trace_dir.mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    predictions_path = out / "predictions.jsonl"
    manifest = {"config": config.to_dict(), "config_fingerprint": config.fingerprint(),
                "repeats": repeats, "scorer": scorer, "method": method, "data": metadata}
    if scorer == "math_verify":
        manifest["scorer_versions"] = {package: importlib.metadata.version(package)
                                       for package in ["math-verify", "latex2sympy2-extended", "sympy"]}
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Run already exists at {out}. Use --resume or select a new output directory.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume manifest differs: configuration, source code, dataset, scorer, or repeat count changed")
    else:
        if predictions_path.exists():
            raise ValueError("Predictions exist without a manifest; select a new output directory")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    completed, existing = set(), []
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Incomplete prediction line. Remove only the final incomplete line before resuming.") from exc
            key = (record["problem_id"], record["run"])
            if key in completed:
                raise ValueError(f"Duplicate completed trial: {key}")
            completed.add(key)
            existing.append(record)
    solver = ThinkBeyondInstances(backend, config)
    expected = len(records) * repeats
    with predictions_path.open("a", encoding="utf-8") as handle:
        for run in range(repeats):
            for item in records:
                key = (item["id"], run)
                if key in completed:
                    continue
                seed = derived_seed(config.seed, item["id"], item["question"], run)
                start = time.perf_counter()
                try:
                    # Only question and seed cross the boundary into inference.
                    result = (solve_baseline(backend, config, item["question"], seed) if method == "baseline"
                              else solver.solve(item["question"], seed=seed))
                except InferenceError as exc:
                    result = exc.trace
                except (RuntimeError, ValueError) as exc:
                    result = {"answer": "", "status": "error", "decision": "error",
                              "error": str(exc), "seed": seed, "latency_seconds": time.perf_counter() - start}
                trace_key = hashlib.sha256(f"{item['id']}:{run}".encode()).hexdigest()[:24]
                trace_path = trace_dir / f"{trace_key}.json"
                trace_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                # The reference label is consulted only here, after prediction is complete.
                grading = (score_answer(result["answer"], item["answer"], scorer) if result["status"] != "error"
                           else {"correct": False, "status": "inference_error"})
                correct = grading["correct"]
                record = {"problem_id": item["id"], "run": run, "seed": seed,
                          "answer": result["answer"], "reference_answer": item["answer"], "correct": correct,
                          "status": result["status"], "decision": result["decision"],
                          "usage": result.get("usage", {}), "latency_seconds": result["latency_seconds"],
                          "trace": str(trace_path.relative_to(out)), "error": result.get("error")}
                record["grading"] = grading
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                existing.append(record)
                print(f"[{len(existing)}/{expected}] id={item['id']} run={run} status={record['status']} correct={correct}", flush=True)
    metrics = summarize(existing, repeats, scorer)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
