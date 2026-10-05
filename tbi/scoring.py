"""Evaluation-only scoring; this module is never imported by the reasoning pipeline."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .arithmetic import answer_equal


def require_scorer(mode: str) -> None:
    if mode == "math_verify" and importlib.util.find_spec("math_verify") is None:
        raise RuntimeError("Install requirements-eval.txt before selecting --scorer math_verify")


def score_answer(prediction: str, reference: str, mode: str) -> dict:
    if mode in {"exact", "numeric"}:
        correct = answer_equal(prediction, reference, mode)
        return {"correct": correct, "status": "matched" if correct else "unmatched"}
    if mode != "math_verify":
        raise ValueError(f"Unknown scorer: {mode}")
    require_scorer(mode)
    if not prediction.strip():
        return {"correct": False, "status": "empty_prediction"}
    if prediction == reference:
        return {"correct": True, "status": "exact_match"}
    # A separate process supplies a portable hard deadline, including on Windows.
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("scoring_worker.py"))],
                                 input=json.dumps({"prediction": prediction, "reference": reference}),
                                 capture_output=True, text=True, encoding="utf-8", timeout=12,
                                 check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"correct": False, "status": "grader_timeout"}
    if process.returncode:
        return {"correct": False, "status": "grader_error", "reason": process.stderr[-1000:]}
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"correct": False, "status": "grader_error", "reason": "Invalid scorer output"}
