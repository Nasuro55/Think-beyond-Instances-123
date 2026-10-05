"""Solve one mathematical problem with a configured real model."""

import argparse
import json
from pathlib import Path

from tbi.backends import create_backend
from tbi.config import Config
from tbi.pipeline import InferenceError, ThinkBeyondInstances

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--question-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/single_problem.json")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = Config.load(args.config)
    question = args.question if args.question is not None else args.question_file.read_text(encoding="utf-8-sig")
    backend = create_backend(config.model, ROOT)
    try:
        try:
            result = ThinkBeyondInstances(backend, config).solve(question, seed=args.seed)
        except InferenceError as exc:
            result = exc.trace
    finally:
        backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Answer: {result['answer']}\nStatus: {result['status']}\nDecision: {result['decision']}")
    print(f"Trace: {args.output.resolve()}")
    if result["status"] == "error":
        print(f"Error: {result['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
