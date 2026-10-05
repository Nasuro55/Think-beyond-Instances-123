"""Run a clearly labeled offline replay of the paper's illustrative example."""

import argparse
import json
from pathlib import Path

from tbi.config import Config
from tbi.pipeline import ThinkBeyondInstances
from tests.fixtures import QUESTION, ReplayBackend, paper_script

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/offline_demo.json")
    args = parser.parse_args()
    config = Config.load(ROOT / "configs" / ("enhanced.json" if args.enhanced else "paper.json"))
    backend = ReplayBackend(paper_script(config.pipeline.mutation_count))
    result = ThinkBeyondInstances(backend, config).solve(QUESTION)
    result["demo_only"] = True
    result["warning"] = "Scripted offline control-flow replay. No model was called. This is not evidence of benchmark accuracy."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["warning"])
    print(f"Replay answer: {result['answer']}; selected cards: {len(result['chain'])}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
