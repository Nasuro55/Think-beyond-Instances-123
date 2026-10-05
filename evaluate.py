"""Evaluate TBI, a single-model baseline, or an explicit component ablation."""

import argparse
import json
from pathlib import Path

from tbi.backends import create_backend
from tbi.config import Config
from tbi.data import load_jsonl, registry
from tbi.evaluation import code_hash, evaluate, file_hash
from tbi.scoring import require_scorer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=registry(ROOT))
    source.add_argument("--data", type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scorer", choices=["exact", "numeric", "math_verify"], default="exact")
    parser.add_argument("--method", choices=["tbi", "baseline"], default="tbi")
    parser.add_argument("--ablation", choices=["full", "expert_voting", "no_mutation", "no_cards",
                                              "no_router_retry", "no_chain_validation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_scorer(args.scorer)
    config = Config.load(args.config)
    if args.ablation:
        config.pipeline.ablation = args.ablation
    config.validate()
    spec = registry(ROOT).get(args.dataset, {})
    data_path = args.data or ROOT / spec["local_path"]
    records = load_jsonl(data_path)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        records = records[:args.limit]
    repeats = args.repeats if args.repeats is not None else spec.get("recommended_repeats", 1)
    if repeats < 1:
        parser.error("--repeats must be positive")
    metadata = {"dataset_name": args.dataset or "custom", "dataset_hash": file_hash(data_path),
                "source_code_hash": code_hash(ROOT), "count": len(records),
                "problem_ids": [r["id"] for r in records], "dataset_spec": spec}
    meta_path = data_path.with_suffix(".meta.json")
    if meta_path.exists():
        metadata["download_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Scorer: {args.scorer}. Trials: {len(records) * repeats}. No labels enter inference.")
    backend = create_backend(config.model, ROOT)
    try:
        metrics = evaluate(backend, config, records, args.output.resolve(), repeats=repeats,
                           scorer=args.scorer, method=args.method, metadata=metadata, resume=args.resume)
    finally:
        backend.close()
    print(json.dumps({key: metrics[key] for key in ["accuracy", "avg_at_n", "error_trials", "fallback_trials", "total_usage"]}, indent=2))


if __name__ == "__main__":
    main()
