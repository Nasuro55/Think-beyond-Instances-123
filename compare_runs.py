"""Compare paired runs with a problem-level bootstrap and inference cost accounting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def read_run(folder: Path) -> tuple[dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    records = {}
    for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["problem_id"], row["run"])
        if key in records:
            raise ValueError("Duplicate trial in run")
        records[key] = row
    return manifest, records


def compare(reference: Path, candidate: Path, bootstrap: int = 2000, seed: int = 42) -> dict:
    ref_manifest, ref = read_run(reference)
    cand_manifest, cand = read_run(candidate)
    if ref_manifest["data"]["dataset_hash"] != cand_manifest["data"]["dataset_hash"]:
        raise ValueError("Runs use different dataset files")
    if ref_manifest["scorer"] != cand_manifest["scorer"] or set(ref) != set(cand) or not ref:
        raise ValueError("Runs must use the same scorer and complete paired trials")
    grouped = defaultdict(list)
    for key in sorted(ref):
        grouped[key[0]].append(int(cand[key]["correct"]) - int(ref[key]["correct"]))
    differences = [statistics.mean(v) for v in grouped.values()]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(bootstrap))
    interval = [samples[int(0.025 * (bootstrap - 1))], samples[int(0.975 * (bootstrap - 1))]]
    def cost(rows: dict) -> dict:
        return {key: sum(row.get("usage", {}).get(key, 0) for row in rows.values()) / len(rows)
                for key in ["total_requests", "model_calls", "completion_tokens", "cache_hits"]}
    return {"reference": str(reference.resolve()), "candidate": str(candidate.resolve()),
            "reference_accuracy": statistics.mean(int(r["correct"]) for r in ref.values()),
            "candidate_accuracy": statistics.mean(int(r["correct"]) for r in cand.values()),
            "paired_accuracy_gain": statistics.mean(differences), "bootstrap_95_percent_interval": interval,
            "problems": len(grouped), "trials": len(ref),
            "reference_cost_per_trial": cost(ref), "candidate_cost_per_trial": cost(cand),
            "reference_error_trials": sum(r["status"] == "error" for r in ref.values()),
            "candidate_error_trials": sum(r["status"] == "error" for r in cand.values()),
            "note": "Bootstrap resamples problems, retaining all repeated trials within a problem. Costs are reported, not assumed equal. A positive gain requires an actual completed experiment."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    result = compare(args.reference, args.candidate, args.bootstrap)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
