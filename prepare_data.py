"""Download a benchmark with an explicit adapter and pinned upstream revision."""

import argparse
from pathlib import Path

from tbi.data import download_dataset, registry

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=registry(ROOT), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    spec = registry(ROOT)[args.dataset]
    print(f"Source: {spec['url']}\nNotes: {spec['notes']}")
    path = download_dataset(ROOT, args.dataset, limit=args.limit, output=args.output, revision=args.revision)
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
