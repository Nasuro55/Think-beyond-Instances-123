"""Benchmark adapters. Labels are loaded for scoring, never forwarded to the solver."""

from pathlib import Path
import hashlib
import json


def registry(root: Path) -> dict:
    return json.loads((root / "configs" / "datasets.json").read_text(encoding="utf-8"))


def record_from_row(row: dict, spec: dict, row_index: int) -> dict:
    question = row.get(spec["question_field"])
    answer = row.get(spec["answer_field"])
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {row_index}: question is missing or empty")
    if answer is None:
        raise ValueError(f"Row {row_index}: reference answer is missing")
    if isinstance(answer, list):
        if len(answer) == 1:
            answer = answer[0]
        else:
            # Preserve multiple required answers; do not treat them as interchangeable alternatives.
            answer = json.dumps(answer, ensure_ascii=False)
    if not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
        raise ValueError(f"Row {row_index}: unsupported answer representation")
    identity = row.get(spec.get("id_field", "id"))
    if identity is None:
        identity = hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Row {row_index}: metadata must be an object")
    metadata = {**metadata, **{key: row[key] for key in ["subject", "level", "difficulty", "subfield",
                                                       "answer_type", "is_multiple_answer", "unit"] if key in row}}
    return {"id": str(identity), "question": question, "answer": str(answer), "metadata": metadata}


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run prepare_data.py or supply your JSONL file.")
    records, seen = [], set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if "question" not in row and "problem" in row:
                row["question"] = row["problem"]
            item = record_from_row(row, {"question_field": "question", "answer_field": "answer", "id_field": "id"}, line_number)
            if item["id"] in seen:
                raise ValueError(f"Duplicate problem id {item['id']} at line {line_number}")
            seen.add(item["id"])
            records.append(item)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def download_dataset(root: Path, name: str, *, limit: int | None = None,
                     output: Path | None = None, revision: str | None = None) -> Path:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install requirements-data.txt before downloading benchmarks") from exc
    specs = registry(root)
    if name not in specs:
        raise ValueError(f"Unknown dataset: {name}")
    spec = specs[name]
    token = None
    info = HfApi().dataset_info(spec["repo_id"], revision=revision or spec.get("revision"), token=token)
    pinned_revision = info.sha
    dataset = load_dataset(spec["repo_id"], name=spec.get("subset"), split=spec["split"],
                           revision=pinned_revision, streaming=True)
    target = output or root / spec["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}; choose --output with a new filename")
    count = 0
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(dataset):
                if limit is not None and count >= limit:
                    break
                item = record_from_row(row, spec, index)
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("The requested dataset split is empty")
        load_jsonl(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {"dataset": name, "repo_id": spec["repo_id"], "subset": spec.get("subset"),
                "split": spec["split"], "resolved_revision": pinned_revision,
                "count": count, "limit": limit, "notes": spec.get("notes", "")}
    target.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
