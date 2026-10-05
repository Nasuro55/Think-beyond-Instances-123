"""Fail-closed parsing and stage-specific response validation."""

import json
import re


class SchemaError(ValueError):
    pass


def parse_object(text: str) -> dict:
    if "<think>" in text and "</think>" not in text:
        raise SchemaError("Incomplete thinking block; increase max_tokens")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Expected one complete JSON object: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise SchemaError("Response must be a JSON object")
    return obj


def text_field(obj: dict, key: str, empty: bool = False) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise SchemaError(f"{key} must be {'a string' if empty else 'a nonempty string'}")
    return value


def boolean_field(obj: dict, key: str) -> bool:
    if type(obj.get(key)) is not bool:
        raise SchemaError(f"{key} must be a JSON boolean, not a string")
    return obj[key]


def string_list(obj: dict, key: str, nonempty: bool = False) -> list:
    value = obj.get(key)
    if not isinstance(value, list) or (nonempty and not value):
        raise SchemaError(f"{key} must be a list")
    if any(not isinstance(v, str) or not v.strip() for v in value):
        raise SchemaError(f"{key} contains an invalid string")
    return value


def validate_checks(obj: dict) -> None:
    checks = obj.get("checks", [])
    if not isinstance(checks, list) or len(checks) > 64:
        raise SchemaError("checks must be a list with at most 64 equations")
    for check in checks:
        if not isinstance(check, dict):
            raise SchemaError("Each check must be an object")
        text_field(check, "left")
        text_field(check, "right")
    obj["checks"] = checks


def validate(stage: str, obj: dict, variants: int = 1) -> dict:
    if stage.startswith("expert_") or stage in {"integrate", "generate", "baseline"}:
        text_field(obj, "answer")
        string_list(obj, "steps", nonempty=True)
        validate_checks(obj)
    elif stage == "mutate":
        text_field(obj, "question")
    elif stage == "validate_mutation":
        boolean_field(obj, "valid")
        boolean_field(obj, "same_structure")
        text_field(obj, "reason")
    elif stage == "cards":
        cards = obj.get("cards")
        if not isinstance(cards, list) or len(cards) != 4:
            raise SchemaError("Exactly four cards are required")
        if any(not isinstance(c, dict) for c in cards):
            raise SchemaError("Each card must be an object")
        if [c.get("id") for c in cards] != list("ABCD"):
            raise SchemaError("Card IDs must be A, B, C, D in order")
        for card in cards:
            for key in ["ale", "cg"]:
                text_field(card, key)
            for key in ["assumptions", "inputs", "outputs"]:
                string_list(card, key)
            validate_checks(card)
    elif stage == "route":
        if "selected_id" not in obj or obj["selected_id"] not in [None, "A", "B", "C", "D"]:
            raise SchemaError("selected_id must be A/B/C/D or null")
        text_field(obj, "reason")
        transfer = obj.get("transfer")
        if not isinstance(transfer, list):
            raise SchemaError("transfer must be a list")
        if obj["selected_id"] is not None and len(transfer) != variants:
            raise SchemaError("A selected card requires transfer evidence for every variant")
        seen = set()
        for item in transfer:
            if not isinstance(item, dict):
                raise SchemaError("Transfer evidence must be an object")
            index = item.get("variant_index")
            if type(index) is not int or not 0 <= index < variants or index in seen:
                raise SchemaError("Invalid or duplicate variant_index")
            seen.add(index)
            boolean_field(item, "valid")
            text_field(item, "grounding")
            validate_checks(item)
    elif stage == "audit":
        boolean_field(obj, "valid")
        text_field(obj, "reason")
        if "bad_step" not in obj or (obj["bad_step"] is not None and
                                      (type(obj["bad_step"]) is not int or obj["bad_step"] < 1)):
            raise SchemaError("bad_step must be a one-based step number or null")
    elif stage == "verify":
        boolean_field(obj, "valid")
        text_field(obj, "reason")
    else:
        raise SchemaError(f"Unknown stage: {stage}")
    return obj
