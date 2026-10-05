"""Isolated optional Math-Verify worker. No model or inference code is loaded here."""

import json
from fractions import Fraction
import re
import sys


def main() -> None:
    from math_verify import LatexExtractionConfig, parse, verify

    payload = json.load(sys.stdin)
    def collection_kind(answer: str) -> str | None:
        text = answer.strip().strip("$").replace(r"\left", "").replace(r"\right", "")
        if text.startswith(r"\boxed{") and text.endswith("}"):
            text = text[7:-1]
        text = text.replace(r"\lbrace", r"\{").replace(r"\rbrace", r"\}")
        if text.startswith(r"\{") and text.endswith(r"\}"):
            return "set"
        if text.startswith("(") and text.endswith(")") and "," in text:
            return "tuple_or_open_interval"
        if text.startswith("[") and text.endswith("]") and "," in text:
            return "closed_interval_or_list"
        return None

    gold_kind = collection_kind(payload["reference"])
    predicted_kind = collection_kind(payload["prediction"])
    if gold_kind and predicted_kind and (gold_kind == "set") != (predicted_kind == "set"):
        print(json.dumps({"correct": False, "status": "collection_type_mismatch"}))
        return

    def parsed(answer: str):
        text = answer.strip().strip("$")
        if text.startswith(r"\boxed{") and text.endswith("}"):
            text = text[7:-1]
        percent = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*\\?%", text)
        if percent:
            value = Fraction(percent[1]) / 100
            text = rf"\frac{{{value.numerator}}}{{{value.denominator}}}"
        elif "%" in text:
            # Reject unsupported percentage syntax rather than accidentally dropping its unit.
            return []
        if not (text.startswith("$") and text.endswith("$")):
            text = "$" + text + "$"
        return parse(text, extraction_config=[LatexExtractionConfig()],
                     fallback_mode="no_fallback", parsing_timeout=None)
    gold, predicted = parsed(payload["reference"]), parsed(payload["prediction"])
    if not gold or not predicted:
        result = {"correct": False, "status": "grader_parse_failure"}
    else:
        correct = bool(verify(gold, predicted, strict=True, timeout_seconds=None))
        result = {"correct": correct, "status": "matched" if correct else "unmatched"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
