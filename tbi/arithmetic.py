"""Conservative exact arithmetic checks, without eval or generated code execution."""

import ast
from fractions import Fraction
import re


class UnsupportedExpression(ValueError):
    pass


def rational_value(expression: str) -> Fraction:
    if not isinstance(expression, str) or len(expression) > 256:
        raise UnsupportedExpression("Expression is missing or too long")
    try:
        root = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise UnsupportedExpression("Unsupported syntax") from exc
    if len(list(ast.walk(root))) > 64:
        raise UnsupportedExpression("Expression is too complex")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                value = left / right
            elif isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right) > 16:
                    raise UnsupportedExpression("Exponent must be an integer between -16 and 16")
                value = left ** int(right)
            else:
                raise UnsupportedExpression("Unsupported operator")
        else:
            raise UnsupportedExpression("Only rational arithmetic is supported")
        if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
            raise UnsupportedExpression("Result is too large")
        return value

    return visit(root.body)


def check_arithmetic(checks: list[dict]) -> list[dict]:
    reports = []
    for check in checks:
        left, right = check["left"], check["right"]
        try:
            a, b = rational_value(left), rational_value(right)
            reports.append({"left": left, "right": right,
                            "status": "pass" if a == b else "fail",
                            "computed_left": str(a), "computed_right": str(b)})
        except ZeroDivisionError as exc:
            reports.append({"left": left, "right": right, "status": "fail", "reason": str(exc)})
        except UnsupportedExpression as exc:
            reports.append({"left": left, "right": right, "status": "unknown", "reason": str(exc)})
    return reports


def answer_equal(a: str, b: str, mode: str = "exact") -> bool:
    if not a.strip() or not b.strip():
        return False
    if a == b:
        return True
    if mode == "exact":
        return False
    # Restrict numeric comparison to complete scalar answers; never extract a trailing number.
    def scalar(text: str) -> str:
        text = text.strip().strip("$")
        boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            text = boxed.group(1)
        frac = re.fullmatch(r"\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}", text)
        return f"({frac[1]})/({frac[2]})" if frac else text
    try:
        return rational_value(scalar(a)) == rational_value(scalar(b))
    except (UnsupportedExpression, ZeroDivisionError):
        return False


def majority_index(solutions: list[dict], mode: str) -> tuple[int, int]:
    counts = [sum(answer_equal(s["answer"], other["answer"], mode) for other in solutions)
              for s in solutions]
    index = max(range(len(counts)), key=counts.__getitem__)
    return index, counts[index]


def reconcile(framework: dict, experts: list[dict], mode: str) -> tuple[dict, str]:
    if any(answer_equal(framework["answer"], e["answer"], mode) for e in experts):
        return framework, "framework_matches_expert"
    index, count = majority_index(experts, mode)
    if count >= 2:
        return experts[index], "expert_majority"
    return framework, "framework_all_experts_disagree"
