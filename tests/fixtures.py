"""Explicit scripted responses for testing control flow, without any language model."""

from collections import deque
from copy import deepcopy
import json

from tbi.backends import Response

QUESTION = "Given real numbers x and y with x + y = 10 and x**3 + y**3 = 370, find x**2 + y**2."
VARIANTS = ["Given real numbers x and y with x + y = 6 and x**3 + y**3 = 72, find x**2 + y**2.",
            "Given real numbers x and y with x + y = 8 and x**3 + y**3 = 152, find x**2 + y**2."]


def solution(answer: str, steps: list[str] | None = None, checks: list[dict] | None = None) -> dict:
    return {"steps": steps or ["Step 1: Compute the requested value"],
            "answer": answer, "checks": checks or []}


def cards(step: int) -> dict:
    expressions = ["For s=x+y, p=xy, and c=x^3+y^3, c=s^3-3*p*s.",
                   "For s != 0, solve the identity for p: p=(s^3-c)/(3*s).",
                   "Compute the square sum using u=s^2-2*p."]
    grounded = ["With s=10 and c=370: 370=1000-30*p.",
                "p=(1000-370)/30=21.", "u=100-42=58."]
    checks = [[{"left": "10**3", "right": "1000"}],
              [{"left": "(10**3-370)/(3*10)", "right": "21"}],
              [{"left": "10**2-2*21", "right": "58"}]]
    return {"cards": [{"id": letter, "ale": expressions[step] + f" Perspective {letter}.",
                       "cg": grounded[step], "assumptions": ["x and y are real; s is nonzero"],
                       "inputs": ["s", "c"] if step < 2 else ["s", "p"],
                       "outputs": ["cubic identity", "p", "u"][step:step+1],
                       "checks": deepcopy(checks[step])} for letter in "ABCD"]}


def route(step: int, variants: int = 1) -> dict:
    groundings = [["72=216-18*p", "p=(216-72)/18=8", "u=36-16=20"],
                  ["152=512-24*p", "p=(512-152)/24=15", "u=64-30=34"]]
    numeric = [[("6**3", "216"), ("(6**3-72)/(3*6)", "8"), ("6**2-2*8", "20")],
               [("8**3", "512"), ("(8**3-152)/(3*8)", "15"), ("8**2-2*15", "34")]]
    return {"selected_id": "A", "reason": "The symbolic operation transfers to every variant.",
            "transfer": [{"variant_index": i, "valid": True, "grounding": groundings[i][step],
                          "checks": [{"left": numeric[i][step][0], "right": numeric[i][step][1]}]}
                         for i in range(variants)]}


def paper_script(variants: int = 1) -> dict:
    return {"expert_algebraic": [solution("142", ["Step 1: An intentionally incorrect expert calculation gives 142."])],
            "expert_strategic": [solution("142", ["Step 1: Another intentionally incorrect expert calculation gives 142."])],
            "expert_geometric": [solution("58", ["Step 1: xy=21; x^2+y^2=100-42=58."])],
            "integrate": [solution("58", ["Step 1: Use the sum-of-cubes identity.",
                                          "Step 2: Solve for xy=21.",
                                          "Step 3: Compute x^2+y^2=58."])],
            "mutate": [{"question": question} for question in VARIANTS[:variants]],
            "validate_mutation": [{"valid": True, "same_structure": True,
                                    "reason": "Real roots exist; the target and identities are preserved."}] * variants,
            "cards": [cards(step) for step in range(3)],
            "route": [route(step, variants) for step in range(3)],
            "audit": [{"valid": True, "bad_step": None, "reason": "All dependencies and computations are consistent."}],
            "generate": [solution("58", ["Step 1: xy=(1000-370)/30=21.",
                                          "Step 2: x^2+y^2=100-42=58."],
                                  [{"left": "10**2-2*21", "right": "58"}])],
            "verify": [{"valid": True, "reason": "Independent recomputation confirms the target."}]}


class ReplayBackend:
    """Reject unscripted calls; never guess answers or access benchmark reference labels."""

    parallel_safe = False
    is_test_fixture = True

    def __init__(self, script: dict):
        self.script = {key: deque(deepcopy(values)) for key, values in script.items()}
        self.requests = []

    def generate(self, stage: str, messages: list[dict], **kwargs) -> Response:
        self.requests.append({"stage": stage, "messages": deepcopy(messages), **kwargs})
        if stage not in self.script or not self.script[stage]:
            raise AssertionError(f"No scripted response for {stage}")
        value = self.script[stage].popleft()
        if isinstance(value, Exception):
            raise value
        return Response(value if isinstance(value, str) else json.dumps(value),
                        finish_reason="stop")

    def close(self) -> None:
        pass
