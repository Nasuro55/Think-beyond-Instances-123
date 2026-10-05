"""Three-stage pipeline implementing Sections 2.2-2.4 of the supplied paper."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import threading
import time

from .arithmetic import check_arithmetic, majority_index, reconcile
from .backends import BackendError
from .config import Config, derived_seed
from .prompts import EXPERTS, messages
from .schemas import SchemaError, parse_object, validate


class PipelineFailure(RuntimeError):
    pass


class InferenceError(RuntimeError):
    def __init__(self, message: str, trace: dict):
        super().__init__(message)
        self.trace = trace


class JSONAgent:
    def __init__(self, backend, config: Config, seed: int):
        self.backend, self.config, self.seed = backend, config, seed
        self.calls = []
        self.counters = {}
        self.lock = threading.Lock()

    def ask(self, stage: str, payload: dict, variants: int = 1) -> dict:
        p = self.config.pipeline
        chat = messages(stage, payload, p.numeric_checks)
        with self.lock:
            number = self.counters.get(stage, 0)
            self.counters[stage] = number + 1
        for repair in range(p.json_repair_attempts + 1):
            seed = derived_seed(self.seed, stage, number, repair)
            record = {"stage": stage, "stage_call": number, "repair": repair,
                      "seed": seed, "messages": list(chat)}
            try:
                response = self.backend.generate(
                    stage, chat, temperature=p.mutation_temperature if stage == "mutate" else p.temperature,
                    top_p=p.top_p, max_tokens=p.max_tokens, seed=seed, top_k=p.top_k)
                record.update(asdict(response))
            except BackendError as exc:
                record["error"] = str(exc)
                with self.lock:
                    self.calls.append(record)
                raise
            try:
                if response.finish_reason == "length":
                    raise SchemaError("Generation reached the token limit; increase max_tokens")
                obj = validate(stage, parse_object(response.text), variants)
                record["schema_valid"] = True
                with self.lock:
                    self.calls.append(record)
                return obj
            except SchemaError as exc:
                record["schema_valid"] = False
                record["schema_error"] = str(exc)
                with self.lock:
                    self.calls.append(record)
                if repair == p.json_repair_attempts:
                    raise SchemaError(f"{stage}: {exc}") from exc
                chat = [*chat, {"role": "assistant", "content": response.text},
                        {"role": "user", "content":
                         f"Your previous response failed validation: {exc}. Return a complete corrected JSON object using the required schema. Do not copy example values. If the token budget is limited, shorten prose while retaining all necessary derivations."}]
        raise SchemaError("Response repair limit exceeded")

    def stats(self) -> dict:
        fresh = [call for call in self.calls if "text" in call and not call.get("cached", False)]
        fixture = getattr(self.backend, "is_test_fixture", False)
        return {"total_requests": len(self.calls), "model_calls": 0 if fixture else len(fresh),
                "fixture_calls": len(fresh) if fixture else 0,
                "cache_hits": sum(bool(c.get("cached")) for c in self.calls),
                "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in self.calls),
                "completion_tokens": sum(c.get("completion_tokens", 0) for c in self.calls),
                "fresh_completion_tokens": sum(c.get("completion_tokens", 0) for c in fresh),
                "schema_failures": sum(c.get("schema_valid") is False for c in self.calls),
                "backend_errors": sum("error" in c for c in self.calls)}


class ThinkBeyondInstances:
    def __init__(self, backend, config: Config):
        config.validate()
        self.backend, self.config = backend, config

    def solve(self, question: str, *, seed: int | None = None) -> dict:
        # Reference answers are intentionally absent from this interface.
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        p = self.config.pipeline
        trial_seed = self.config.seed if seed is None else seed
        agent = JSONAgent(self.backend, self.config, trial_seed)
        events, chain, variants = [], [], []
        started = time.perf_counter()
        names = list(EXPERTS)
        try:
            if p.parallel_experts and getattr(self.backend, "parallel_safe", False):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(agent.ask, name, {"question": question}) for name in names]
                    experts = [future.result() for future in futures]
            else:
                experts = [agent.ask(name, {"question": question}) for name in names]
        except (BackendError, SchemaError) as exc:
            trace = {"question": question, "answer": "", "status": "error", "decision": "expert_stage_error",
                     "error": str(exc), "seed": trial_seed, "calls": agent.calls, "events": [],
                     "usage": agent.stats(), "latency_seconds": time.perf_counter() - started,
                     "config_fingerprint": self.config.fingerprint()}
            raise InferenceError(str(exc), trace) from exc
        for name, expert in zip(names, experts):
            expert["role"] = name.removeprefix("expert_")

        def finish(solution: dict, decision: str, status: str = "ok") -> dict:
            return {"question": question, "answer": solution["answer"], "solution": solution,
                    "status": status, "decision": decision, "seed": trial_seed,
                    "experts": experts, "variants": variants, "chain": chain,
                    "events": events, "calls": agent.calls, "usage": agent.stats(),
                    "latency_seconds": time.perf_counter() - started,
                    "config_fingerprint": self.config.fingerprint()}

        def arithmetic_ok(obj: dict, location: str) -> bool:
            if not p.numeric_checks:
                return True
            reports = check_arithmetic(obj.get("checks", []))
            events.append({"event": "arithmetic_check", "location": location, "reports": reports})
            return all(report["status"] != "fail" for report in reports)

        def verify_solution(solution: dict, location: str) -> dict:
            reports = check_arithmetic(solution.get("checks", [])) if p.numeric_checks else []
            verdict = agent.ask("verify", {"question": question, "solution": solution,
                                           "arithmetic_reports": reports})
            if any(report["status"] == "fail" for report in reports):
                verdict = {"valid": False, "reason": "Exact arithmetic check failed: " + json.dumps(reports)}
            events.append({"event": "answer_verification", "location": location, **verdict})
            return verdict

        index, count = majority_index(experts, p.answer_mode)
        if p.ablation == "expert_voting":
            return finish(experts[index] if count >= 2 else experts[0], "expert_voting", "ablation")

        try:
            if count == 3:
                clean = all(arithmetic_ok(e, e["role"]) for e in experts)
                approved = clean and (not p.verify_consensus or verify_solution(experts[0], "consensus")["valid"])
                if approved:
                    return finish(experts[0], "unanimous_experts")
                events.append({"event": "consensus_rejected", "reason": "Auxiliary verification rejected unanimous answers"})

            preliminary = agent.ask("integrate", {"question": question, "expert_solutions": experts})
            events.append({"event": "preliminary_solution", "solution": preliminary})
            steps = preliminary["steps"]
            if len(steps) > p.max_reasoning_steps:
                raise PipelineFailure("The preliminary solution exceeds max_reasoning_steps")
            if p.ablation == "no_cards":
                result = agent.ask("generate", {"question": question, "preliminary_solution": preliminary,
                                                "instruction": "Execute the preliminary reasoning without a logic-card chain"})
                selected, decision = reconcile(result, experts, p.answer_mode)
                return finish(selected, "no_cards:" + decision, "ablation")

            if p.ablation == "no_mutation":
                variants.append(question)
                events.append({"event": "mutation_disabled_ablation"})
            else:
                for variant_number in range(p.mutation_count):
                    feedback = ""
                    for attempt in range(p.mutation_attempts):
                        mutation = agent.ask("mutate", {"question": question, "accepted_variants": variants,
                                                       "feedback": feedback})["question"]
                        if mutation.strip() == question.strip() or mutation.strip() in [v.strip() for v in variants]:
                            verdict = {"valid": False, "same_structure": False,
                                       "reason": "Mutation duplicates the original or an accepted variant"}
                        else:
                            verdict = agent.ask("validate_mutation", {"original_question": question,
                                                                       "mutated_question": mutation})
                        events.append({"event": "mutation_validation", "variant_number": variant_number,
                                       "attempt": attempt, "mutation": mutation, **verdict})
                        if verdict["valid"] and verdict["same_structure"]:
                            variants.append(mutation)
                            break
                        feedback = verdict["reason"]
                    else:
                        raise PipelineFailure("No valid logic-preserving mutation within the attempt budget")

            def select_step(step_index: int, feedback: str = "") -> dict:
                attempts = 1 if p.ablation == "no_router_retry" else p.card_attempts
                for attempt in range(attempts):
                    candidates = agent.ask("cards", {"question": question, "step_index": step_index + 1,
                                                    "current_step": steps[step_index],
                                                    "integrated_answer": preliminary["answer"],
                                                    "selected_prefix": chain[:step_index], "feedback": feedback})["cards"]
                    routed = agent.ask("route", {"original_question": question, "mutated_questions": variants,
                                                 "step_index": step_index + 1,
                                                 "selected_prefix": chain[:step_index], "candidate_cards": candidates},
                                       variants=len(variants))
                    events.append({"event": "card_routing", "step_index": step_index + 1,
                                   "attempt": attempt, "candidates": candidates, "router": routed})
                    selected_id = routed["selected_id"]
                    if selected_id is not None:
                        card = next(c for c in candidates if c["id"] == selected_id)
                        original_ok = arithmetic_ok(card, f"card:{step_index + 1}:{selected_id}")
                        transferred_ok = all(item["valid"] and arithmetic_ok(item, f"transfer:{step_index + 1}:{item['variant_index']}")
                                             for item in routed["transfer"])
                        if original_ok and transferred_ok:
                            return {**card, "step_index": step_index + 1, "transfer": routed["transfer"],
                                    "router_reason": routed["reason"]}
                        feedback = "Selected card has invalid transfer evidence or failed arithmetic checks. " + routed["reason"]
                    else:
                        feedback = routed["reason"]
                raise PipelineFailure(f"No valid logic card at step {step_index + 1}: {feedback}")

            for step_index in range(len(steps)):
                chain.append(select_step(step_index))

            if p.ablation != "no_chain_validation":
                for audit_number in range(p.chain_rollbacks + 1):
                    audit = agent.ask("audit", {"question": question, "mutated_questions": variants, "chain": chain})
                    events.append({"event": "chain_audit", "audit_number": audit_number, **audit})
                    if audit["valid"]:
                        break
                    if audit_number == p.chain_rollbacks:
                        raise PipelineFailure("Chain remains invalid after the allowed rollback: " + audit["reason"])
                    bad_step = audit["bad_step"]
                    first = bad_step - 1 if bad_step is not None else 0
                    if first >= len(chain):
                        raise PipelineFailure("Auditor returned an out-of-range bad_step")
                    # Rebuild dependent downstream cards and transfer evidence after changing a premise.
                    del chain[first:]
                    for step_index in range(first, len(steps)):
                        chain.append(select_step(step_index, "Global audit feedback: " + audit["reason"]))
                    events.append({"event": "chain_rollback", "rebuilt_from_step": first + 1})
            else:
                events.append({"event": "chain_validation_disabled_ablation"})

            feedback = ""
            generation_budget = p.framework_retries + 1 if (p.verify_framework or p.numeric_checks) else 1
            for generation_number in range(generation_budget):
                framework = agent.ask("generate", {"question": question, "chain": chain, "feedback": feedback})
                numeric_valid = arithmetic_ok(framework, "framework_execution")
                if not p.verify_framework and numeric_valid:
                    break
                verdict = (verify_solution(framework, "framework") if p.verify_framework
                           else {"valid": numeric_valid, "reason": "Framework execution failed exact arithmetic checks"})
                if verdict["valid"]:
                    break
                feedback = verdict["reason"]
            else:
                raise PipelineFailure("Framework answer failed independent verification: " + feedback)

            selected, decision = reconcile(framework, experts, p.answer_mode)
            if p.verify_framework and selected is not framework:
                verdict = verify_solution(selected, "selected_expert_majority")
                if not verdict["valid"]:
                    selected, decision = framework, "verified_framework_over_rejected_majority"
            return finish(selected, decision, "ok" if p.ablation == "full" else "ablation")
        except (PipelineFailure, SchemaError, BackendError) as exc:
            events.append({"event": "algebraic_fallback", "reason": str(exc)})
            return finish(experts[0], "algebraic_fallback", "fallback")


def solve_baseline(backend, config: Config, question: str, seed: int) -> dict:
    started = time.perf_counter()
    agent = JSONAgent(backend, config, seed)
    try:
        solution = agent.ask("baseline", {"question": question})
    except (BackendError, SchemaError) as exc:
        raise InferenceError(str(exc), {"question": question, "answer": "", "status": "error",
                                       "decision": "baseline_error", "error": str(exc), "seed": seed,
                                       "calls": agent.calls, "usage": agent.stats(), "events": [],
                                       "latency_seconds": time.perf_counter() - started}) from exc
    return {"question": question, "answer": solution["answer"], "solution": solution,
            "status": "ok", "decision": "single_model_baseline", "seed": seed,
            "calls": agent.calls, "events": [], "usage": agent.stats(),
            "latency_seconds": time.perf_counter() - started,
            "config_fingerprint": config.fingerprint()}
