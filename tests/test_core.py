"""Tests for meaningful mathematical, control-flow, data-isolation, and resume behavior."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from tbi.arithmetic import UnsupportedExpression, answer_equal, check_arithmetic, rational_value, reconcile
from tbi.backends import BackendError
from tbi.config import Config, derived_seed
from tbi.data import load_jsonl, record_from_row
from tbi.evaluation import evaluate, summarize
from tbi.pipeline import InferenceError, ThinkBeyondInstances
from tbi.schemas import SchemaError, parse_object, validate
from .fixtures import QUESTION, ReplayBackend, cards, paper_script, route, solution


class ArithmeticTests(unittest.TestCase):
    def test_exact_rational_arithmetic(self):
        self.assertEqual(rational_value("(10**3-370)/(3*10)"), 21)
        self.assertEqual(rational_value("0.1+0.2"), rational_value("0.3"))

    def test_unsupported_code_is_never_executed(self):
        for expression in ["__import__('os').getcwd()", "x+1", "[1][0]", "2**100000", "(1).__class__"]:
            with self.subTest(expression=expression), self.assertRaises(UnsupportedExpression):
                rational_value(expression)

    def test_failed_and_unknown_checks_are_distinct(self):
        result = check_arithmetic([{"left": "1/0", "right": "0"},
                                   {"left": "2+2", "right": "5"},
                                   {"left": "sqrt(2)", "right": "1.414"}])
        self.assertEqual([r["status"] for r in result], ["fail", "fail", "unknown"])

    def test_no_empty_consensus_or_loose_numeric_extraction(self):
        self.assertFalse(answer_equal("", ""))
        self.assertFalse(answer_equal("The answer is 58", "58", "numeric"))
        self.assertFalse(answer_equal("50%", "50", "numeric"))
        self.assertFalse(answer_equal("(1,2)", "12", "numeric"))
        self.assertTrue(answer_equal(r"\frac{1}{2}", "0.5", "numeric"))
        self.assertFalse(answer_equal("0.5", "1/2", "exact"))

    def test_final_reconciliation_all_three_branches(self):
        cases = [("58", ["142", "142", "58"], "58", "framework_matches_expert"),
                 ("17", ["142", "142", "58"], "142", "expert_majority"),
                 ("17", ["142", "24", "58"], "17", "framework_all_experts_disagree")]
        for framework, experts, answer, decision in cases:
            selected, actual_decision = reconcile(solution(framework), [solution(a) for a in experts], "exact")
            self.assertEqual((selected["answer"], actual_decision), (answer, decision))


class SchemaTests(unittest.TestCase):
    def test_thinking_block_and_nested_latex(self):
        value = {"answer": r"\frac{1}{2}", "steps": ["Step 1: Derive the fraction"]}
        parsed = parse_object("<think>Internal text</think>\n```json\n" + json.dumps(value) + "\n```")
        self.assertEqual(parsed, value)

    def test_invalid_booleans_and_truncated_json_are_rejected(self):
        for value in ["false", "true", 1, None]:
            with self.assertRaises(SchemaError):
                validate("verify", {"valid": value, "reason": "test"})
        with self.assertRaises(SchemaError):
            parse_object('<think>{"answer": "58"}')
        with self.assertRaises(SchemaError):
            parse_object('{"answer":"58"')

    def test_router_requires_evidence_for_every_variant(self):
        with self.assertRaises(SchemaError):
            validate("route", route(0), variants=2)
        bad = route(0, 2)
        bad["transfer"][1]["variant_index"] = 0
        with self.assertRaises(SchemaError):
            validate("route", bad, variants=2)

    def test_exactly_four_distinct_card_ids(self):
        obj = cards(0)
        obj["cards"][1]["id"] = "A"
        with self.assertRaises(SchemaError):
            validate("cards", obj)


class PipelineTests(unittest.TestCase):
    def config(self) -> Config:
        cfg = Config()
        cfg.pipeline.parallel_experts = False
        cfg.pipeline.json_repair_attempts = 0
        return cfg

    def test_successful_full_pipeline_and_priority_over_wrong_majority(self):
        backend = ReplayBackend(paper_script())
        result = ThinkBeyondInstances(backend, self.config()).solve(QUESTION)
        self.assertEqual(result["answer"], "58")
        self.assertEqual(len(result["chain"]), 3)
        self.assertEqual(result["decision"], "framework_matches_expert")
        self.assertEqual(result["usage"]["total_requests"], 14)

    def test_unanimous_experts_skip_abstraction(self):
        script = {name: [solution("5")] for name in ["expert_algebraic", "expert_strategic", "expert_geometric"]}
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve("If 3*x+7=22, find x.")
        self.assertEqual(result["decision"], "unanimous_experts")
        self.assertEqual(result["usage"]["total_requests"], 3)

    def test_router_rejection_regenerates_with_feedback(self):
        script = paper_script()
        script["cards"].insert(0, cards(0))
        script["route"].insert(0, {"selected_id": None, "reason": "ALE leaked original numbers", "transfer": []})
        backend = ReplayBackend(script)
        result = ThinkBeyondInstances(backend, self.config()).solve(QUESTION)
        self.assertEqual(result["status"], "ok")
        card_requests = [r for r in backend.requests if r["stage"] == "cards"]
        self.assertIn("ALE leaked original numbers", card_requests[1]["messages"][1]["content"])

    def test_mutation_failure_does_not_reuse_original(self):
        script = paper_script()
        script["mutate"] = [{"question": QUESTION}] * 3
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["variants"], [])
        self.assertEqual(result["answer"], "142")

    def test_structurally_different_mutation_is_rejected(self):
        script = paper_script()
        script["mutate"] *= 3
        script["validate_mutation"] = [{"valid": True, "same_structure": False,
                                       "reason": "The target changed"}] * 3
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual(result["status"], "fallback")

    def test_single_rollback_rebuilds_dependent_suffix(self):
        script = paper_script()
        script["audit"].insert(0, {"valid": False, "bad_step": 2, "reason": "Step 2 must be recomputed"})
        script["cards"] += [cards(1), cards(2)]
        script["route"] += [route(1), route(2)]
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual(result["status"], "ok")
        rollback = [e for e in result["events"] if e["event"] == "chain_rollback"]
        self.assertEqual(rollback[0]["rebuilt_from_step"], 2)
        self.assertEqual(len(result["chain"]), 3)

    def test_second_failed_chain_audit_falls_back_to_algebraic_expert(self):
        script = paper_script()
        script["audit"] = [{"valid": False, "bad_step": 2, "reason": "Dependency failure"}] * 2
        script["cards"] += [cards(1), cards(2)]
        script["route"] += [route(1), route(2)]
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual((result["answer"], result["status"]), ("142", "fallback"))
        self.assertEqual(sum(e["event"] == "chain_rollback" for e in result["events"]), 1)

    def test_module_backend_error_produces_a_logged_fallback(self):
        script = paper_script()
        script["mutate"] = [BackendError("Unavailable endpoint")]
        result = ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["usage"]["backend_errors"], 1)

    def test_json_repair_is_a_real_additional_request(self):
        cfg = self.config()
        cfg.pipeline.json_repair_attempts = 1
        script = paper_script()
        script["integrate"].insert(0, "invalid JSON")
        result = ThinkBeyondInstances(ReplayBackend(script), cfg).solve(QUESTION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["usage"]["schema_failures"], 1)

    def test_enhanced_two_variants_and_numeric_checks(self):
        cfg = self.config()
        cfg.pipeline.mutation_count = 2
        cfg.pipeline.numeric_checks = True
        cfg.pipeline.verify_framework = True
        result = ThinkBeyondInstances(ReplayBackend(paper_script(2)), cfg).solve(QUESTION)
        self.assertEqual(result["answer"], "58")
        self.assertEqual(len(result["variants"]), 2)
        reports = [r for e in result["events"] if e["event"] == "arithmetic_check" for r in e["reports"]]
        self.assertTrue(reports)
        self.assertTrue(all(r["status"] == "pass" for r in reports))

    def test_bad_numeric_check_triggers_card_regeneration(self):
        cfg = self.config()
        cfg.pipeline.numeric_checks = True
        script = paper_script()
        bad = cards(0)
        bad["cards"][0]["checks"] = [{"left": "2+2", "right": "5"}]
        script["cards"].insert(0, bad)
        script["route"].insert(0, route(0))
        result = ThinkBeyondInstances(ReplayBackend(script), cfg).solve(QUESTION)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(any(r["status"] == "fail" for e in result["events"] if e["event"] == "arithmetic_check" for r in e["reports"]))

    def test_solve_signature_has_no_reference_answer(self):
        self.assertEqual(set(inspect.signature(ThinkBeyondInstances.solve).parameters), {"self", "question", "seed"})

    def test_expert_failure_preserves_real_calls_and_error_stats(self):
        script = paper_script()
        script["expert_strategic"] = [BackendError("Model unavailable")]
        with self.assertRaises(InferenceError) as captured:
            ThinkBeyondInstances(ReplayBackend(script), self.config()).solve(QUESTION)
        self.assertEqual(captured.exception.trace["usage"]["total_requests"], 2)
        self.assertEqual(captured.exception.trace["usage"]["backend_errors"], 1)

    def test_enhanced_consensus_rejection_runs_full_pipeline(self):
        cfg = self.config()
        cfg.pipeline.verify_consensus = True
        script = paper_script()
        script["expert_geometric"] = [solution("142")]
        script["verify"] = [{"valid": False, "reason": "Incorrect square sum"}]
        result = ThinkBeyondInstances(ReplayBackend(script), cfg).solve(QUESTION)
        self.assertEqual(len(result["chain"]), 3)
        self.assertTrue(any(e["event"] == "consensus_rejected" for e in result["events"]))

    def test_enhanced_framework_verification_retries_execution(self):
        cfg = self.config()
        cfg.pipeline.verify_framework = True
        script = paper_script()
        script["generate"].insert(0, solution("59"))
        script["verify"] = [{"valid": False, "reason": "Final subtraction is wrong"},
                            {"valid": True, "reason": "Calculation is correct"}]
        result = ThinkBeyondInstances(ReplayBackend(script), cfg).solve(QUESTION)
        self.assertEqual(result["answer"], "58")
        self.assertEqual(sum(c["stage"] == "generate" for c in result["calls"]), 2)

    def test_enhanced_rejects_incorrect_selected_expert_majority(self):
        cfg = self.config()
        cfg.pipeline.verify_framework = True
        script = paper_script()
        script["generate"] = [solution("17")]
        script["verify"] = [{"valid": True, "reason": "Scripted framework validation"},
                            {"valid": False, "reason": "Scripted majority rejection"}]
        result = ThinkBeyondInstances(ReplayBackend(script), cfg).solve(QUESTION)
        self.assertEqual(result["answer"], "17")
        self.assertEqual(result["decision"], "verified_framework_over_rejected_majority")


class EvaluationTests(unittest.TestCase):
    def test_changing_labels_does_not_change_inference_seeds_or_requests(self):
        cfg = Config()
        cfg.pipeline.parallel_experts = False
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            for index, label in enumerate(["58", "A_DIFFERENT_REFERENCE"]):
                backend = ReplayBackend(paper_script())
                evaluate(backend, cfg, [{"id": "same_id", "question": QUESTION, "answer": label}],
                         Path(directory) / str(index), repeats=1, scorer="exact", method="tbi",
                         metadata={"dataset_hash": f"different_hash_{index}"})
                requests.append(backend.requests)
        self.assertEqual(requests[0], requests[1])

    def test_labels_are_isolated_and_resume_skips_completed_trials(self):
        cfg = Config()
        cfg.pipeline.parallel_experts = False
        backend = ReplayBackend(paper_script())
        label = "SECRET_REFERENCE_LABEL_7e843f"
        items = [{"id": "example", "question": QUESTION, "answer": label}]
        metadata = {"dataset_hash": "test-dataset", "source_code_hash": "test-source"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = evaluate(backend, cfg, items, output, repeats=1, scorer="exact",
                             method="tbi", metadata=metadata)
            self.assertEqual(first["accuracy"], 0.0)
            self.assertNotIn(label, json.dumps(backend.requests))
            calls = len(backend.requests)
            second = evaluate(backend, cfg, items, output, repeats=1, scorer="exact",
                              method="tbi", metadata=metadata, resume=True)
            self.assertEqual(second["accuracy"], 0.0)
            self.assertEqual(len(backend.requests), calls)
            cfg.seed += 1
            with self.assertRaises(ValueError):
                evaluate(backend, cfg, items, output, repeats=1, scorer="exact",
                         method="tbi", metadata=metadata, resume=True)

    def test_avg_at_n_differs_from_any_success(self):
        records = [{"problem_id": "p", "run": i, "correct": i == 0, "status": "ok"} for i in range(4)]
        metrics = summarize(records, 4, "exact")
        self.assertEqual(metrics["avg_at_n"], 0.25)
        self.assertEqual(metrics["any_success_at_n"], 1.0)

    def test_error_trials_are_not_dropped(self):
        metrics = summarize([{"problem_id": "p", "run": 0, "correct": False, "status": "error"}], 1, "exact")
        self.assertEqual((metrics["accuracy"], metrics["error_trials"]), (0.0, 1))

    def test_seed_derivation_is_stable_and_separates_trials(self):
        self.assertEqual(derived_seed(42, "p", 0), derived_seed(42, "p", 0))
        self.assertNotEqual(derived_seed(42, "p", 0), derived_seed(42, "p", 1))

    def test_dataset_adapter_and_duplicate_detection(self):
        row = record_from_row({"question": "Compute 1+1", "final_answer": ["2"]},
                              {"question_field": "question", "answer_field": "final_answer"}, 0)
        self.assertEqual(row["answer"], "2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
