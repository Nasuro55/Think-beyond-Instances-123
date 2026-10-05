"""Optional real mathematical grading tests, separate from inference."""

import importlib.util
import unittest

from tbi.scoring import score_answer


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "Optional Math-Verify dependency is not installed")
class MathVerifyTests(unittest.TestCase):
    def test_fraction_and_algebraic_equivalence(self):
        self.assertTrue(score_answer("0.5", r"\frac{1}{2}", "math_verify")["correct"])
        self.assertTrue(score_answer("2*(x+1)", "2*x+2", "math_verify")["correct"])

    def test_latex_coordinates(self):
        self.assertTrue(score_answer(r"(3,\frac{\pi}{2})", r"\left(3,\frac{\pi}{2}\right)", "math_verify")["correct"])

    def test_sets_and_ordered_tuples_are_distinct(self):
        self.assertFalse(score_answer("(1,2)", r"\{1,2\}", "math_verify")["correct"])

    def test_percent_keeps_its_semantics(self):
        self.assertFalse(score_answer("50%", "50", "math_verify")["correct"])
        self.assertTrue(score_answer("50%", "0.5", "math_verify")["correct"])

    def test_incorrect_or_empty_prediction_fails(self):
        self.assertFalse(score_answer("58", "59", "math_verify")["correct"])
        self.assertFalse(score_answer("", "59", "math_verify")["correct"])


if __name__ == "__main__":
    unittest.main()
