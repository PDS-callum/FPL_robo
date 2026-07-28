"""Unit tests for optimiser quality tie-break helpers."""

from __future__ import annotations

import unittest

import pandas as pd
import pulp

from fpl_robo.optimize.tiebreak import attach_quality_scores, solve_lexicographic


class QualityTiebreakTests(unittest.TestCase):
    def test_attach_quality_scores_adds_column(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "A",
                    "form": 8.0,
                    "expected_goals": 0.6,
                    "expected_assists": 0.3,
                    "ict_index": 90.0,
                    "selected_by_percent": 40.0,
                    "ev": 5.0,
                },
                {
                    "id": 2,
                    "web_name": "B",
                    "form": 2.0,
                    "expected_goals": 0.0,
                    "expected_assists": 0.0,
                    "ict_index": 10.0,
                    "selected_by_percent": 1.0,
                    "ev": 5.0,
                },
            ]
        )
        out = attach_quality_scores(df)
        self.assertIn("quality", out.columns)
        self.assertGreater(float(out.loc[0, "quality"]), float(out.loc[1, "quality"]))

    def test_attach_quality_scores_uses_kg_context(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "A",
                    "form": 5.0,
                    "xgi90": 0.7,
                    "minutes": 2000,
                    "ict_index": 50.0,
                    "selected_by_percent": 10.0,
                    "context_factor": 1.2,
                    "set_piece_score": 0.08,
                    "role_stability": 1.0,
                    "is_focal_attacker": True,
                    "ev": 5.0,
                },
                {
                    "id": 2,
                    "web_name": "B",
                    "form": 5.0,
                    "xgi90": 0.1,
                    "minutes": 2000,
                    "ict_index": 50.0,
                    "selected_by_percent": 10.0,
                    "context_factor": 0.9,
                    "set_piece_score": 0.0,
                    "role_stability": 0.92,
                    "is_focal_attacker": False,
                    "ev": 5.0,
                },
            ]
        )
        out = attach_quality_scores(df)
        self.assertGreater(float(out.loc[0, "quality"]), float(out.loc[1, "quality"]))
        self.assertIn("q_context", out.columns)
        self.assertIn("q_set_piece", out.columns)

    def test_attach_quality_scores_zeros_thin_sample_xgi90(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "Cameo",
                    "form": 5.0,
                    "xgi90": 3.6,
                    "minutes": 2,
                    "ict_index": 50.0,
                    "selected_by_percent": 0.1,
                    "ev": 5.0,
                },
                {
                    "id": 2,
                    "web_name": "Solid",
                    "form": 5.0,
                    "xgi90": 0.5,
                    "minutes": 2000,
                    "ict_index": 50.0,
                    "selected_by_percent": 10.0,
                    "ev": 5.0,
                },
            ]
        )
        out = attach_quality_scores(df)
        self.assertEqual(float(out.loc[0, "q_xgi"]), 0.0)
        self.assertGreater(float(out.loc[1, "q_xgi"]), 0.0)

    def test_lexicographic_keeps_primary_then_picks_quality(self):
        # Higher primary always wins, even with worse quality.
        prob = pulp.LpProblem("lex_toy", pulp.LpMaximize)
        a = pulp.LpVariable("a", cat="Binary")
        b = pulp.LpVariable("b", cat="Binary")
        c = pulp.LpVariable("c", cat="Binary")
        prob += a + b + c == 1
        primary = 5 * a + 5 * b + 6 * c
        secondary = 10 * a + 1 * b + 0 * c
        status = solve_lexicographic(prob, primary, secondary, tolerance=1e-6)
        self.assertEqual(pulp.LpStatus[status], "Optimal")
        self.assertGreater(c.value(), 0.5)

        # Equal primary → higher quality wins.
        prob2 = pulp.LpProblem("lex_toy2", pulp.LpMaximize)
        x = pulp.LpVariable("x", cat="Binary")
        y = pulp.LpVariable("y", cat="Binary")
        prob2 += x + y == 1
        primary2 = 5 * x + 5 * y
        secondary2 = 10 * x + 1 * y
        status2 = solve_lexicographic(prob2, primary2, secondary2, tolerance=1e-6)
        self.assertEqual(pulp.LpStatus[status2], "Optimal")
        self.assertGreater(x.value(), 0.5)
        self.assertLess(y.value(), 0.5)


if __name__ == "__main__":
    unittest.main()
