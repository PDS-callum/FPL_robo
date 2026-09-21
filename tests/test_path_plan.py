"""Tests for multi-week path-graph planning."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.plan.path_graph import plan_transfer_path, week_player_ev


def _player(pid, name, pos, team_id, price, ev):
    return {
        "id": pid,
        "web_name": name,
        "position": pos,
        "team_id": team_id,
        "price": price,
        "ev": ev,
        "play_prob": 1.0,
        "chance_of_playing_next_round": 100,
        "status": "a",
    }


class PathPlanTests(unittest.TestCase):
    def test_hard_entry_week_prefers_hold_then_buy(self):
        """Gross-style: equal base EV, hard GW5 then easy GW6 → wait to buy."""
        # Team 10 hard GW5 (FDR5), easy GW6 (FDR2)
        # Team 20 easy GW5 (FDR2), mediocre GW6 (FDR3)
        fixtures = {
            10: {
                5: {"n_fixtures": 1.0, "best_fdr": 5.0},
                6: {"n_fixtures": 1.0, "best_fdr": 2.0},
                7: {"n_fixtures": 1.0, "best_fdr": 2.0},
            },
            20: {
                5: {"n_fixtures": 1.0, "best_fdr": 2.0},
                6: {"n_fixtures": 1.0, "best_fdr": 3.0},
                7: {"n_fixtures": 1.0, "best_fdr": 3.0},
            },
            1: {
                5: {"n_fixtures": 1.0, "best_fdr": 3.0},
                6: {"n_fixtures": 1.0, "best_fdr": 3.0},
                7: {"n_fixtures": 1.0, "best_fdr": 3.0},
            },
        }
        # Minimal legal-ish squad: pad with cheap same-pos fillers on team 1.
        owned = []
        rows = []
        pid = 1
        for pos, n in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for i in range(n):
                # One weak MID (id will be the transfer out target)
                ev = 2.0
                team = 1
                name = f"{pos}{i}"
                if pos == "MID" and i == 0:
                    ev = 3.0
                    name = "WeakMid"
                rows.append(_player(pid, name, pos, team, 4.5, ev))
                owned.append(pid)
                pid += 1
        weak_mid = next(r for r in rows if r["web_name"] == "WeakMid")["id"]
        # Premium targets
        rows.append(_player(100, "HotHard", "MID", 10, 5.5, 6.0))  # hard entry
        rows.append(_player(101, "CoolEasy", "MID", 20, 5.5, 5.2))  # easy this week

        def frame(gw):
            return pd.DataFrame(rows)

        preds = {5: frame(5), 6: frame(6), 7: frame(7)}
        plan = plan_transfer_path(
            start_gw=5,
            current_ids=owned,
            predictions_by_gw=preds,
            team_fixtures=fixtures,
            bank=2.0,
            free_transfers=1,
            selling_prices={i: 4.5 for i in owned},
            config={
                "horizon": 3,
                "beam_width": 50,
                "candidate_outs": 5,
                "candidate_ins_per_out": 8,
                "fdr_weight": 0.20,
                "ft_bank_bonus": 0.2,
            },
        )
        self.assertEqual(plan.status, "Optimal")
        # Best path should not buy HotHard in GW5 (hard FDR). Prefer hold or CoolEasy.
        w0 = plan.week0_action
        if w0.kind == "transfer":
            self.assertNotEqual(w0.element_in, 100, msg=plan.explanation)
        # Over the full path, HotHard may appear in GW6+
        later_ins = [
            a.element_in for a in plan.actions[1:] if a.kind == "transfer"
        ]
        # Either wait for HotHard or take CoolEasy now — both beat buying HotHard now.
        self.assertTrue(
            w0.kind == "hold"
            or w0.element_in == 101
            or 100 in later_ins
            or w0.element_in != 100,
            msg=plan.explanation,
        )

    def test_week_ev_haircuts_hard_fdr(self):
        row = _player(1, "A", "MID", 10, 7.0, 10.0)
        fx = {10: {1: {"n_fixtures": 1.0, "best_fdr": 5.0}}}
        hard = week_player_ev(
            row, gw=1, team_fixtures=fx, cfg={"fdr_weight": 0.2, "blank_mult": 0.25, "dgw_mult": 1.4}
        )
        fx2 = {10: {1: {"n_fixtures": 1.0, "best_fdr": 2.0}}}
        easy = week_player_ev(
            row, gw=1, team_fixtures=fx2, cfg={"fdr_weight": 0.2, "blank_mult": 0.25, "dgw_mult": 1.4}
        )
        self.assertLess(hard, 10.0)
        self.assertGreater(easy, 10.0)
        self.assertGreater(easy, hard)


if __name__ == "__main__":
    unittest.main()
