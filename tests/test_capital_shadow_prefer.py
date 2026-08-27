"""Capital / selling-price and shadow-prefer gate regressions."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import pandas as pd

from fpl_robo.data.prices import effective_sell_price, pick_selling_price_pounds
from fpl_robo.optimize.single_gw import SquadSolution
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.runtime.season_runner import (
    _apply_shadow_mip_prefer_gate,
    _is_cheap_same_pos_swap,
)


class SellPriceTests(unittest.TestCase):
    def test_missing_or_zero_selling_falls_back(self):
        self.assertIsNone(pick_selling_price_pounds({"selling_price": 0}))
        self.assertIsNone(pick_selling_price_pounds({}))
        self.assertEqual(pick_selling_price_pounds({"selling_price": 75}), 7.5)
        self.assertEqual(pick_selling_price_pounds({"purchase_price": 50}), 5.0)

    def test_effective_sell_ignores_zero_entries(self):
        self.assertEqual(
            effective_sell_price(1, {1: 0.0}, fallback=7.4),
            7.4,
        )
        self.assertEqual(
            effective_sell_price(1, {1: 7.4}, fallback=9.0),
            7.4,
        )
        self.assertEqual(
            effective_sell_price(2, {1: 7.4}, fallback=5.0),
            5.0,
        )


class ShadowPreferTests(unittest.TestCase):
    def _preds(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"id": 469, "web_name": "N.Williams", "position": "DEF", "price": 5.0, "ev": 1.2, "ep_anchor": 1.2},
                {"id": 532, "web_name": "Ballard", "position": "DEF", "price": 5.0, "ev": 1.8, "ep_anchor": 1.8},
                {"id": 106, "web_name": "Thiago", "position": "FWD", "price": 8.0, "ev": 2.3, "ep_anchor": 2.3},
                {"id": 165, "web_name": "João Pedro", "position": "FWD", "price": 7.6, "ev": 3.2, "ep_anchor": 3.2},
            ]
        )

    def _plan(self, transfers, *, made=None) -> TransferPlan:
        sol = MagicMock(spec=SquadSolution)
        return TransferPlan(
            status="Optimal",
            objective=30.0,
            budget_left=0.0,
            free_transfers=1,
            transfers_made=made if made is not None else len(transfers),
            hits=0,
            hit_cost=0,
            transfers=transfers,
            solution=sol,
            meta={"consensus": {"mode": "best_obj"}},
        )

    def test_cheap_def_swap_detected(self):
        preds = self._preds()
        self.assertTrue(
            _is_cheap_same_pos_swap(
                [{"element_out": 469, "element_in": 532}],
                preds,
                max_price=5.5,
            )
        )
        self.assertFalse(
            _is_cheap_same_pos_swap(
                [{"element_out": 106, "element_in": 165}],
                preds,
                max_price=5.5,
            )
        )

    def test_prefer_shadow_over_cheap_vector_swap(self):
        preds = self._preds()
        vector = self._plan([{"element_out": 469, "element_in": 532}])
        shadow = self._plan([{"element_out": 106, "element_in": 165}])
        shadow.objective = 33.0
        out = _apply_shadow_mip_prefer_gate(
            plan=vector,
            shadow=shadow,
            predictions=preds,
            owned_risk_ids=set(),
        )
        self.assertEqual(out.transfers[0]["element_in"], 165)
        self.assertTrue((out.meta or {}).get("shadow_mip_prefer", {}).get("applied"))


if __name__ == "__main__":
    unittest.main()
