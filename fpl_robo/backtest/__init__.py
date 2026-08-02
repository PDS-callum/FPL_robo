"""Season-level backtesting."""

from fpl_robo.backtest.season import SeasonBacktestResult, run_season_backtest, write_season_report
from fpl_robo.backtest.vector_season import (
    compare_vector_vs_baseline,
    run_vector_evaluation_suite,
    run_vector_season_backtest,
)

__all__ = [
    "SeasonBacktestResult",
    "run_season_backtest",
    "write_season_report",
    "run_vector_season_backtest",
    "compare_vector_vs_baseline",
    "run_vector_evaluation_suite",
]
