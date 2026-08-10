"""Tune/test split validation for the session-grid strategies' entry-tick
threshold, to separate parameter *search* from the number that validates it.

GRID_ENTRY_TICKS_TIGHT (used by full_day_grid_fade / morning_grid_fade) was
originally chosen by sweeping candidate values directly against the entire
out-of-sample (OOS) window that backtest_all_strategies then reports
performance on - a real selection-bias risk: the "out-of-sample" number was
partly a fit number. This module fixes that by splitting the OOS window
itself, chronologically, into an earlier TUNE sub-period (candidates are
swept and the best one picked here, using the same composite score the
Evaluator Agent uses) and a later, untouched TEST sub-period (the locked-in
choice is backtested there exactly once, and never used to pick anything)."""
import json

from crewai.tools import tool

from pair_crew import config
from pair_crew.tools.backtest_tools import build_backtest_dataframe, run_backtest_for_strategy

TUNE_TEST_PATH = config.OUTPUT_DIR / "tune_test_validation.json"


def split_tune_test(bt_df):
    """Chronological split of the OOS window: first TUNE_TEST_SPLIT_FRACTION
    for tuning, the untouched remainder for the held-out test."""
    n_tune = int(len(bt_df) * config.TUNE_TEST_SPLIT_FRACTION)
    return bt_df.iloc[:n_tune], bt_df.iloc[n_tune:]


def _composite_score(m: dict) -> float:
    """Identical formula to evaluate_tools.evaluate_strategies' composite
    score, so the TUNE-period selection criterion matches what the Evaluator
    Agent will later judge the final strategy on."""
    sharpe = max(min(m["sharpe_ratio_annualized"], 3), -3)
    calmar = max(min(m["calmar_ratio"], 3), -3) if m["calmar_ratio"] is not None else 0
    pf = max(min((m["profit_factor"] or 0) - 1, 2), -2)
    return round(0.4 * sharpe + 0.3 * calmar + 0.2 * pf + 0.1 * (m["win_rate"] - 0.5) * 2, 4)


def _tune_and_validate_one(base_strategy: dict, tune_df, test_df) -> dict:
    sid = base_strategy["id"]
    sweep = []
    for entry_ticks in config.TUNE_ENTRY_TICKS_CANDIDATES:
        strat = json.loads(json.dumps(base_strategy))
        strat["params"]["entry_ticks"] = entry_ticks
        m, _ = run_backtest_for_strategy(strat, tune_df)
        eligible = m["n_trades"] >= config.TUNE_MIN_TRADES
        sweep.append(
            {
                "entry_ticks": entry_ticks,
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "total_return_pct": m["total_return_pct"],
                "sharpe_ratio_annualized": m["sharpe_ratio_annualized"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "eligible": eligible,
                "composite_score": _composite_score(m) if eligible else None,
            }
        )

    eligible_rows = [r for r in sweep if r["eligible"]]
    chosen = max(eligible_rows, key=lambda r: r["composite_score"]) if eligible_rows else None

    result = {
        "strategy_id": sid,
        "tune_period": [str(tune_df.index.min()), str(tune_df.index.max())],
        "tune_n_bars": len(tune_df),
        "test_period": [str(test_df.index.min()), str(test_df.index.max())],
        "test_n_bars": len(test_df),
        "tune_min_trades_floor": config.TUNE_MIN_TRADES,
        "tune_sweep": sweep,
        "chosen_entry_ticks": chosen["entry_ticks"] if chosen else None,
        "chosen_by": "max composite_score among candidates with >= tune_min_trades_floor trades on the TUNE period",
    }

    if chosen is None:
        result["held_out_test_metrics"] = None
        result["note"] = (
            "No candidate met the tune-period minimum trade floor - no threshold could be "
            "honestly validated from this data; the TUNE sub-period is too thin for this "
            "strategy's trade frequency."
        )
        return result

    locked = json.loads(json.dumps(base_strategy))
    locked["params"]["entry_ticks"] = chosen["entry_ticks"]
    test_metrics, _ = run_backtest_for_strategy(locked, test_df)
    result["held_out_test_metrics"] = test_metrics
    return result


@tool("tune_test_split_grid_entry_ticks")
def tune_test_split_grid_entry_ticks() -> str:
    """
    Tune/test split validation of the entry-tick threshold for every
    session-grid strategy in outputs/strategies.json (engine="session_grid":
    afternoon_grid_fade, full_day_grid_fade, morning_grid_fade).

    Splits the existing out-of-sample backtest window chronologically in
    half: the earlier TUNE sub-period is used to sweep candidate entry_ticks
    values (config.TUNE_ENTRY_TICKS_CANDIDATES) and pick the best by the same
    composite score the Evaluator Agent uses, requiring at least
    config.TUNE_MIN_TRADES trades on the TUNE period for a candidate to be
    eligible. The chosen value is then backtested exactly once on the later
    TEST sub-period, which the sweep never touched - this is the honest,
    selection-bias-free out-of-sample number for that threshold.

    Saves the full sweep, the chosen threshold, and its held-out test metrics
    for every session-grid strategy to outputs/tune_test_validation.json and
    returns it as a JSON string.
    """
    strategies = json.loads(config.STRATEGIES_PATH.read_text())
    grid_strategies = [s for s in strategies if s.get("engine") == "session_grid"]

    bt_df = build_backtest_dataframe()
    tune_df, test_df = split_tune_test(bt_df)

    results = {s["id"]: _tune_and_validate_one(s, tune_df, test_df) for s in grid_strategies}
    TUNE_TEST_PATH.write_text(json.dumps(results, indent=2))
    return json.dumps(results, indent=2)
