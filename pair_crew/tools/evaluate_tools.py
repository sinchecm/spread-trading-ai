"""Evaluator Agent tool: stability-test, rank, reject overfit/unstable
strategies, and select the final strategy."""
import json

import numpy as np

from crewai.tools import tool

from pair_crew import config
from pair_crew.tools.backtest_tools import build_backtest_dataframe, run_backtest_for_strategy

MIN_TRADES_FOR_SIGNIFICANCE = 8
MAX_ACCEPTABLE_DRAWDOWN_PCT = -20.0


@tool("evaluate_strategies")
def evaluate_strategies() -> str:
    """
    Compare all backtested strategies (outputs/backtest_results.json /
    outputs/strategies.json), stress-test each for stability by re-running
    the identical backtest separately on the first and second half of the
    out-of-sample period, and reject strategies that are overfit, based on
    too few trades to be statistically meaningful, unprofitable after
    costs, or unstable (profitable in one half of the OOS period and sharply
    unprofitable in the other).

    Surviving strategies are ranked by a composite score combining
    out-of-sample Sharpe ratio, Calmar ratio (return / max drawdown), win
    rate, and profit factor. The top-ranked surviving strategy is selected
    as the final recommendation, with an explicit justification covering
    out-of-sample performance, risk-adjusted return, drawdown control, and
    practicality (trade frequency / cost burden relative to edge).

    Saves the full evaluation to outputs/evaluation.json and returns it as a
    JSON string.
    """
    strategies = {s["id"]: s for s in json.loads(config.STRATEGIES_PATH.read_text())}
    backtest = json.loads(config.BACKTEST_RESULTS_PATH.read_text())["results"]
    bt_df = build_backtest_dataframe()
    mid = len(bt_df) // 2
    first_half, second_half = bt_df.iloc[:mid], bt_df.iloc[mid:]

    evaluation = {}
    for sid, strat in strategies.items():
        full = backtest[sid]
        half1, _ = run_backtest_for_strategy(strat, first_half)
        half2, _ = run_backtest_for_strategy(strat, second_half)

        reasons_to_reject = []
        if full["n_trades"] < MIN_TRADES_FOR_SIGNIFICANCE:
            reasons_to_reject.append(
                f"only {full['n_trades']} trades in the OOS period - too few to draw a "
                "statistically reliable conclusion, even though headline metrics may look good"
            )
        if full["profit_factor"] is not None and full["profit_factor"] < 1.0:
            reasons_to_reject.append("profit factor below 1.0 - unprofitable after costs")
        if full["sharpe_ratio_annualized"] <= 0:
            reasons_to_reject.append("non-positive out-of-sample Sharpe ratio")
        if full["max_drawdown_pct"] < MAX_ACCEPTABLE_DRAWDOWN_PCT:
            reasons_to_reject.append(
                f"max drawdown {full['max_drawdown_pct']}% exceeds the "
                f"{MAX_ACCEPTABLE_DRAWDOWN_PCT}% risk tolerance"
            )
        half_returns = [half1["total_return_pct"], half2["total_return_pct"]]
        unstable = (half_returns[0] > 0 and half_returns[1] < -2 * abs(half_returns[0])) or (
            half_returns[1] > 0 and half_returns[0] < -2 * abs(half_returns[1])
        )
        if unstable:
            reasons_to_reject.append(
                f"unstable across sub-periods (half-1 return {half_returns[0]}%, "
                f"half-2 return {half_returns[1]}%) - looks overfit to one regime"
            )

        sharpe_clipped = max(min(full["sharpe_ratio_annualized"], 3), -3)
        calmar_clipped = max(min(full["calmar_ratio"], 3), -3) if full["calmar_ratio"] is not None else 0
        pf_score = max(min((full["profit_factor"] or 0) - 1, 2), -2)
        composite_score = round(
            0.4 * sharpe_clipped + 0.3 * calmar_clipped + 0.2 * pf_score + 0.1 * (full["win_rate"] - 0.5) * 2,
            4,
        )

        evaluation[sid] = {
            "name": strat["name"],
            "full_period_metrics": full,
            "sub_period_stability": {
                "first_half": {
                    "total_return_pct": half1["total_return_pct"],
                    "sharpe_ratio_annualized": half1["sharpe_ratio_annualized"],
                    "n_trades": half1["n_trades"],
                },
                "second_half": {
                    "total_return_pct": half2["total_return_pct"],
                    "sharpe_ratio_annualized": half2["sharpe_ratio_annualized"],
                    "n_trades": half2["n_trades"],
                },
                "flagged_unstable": unstable,
            },
            "composite_score": composite_score,
            "rejected": len(reasons_to_reject) > 0,
            "rejection_reasons": reasons_to_reject,
        }

    survivors = {k: v for k, v in evaluation.items() if not v["rejected"]}
    if survivors:
        final_id = max(survivors, key=lambda k: survivors[k]["composite_score"])
        final = evaluation[final_id]
        justification = (
            f"'{final['name']}' ({final_id}) is the top-ranked strategy that survives all "
            "rejection filters. Out-of-sample it produced a Sharpe ratio of "
            f"{final['full_period_metrics']['sharpe_ratio_annualized']}, a Calmar ratio of "
            f"{final['full_period_metrics']['calmar_ratio']}, a win rate of "
            f"{final['full_period_metrics']['win_rate']*100:.1f}%, and a max drawdown of "
            f"{final['full_period_metrics']['max_drawdown_pct']}% over "
            f"{final['full_period_metrics']['n_trades']} trades, with consistent (not purely "
            "one-regime) performance across both halves of the out-of-sample window, and a "
            "trade count high enough to be implementable and statistically meaningful without "
            "being so high that costs dominate."
        )
    else:
        final_id = None
        tentative_id = max(evaluation, key=lambda k: evaluation[k]["composite_score"])
        tentative = evaluation[tentative_id]
        justification = (
            "No strategy survived the rejection filters on this out-of-sample window. All "
            "candidates were either unprofitable after costs, had too few trades to be "
            "statistically meaningful, breached the drawdown tolerance, or were unstable "
            "across sub-periods. Recommendation: do NOT deploy any variant as-is. The least-bad "
            f"candidate, '{tentative['name']}' ({tentative_id}, composite score "
            f"{tentative['composite_score']}), is flagged only as a starting point for further "
            "research (more history, looser thresholds, larger universe of pairs) rather than a "
            "validated, ready-to-trade strategy - its own rejection reasons were: "
            f"{'; '.join(tentative['rejection_reasons'])}."
        )

    result = {
        "evaluation_by_strategy": evaluation,
        "final_selected_strategy_id": final_id,
        "tentative_pick_if_none_survive": None if final_id else tentative_id,
        "justification": justification,
        "rejection_criteria_used": {
            "min_trades_for_significance": MIN_TRADES_FOR_SIGNIFICANCE,
            "max_acceptable_drawdown_pct": MAX_ACCEPTABLE_DRAWDOWN_PCT,
            "requires_profit_factor_gte": 1.0,
            "requires_positive_sharpe": True,
            "requires_sub_period_stability": True,
        },
    }
    config.EVALUATION_PATH.write_text(json.dumps(result, indent=2))
    return json.dumps(result, indent=2)
