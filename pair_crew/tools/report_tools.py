"""Evaluator Agent tool: compile every stage's saved artifacts into the
final markdown deliverable."""
import json
from datetime import datetime

from crewai.tools import tool

from pair_crew import config


def _load(path):
    return json.loads(path.read_text())


def _span_phrase(start_str: str, end_str: str) -> str:
    """Human-readable span between two timestamps, choosing days/weeks/
    months/years so the report stays accurate whether the loaded dataset is
    a few months or several years."""
    days = (datetime.fromisoformat(end_str) - datetime.fromisoformat(start_str)).days
    if days < 14:
        return f"~{days} days"
    if days < 60:
        return f"~{days // 7} weeks"
    if days < 730:
        return f"~{days / 30.44:.1f} months"
    return f"~{days / 365.25:.1f} years"


@tool("compile_final_report")
def compile_final_report() -> str:
    """
    Compile the final markdown deliverable from every artifact produced
    earlier in the pipeline: outputs/data_preprocessing_summary.json,
    outputs/pair_stats.json, outputs/feature_engineering_summary.json,
    outputs/model_comparison.json, outputs/strategies.json,
    outputs/backtest_results.json, and outputs/evaluation.json.

    Must be called last, after evaluate_strategies. Writes the full report
    to outputs/final_report.md with sections: executive summary, data
    preprocessing summary, pair relationship analysis, feature engineering,
    model comparison, recommended ML model, candidate strategies, backtest
    results, strategy evaluation, final selected strategy, and key
    limitations. Returns the full markdown text.
    """
    data_summary = _load(config.DATA_SUMMARY_PATH)
    pair_stats = _load(config.PAIR_STATS_PATH)
    feat_summary = _load(config.FEATURE_SUMMARY_PATH)
    model_comp = _load(config.MODEL_COMPARISON_PATH)
    strategies = _load(config.STRATEGIES_PATH)
    backtest = _load(config.BACKTEST_RESULTS_PATH)
    evaluation = _load(config.EVALUATION_PATH)

    train_stats = pair_stats["train_window_stats"]
    final_id = evaluation["final_selected_strategy_id"]
    tentative_id = evaluation.get("tentative_pick_if_none_survive")
    strategies_by_id = {s["id"]: s for s in strategies}
    n_strategies = len(strategies)

    lines = []
    A = lines.append

    title_suffix = (
        "CrewAI Multi-Agent Analysis"
        if config.ENTRY_POINT == "crewai_agents"
        else "Deterministic Pipeline Analysis"
    )
    A(f"# MHI / HHI Pair Trading Strategy — {title_suffix}\n")

    # --- Executive summary ---
    A("## 1. Executive Summary\n")
    if final_id:
        strat = strategies_by_id[final_id]
        A(
            f"After cleaning and aligning both instruments, testing the pair for cointegration, "
            f"engineering spread-based features, comparing several ML models with time-series-aware "
            f"validation, building and backtesting {n_strategies} candidate strategies with realistic costs, "
            f"and stress-testing each for stability, the **{strat['name']}** strategy (`{final_id}`) "
            f"is selected as the final recommendation. {evaluation['justification']}\n"
        )
    else:
        A(
            f"After cleaning and aligning both instruments, testing the pair for cointegration, "
            f"engineering spread-based features, comparing several ML models with time-series-aware "
            f"validation, and building/backtesting {n_strategies} candidate strategies with realistic costs, "
            f"**no strategy survived rigorous out-of-sample stability and profitability screening** "
            f"on this dataset. {evaluation['justification']}\n"
        )

    # --- Data preprocessing ---
    A("## 2. Data Preprocessing Summary\n")
    A(f"- Raw MHI bars: {data_summary['mhi_raw']['raw_rows']}, raw HHI bars: {data_summary['hhi_raw']['raw_rows']} (both 1-minute, trade-driven, no duplicate timestamps or invalid OHLC rows found).")
    A(f"- Both series resampled to **{data_summary['resample_freq']}** bars; short no-trade gaps forward-filled up to {data_summary['max_forward_fill_bars']} consecutive bars (last-traded-price carry-forward, causal / no lookahead).")
    A(f"- MHI resampled bars with no trade: {data_summary['mhi_resampled']['bars_with_no_trade']}/{data_summary['mhi_resampled']['bars_after_resample']}; HHI: {data_summary['hhi_resampled']['bars_with_no_trade']}/{data_summary['hhi_resampled']['bars_after_resample']}.")
    A(f"- Instruments inner-aligned on their overlapping calendar range {data_summary['overlap_date_range'][0]} to {data_summary['overlap_date_range'][1]}, yielding **{data_summary['aligned_rows']} aligned bars** ({data_summary['aligned_date_range'][0]} to {data_summary['aligned_date_range'][1]}).")
    A(f"- {data_summary['notes']}\n")

    # --- Pair relationship ---
    A("## 3. Pair Relationship Analysis\n")
    A(f"**Verdict: {pair_stats['verdict']}** — {pair_stats['reasoning']}\n")
    A("Training-window statistics (used for the verdict, to avoid lookahead into the out-of-sample test period):\n")
    A(f"- Price correlation: {train_stats['price_correlation']}, return correlation: {train_stats['return_correlation']}")
    fhr = train_stats["fixed_hedge_ratio"]
    A(f"- Fixed HKEX contract ratio (traded, not estimated): {fhr['mhi_lots']} MHI lots (x{fhr['mhi_multiplier']} multiplier) : {fhr['hhi_lots']} HHI lot(s) (x{fhr['hhi_multiplier']} multiplier), HHI/MHI notional ratio={fhr['ratio']}")
    implied = train_stats["log_price_ols_implied_ratio"]
    A(f"- For comparison only, an unconstrained log-price OLS regression implies alpha={implied['alpha']}, beta={implied['beta']} — the pipeline trades the fixed contract ratio above, not this estimate")
    A(f"- ADF test on the fixed-ratio spread: statistic={train_stats['adf_test_on_spread']['statistic']}, p-value={train_stats['adf_test_on_spread']['p_value']} (stationary at 5%: {train_stats['adf_test_on_spread']['stationary_at_5pct']})")
    A(f"- Engle-Granger cointegration: statistic={train_stats['engle_granger_cointegration']['statistic']}, p-value={train_stats['engle_granger_cointegration']['p_value']} (cointegrated at 10%: {train_stats['engle_granger_cointegration']['cointegrated_at_10pct']})")
    _hl_bars = train_stats["mean_reversion_half_life_bars"] or 0
    A(f"- Mean-reversion half-life: ~{train_stats['mean_reversion_half_life_bars']} bars (~{_hl_bars / 1440:.2f} days at 1-min bars)")
    A(f"- Hurst exponent of spread: {train_stats['hurst_exponent_of_spread']} ({'mean-reverting tendency' if train_stats['hurst_exponent_of_spread'] < 0.5 else 'mild trending tendency — mixed signal with the cointegration result'})")
    A(f"- Rolling {train_stats['rolling_return_correlation']['window_bars']}-bar return correlation: mean={train_stats['rolling_return_correlation']['mean']}, std={train_stats['rolling_return_correlation']['std']} (correlation is time-varying, not constant)\n")

    # --- Feature engineering ---
    A("## 4. Feature Engineering Summary\n")
    A(f"- Rolling windows: short={feat_summary['rolling_window_short_bars']} bars, long={feat_summary['rolling_window_long_bars']} bars; label horizon={feat_summary['label_horizon_bars']} bars.")
    A(f"- Features: {', '.join(feat_summary['feature_columns'])}.")
    A(f"- {feat_summary['rows_after_dropping_warmup_and_label_horizon']} usable rows remained after dropping rolling-window warm-up and label-horizon rows ({feat_summary['rows_dropped']} dropped).")
    A("- Three forward-looking targets were engineered:")
    for t, desc in feat_summary["target_definitions"].items():
        A(f"  - `{t}` (positive rate {feat_summary['target_positive_class_rate'][t]*100:.1f}%): {desc}")
    A(f"- {feat_summary['no_lookahead_design_notes']}\n")

    # --- Model comparison ---
    A("## 5. Model Comparison (time-series-aware, embargoed walk-forward CV)\n")
    for target, models in model_comp["model_comparison_by_target"].items():
        A(f"**Target: `{target}`**\n")
        A("| Model | CV AUC (mean) | CV AUC (std) | CV Accuracy | CV F1 | Robust score |")
        A("|---|---|---|---|---|---|")
        for m, v in models.items():
            A(f"| {m} | {v['cv_auc_mean']} | {v['cv_auc_std']} | {v['cv_accuracy_mean']} | {v['cv_f1_mean']} | {v['robust_score']} |")
        A("")

    # --- Recommended model ---
    A("## 6. Recommended ML Model\n")
    A(f"**Selected: `{model_comp['selected_model']}` predicting `{model_comp['selected_target']}`.** {model_comp['selection_rationale']}\n")
    oos = model_comp["out_of_sample_test_metrics"]
    A(f"True out-of-sample test performance (never touched during model selection): AUC={oos['oos_auc']}, accuracy={oos['oos_accuracy']} (vs. {oos['baseline_accuracy_majority_class']} majority-class baseline), F1={oos['oos_f1']}, n={oos['n_oos_bars']} bars.\n")
    A("Top feature importances:\n")
    for f, v in model_comp["top_feature_importance"].items():
        A(f"- {f}: {v}")
    A("")

    # --- Candidate strategies ---
    A("## 7. Candidate Trading Strategies\n")
    for s in strategies:
        A(f"### {s['name']} (`{s['id']}`)")
        A(f"- **Entry:** {s['entry_logic']}")
        A(f"- **Exit:** {s['exit_logic']}")
        A(f"- **Stop-loss:** {s['stop_loss']}")
        A(f"- **Take-profit:** {s['take_profit']}")
        A(f"- **Hedge ratio:** {s['hedge_ratio']['description']}")
        A(f"- **Position sizing:** {s['position_sizing']['description']}" + (f" {s['position_sizing'].get('conviction_scaling','')}" if 'conviction_scaling' in s['position_sizing'] else ""))
        A(f"- **Costs modeled:** {s['costs']['description']}\n")

    # --- Backtest results ---
    A("## 8. Backtest Results (out-of-sample period: " + backtest["oos_period"][0] + " to " + backtest["oos_period"][1] + f", {backtest['n_oos_bars']} bars)\n")
    A("No-lookahead design: signals use only bar-close information; fills occur at the *next* bar's open; every entry/exit pays real per-lot commission + tick-based slippage on both legs.\n")
    A("| Strategy | Trades | Win rate | Profit factor | Total return | Sharpe (ann.) | Max DD | Calmar |")
    A("|---|---|---|---|---|---|---|---|")
    for sid, m in backtest["results"].items():
        A(f"| {sid} | {m['n_trades']} | {m['win_rate']*100:.1f}% | {m['profit_factor']} | {m['total_return_pct']}% | {m['sharpe_ratio_annualized']} | {m['max_drawdown_pct']}% | {m['calmar_ratio']} |")
    A("")

    grid_results = {sid: m for sid, m in backtest["results"].items() if m.get("trades_by_max_lots")}
    if grid_results:
        A("Session-grid strategies only - how many completed cycles maxed out at each lot count (a cycle that never adds beyond lot 1 exited via take-profit or the session close before any add trigger fired; a cycle that reaches 3 lots either took profit from there or hit the staged stop-loss):\n")
        A("| Strategy | 1 lot | 2 lots | 3 lots |")
        A("|---|---|---|---|")
        for sid, m in grid_results.items():
            by_lots = m["trades_by_max_lots"]
            A(f"| {sid} | {by_lots.get('1', 0)} | {by_lots.get('2', 0)} | {by_lots.get('3', 0)} |")
        A("")

    # --- Evaluation ---
    A("## 9. Strategy Evaluation\n")
    A("| Strategy | Composite score | Rejected? | Reasons |")
    A("|---|---|---|---|")
    for sid, ev in evaluation["evaluation_by_strategy"].items():
        reasons = "; ".join(ev["rejection_reasons"]) if ev["rejection_reasons"] else "—"
        A(f"| {sid} | {ev['composite_score']} | {'Yes' if ev['rejected'] else 'No'} | {reasons} |")
    A("")

    # --- Final strategy ---
    A("## 10. Final Selected Strategy\n")
    if final_id:
        A(f"**{strategies_by_id[final_id]['name']}** (`{final_id}`)\n\n{evaluation['justification']}\n")
    else:
        A(f"**No strategy is recommended for live deployment on this dataset.**\n\n{evaluation['justification']}\n")
        if tentative_id:
            A(f"For reference, the most promising candidate for further R&D is `{tentative_id}`.\n")

    # --- Limitations ---
    data_span = _span_phrase(data_summary["aligned_date_range"][0], data_summary["aligned_date_range"][1])
    oos_span = _span_phrase(backtest["oos_period"][0], backtest["oos_period"][1])
    A("## 11. Key Limitations\n")
    A(f"- **Sample size:** a dataset spanning {data_span} yields {data_summary['aligned_rows']} aligned 1-minute bars; "
      f"the true out-of-sample test window spans {oos_span} ({backtest['n_oos_bars']} bars). Trade counts per strategy "
      "are still modest at this bar size, so backtest metrics carry real statistical uncertainty regardless of how good they look.")
    A(f"- **Real contract economics, no margin/FX modeled:** P&L uses the fixed HKEX {config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} "
      f"MHI:HHI lot ratio and each leg's real contract multiplier (HKD {config.MHI_CONTRACT_MULTIPLIER}/pt MHI, HKD {config.HHI_CONTRACT_MULTIPLIER}/pt HHI), "
      f"plus real per-lot commission (HKD {config.MHI_COMMISSION_PER_LOT} MHI / HKD {config.HHI_COMMISSION_PER_LOT} HHI, per side) and "
      f"{config.SLIPPAGE_TICKS}-tick slippage per leg per side — not a bps-of-notional proxy. Margin requirements are not modeled "
      "(no leverage/margin-call simulation), and both legs are HKD-denominated so no FX conversion applies for an HKD-based account.")
    A("- **Bar-level, not tick-level, execution:** stops/exits are evaluated on 1-minute bar closes and filled at the next bar's open; "
      "genuine intrabar stop-outs, partial fills, and market-impact/liquidity constraints at the moment of execution are not modeled.")
    A(f"- **Stationarity risk:** the cointegration relationship is assessed over a {data_span} span "
      f"({data_summary['aligned_date_range'][0]} to {data_summary['aligned_date_range'][1]}); it may shift with contract rolls, "
      "seasonality, or regime changes within or beyond that sample. The hedge ratio itself is a fixed contract spec, not an estimate, "
      "so it does not carry this risk, but the *validity* of trading it (stationarity, half-life) does - and this pipeline does not "
      "yet detect or adjust for contract rolls (MHI/HHI futures expire monthly) within a multi-month dataset.")
    A(f"- **Model/strategy count multiple-testing risk:** three targets, five models, and {n_strategies} strategies were compared; some of the apparent "
      "edge in any single combination could reflect selection among many candidates rather than genuine, persistent alpha — the sub-period "
      "stability check and trade-count screen in Section 9 are a partial safeguard, not a guarantee.")

    report = "\n".join(lines)
    config.FINAL_REPORT_PATH.write_text(report)
    return report
