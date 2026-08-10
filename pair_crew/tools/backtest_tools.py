"""Event-driven, no-lookahead backtest engine for the candidate pair-trading
strategies, with realistic transaction costs, slippage, and next-bar fills."""
import json

import numpy as np
import pandas as pd
from crewai.tools import tool

from pair_crew import config
from pair_crew.tools import dsl_tools

MIN_VOL = config.MIN_VOL_FLOOR


def _signal(strategy_id: str, p: dict, row: pd.Series, position: int, bars_held: int):
    """Return 'enter_long' | 'enter_short' | 'exit' | None using ONLY values
    already known at this bar's close (row). The caller executes the action
    at the *next* bar's open, so this never looks ahead."""
    z = row["spread_zscore"]
    if strategy_id == "ta_zscore_reversion":
        if position == 0:
            if z > p["z_entry"]:
                return "enter_short"
            if z < -p["z_entry"]:
                return "enter_long"
            return None
        if abs(z) <= p["z_exit"] or abs(z) >= p["z_stop"] or bars_held >= p["max_holding_bars"]:
            return "exit"
        return None

    if strategy_id == "ml_filtered_reversion":
        ml = row["ml_score"]
        if position == 0:
            if z > p["z_entry"] and ml < p["ml_confirm_short_below"]:
                return "enter_short"
            if z < -p["z_entry"] and ml > p["ml_confirm_long_above"]:
                return "enter_long"
            return None
        if abs(z) <= p["z_exit"] or abs(z) >= p["z_stop"] or bars_held >= p["max_holding_bars"]:
            return "exit"
        return None

    if strategy_id == "ml_directional":
        ml = row["ml_score"]
        ma_cross = row["spread_ma_crossover"]
        if position == 0:
            if ml > p["prob_upper"] and ma_cross > 0:
                return "enter_long"
            if ml < p["prob_lower"] and ma_cross < 0:
                return "enter_short"
            return None
        if position == 1 and ml < p["prob_exit"]:
            return "exit"
        if position == -1 and ml > (1 - p["prob_exit"]):
            return "exit"
        if abs(z) >= p["z_stop"] or bars_held >= p["max_holding_bars"]:
            return "exit"
        return None

    if strategy_id == "corr_regime_reversion":
        rsi = row["spread_rsi"]
        corr = row["rolling_corr_returns"]
        if position == 0:
            if z > p["z_entry"] and rsi > p["rsi_overbought"] and corr > p["min_corr_entry"]:
                return "enter_short"
            if z < -p["z_entry"] and rsi < p["rsi_oversold"] and corr > p["min_corr_entry"]:
                return "enter_long"
            return None
        if corr < p["min_corr_exit"]:
            return "exit"
        if abs(z) <= p["z_exit"] or abs(z) >= p["z_stop"] or bars_held >= p["max_holding_bars"]:
            return "exit"
        return None

    raise ValueError(f"unknown strategy id {strategy_id}")


def _build_dsl_signal_fn(strategy: dict):
    """Build a signal_fn (see _run_one_backtest) for a chat-frontend
    strategy with engine="dsl_expression", whose entry/exit/stop rules are
    DSL expression strings (params["entry_long_expr"] / "entry_short_expr"
    / "exit_expr" / "stop_expr") rather than one of the hardcoded strategy
    ids above. Expressions are compiled once here (not re-parsed per bar)
    via dsl_tools' whitelist AST interpreter -- no eval()/exec() involved."""
    compiled = dsl_tools.compile_strategy_params(strategy["params"])

    def signal_fn(row: pd.Series, position: int, bars_held: int):
        env = dsl_tools.row_env(row)
        if position == 0:
            if compiled.entry_long is not None and dsl_tools.eval_expr(compiled.entry_long, env):
                return "enter_long"
            if compiled.entry_short is not None and dsl_tools.eval_expr(compiled.entry_short, env):
                return "enter_short"
            return None
        if compiled.stop is not None and dsl_tools.eval_expr(compiled.stop, env):
            return "exit"
        if dsl_tools.eval_expr(compiled.exit, env):
            return "exit"
        if compiled.max_holding_bars and bars_held >= compiled.max_holding_bars:
            return "exit"
        return None

    return signal_fn


def _run_one_backtest(strategy: dict, bt_df: pd.DataFrame, signal_fn=None) -> dict:
    sid = strategy["id"]
    p = strategy["params"]
    conviction_scaling = p.get("conviction_scaling", False)

    equity = config.INITIAL_CAPITAL
    equity_curve = []
    position = 0
    n_units = 0.0
    last_mark_spread = None
    entry_bar_i = None
    pending_action = None
    pending_ml_score = None
    trade_log = []

    n = len(bt_df)
    total_cost_paid = 0.0

    for i in range(n):
        row = bt_df.iloc[i]

        # 1. execute any action decided at the previous bar's close, at THIS bar's open
        if pending_action is not None:
            open_mhi = row["MHI_Open"]
            open_hhi = row["HHI_Open"]
            open_spread = config.spread_dollar_value(open_mhi, open_hhi)
            if pending_action in ("enter_long", "enter_short") and position == 0:
                direction = 1 if pending_action == "enter_long" else -1
                vol = max(prev_row["spread_vol"], MIN_VOL)  # $ vol of ONE spread unit
                unit_notional = config.spread_unit_notional(open_mhi, open_hhi)
                units_by_risk = equity * config.RISK_FRACTION / vol
                units_by_leverage = equity * config.MAX_LEVERAGE / unit_notional
                sized_units = min(units_by_risk, units_by_leverage)
                if conviction_scaling and pending_ml_score is not None:
                    conviction = min(2 * abs(pending_ml_score - 0.5), 1.0)
                    sized_units *= max(conviction, 0.1)
                cost = config.spread_transaction_cost(sized_units)
                equity -= cost
                total_cost_paid += cost
                position = direction
                n_units = sized_units
                last_mark_spread = open_spread
                entry_bar_i = i
            elif pending_action == "exit" and position != 0:
                pnl = n_units * position * (open_spread - last_mark_spread)
                equity += pnl
                cost = config.spread_transaction_cost(n_units)
                equity -= cost
                total_cost_paid += cost
                trade_log.append(
                    {
                        "entry_bar": entry_bar_i,
                        "exit_bar": i,
                        "direction": "long_spread" if position == 1 else "short_spread",
                        "holding_bars": i - entry_bar_i,
                        "gross_pnl_last_leg": round(float(pnl), 2),
                        "exit_cost": round(float(cost), 2),
                    }
                )
                position = 0
                n_units = 0.0
                last_mark_spread = None
                entry_bar_i = None
            pending_action = None
            pending_ml_score = None

        # 2. mark remaining open position to this bar's close
        if position != 0:
            close_spread = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
            pnl = n_units * position * (close_spread - last_mark_spread)
            equity += pnl
            last_mark_spread = close_spread

        equity_curve.append(equity)

        # 3. decide the action for the NEXT bar using only this bar's close-based signals
        if i < n - 1:
            bars_held = (i - entry_bar_i) if position != 0 else 0
            action = (signal_fn or (lambda r, pos, bh: _signal(sid, p, r, pos, bh)))(row, position, bars_held)
            if action is not None:
                pending_action = action
                pending_ml_score = row.get("ml_score")
        prev_row = row

    # force-close any still-open position at the final bar's close (mark already applied above)
    if position != 0:
        trade_log.append(
            {
                "entry_bar": entry_bar_i,
                "exit_bar": n - 1,
                "direction": "long_spread" if position == 1 else "short_spread",
                "holding_bars": (n - 1) - entry_bar_i,
                "note": "open at end of backtest window, marked-to-market but not charged exit cost",
            }
        )

    return _summarize_backtest(sid, bt_df.index, equity_curve, trade_log, total_cost_paid)


def _summarize_backtest(
    sid: str, index: pd.DatetimeIndex, equity_curve: list, trade_log: list, total_cost_paid: float
) -> tuple[dict, pd.Series]:
    """Shared metrics tail for every backtest engine: turns a raw equity
    curve + trade log into the standard metrics dict (Sharpe, drawdown,
    Calmar, win rate, profit factor, ...) used by backtest_all_strategies
    and evaluate_strategies alike, regardless of which engine produced it."""
    n = len(index)
    equity_curve = pd.Series(equity_curve, index=index)
    bar_returns = equity_curve.pct_change().dropna()

    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    bars_per_year = n / max(span_days, 1) * 365.25

    sharpe = (
        float(bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year))
        if bar_returns.std() > 0
        else 0.0
    )
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    # each trade's total pnl = equity change from just before entry to the exit bar
    trade_pnls = []
    for t in trade_log:
        if "note" in t:
            continue
        entry_i, exit_i = t["entry_bar"], t["exit_bar"]
        pnl = float(equity_curve.iloc[exit_i] - (equity_curve.iloc[entry_i - 1] if entry_i > 0 else config.INITIAL_CAPITAL))
        trade_pnls.append(pnl)

    n_trades = len(trade_pnls)
    wins = [x for x in trade_pnls if x > 0]
    losses = [x for x in trade_pnls if x < 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    total_return_pct = (equity_curve.iloc[-1] / config.INITIAL_CAPITAL - 1) * 100
    avg_holding_bars = float(np.mean([t["holding_bars"] for t in trade_log])) if trade_log else 0.0
    calmar = (total_return_pct / 100) / abs(max_dd) if max_dd < 0 else float("inf")

    # session-grid engines only: how many completed cycles maxed out at 1, 2, or 3 lots
    # (trade_log["lots"] is the lot count at exit, which is monotonically non-decreasing
    # during a cycle, so it equals the peak lot count that cycle ever reached)
    lots_reached = [t["lots"] for t in trade_log if "lots" in t]
    trades_by_max_lots = None
    if lots_reached:
        trades_by_max_lots = {
            str(k): lots_reached.count(k) for k in sorted(set(lots_reached))
        }

    metrics = {
        "strategy_id": sid,
        "n_bars_backtested": n,
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else None,
        "total_return_pct": round(float(total_return_pct), 3),
        "final_equity": round(float(equity_curve.iloc[-1]), 2),
        "sharpe_ratio_annualized": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "calmar_ratio": round(calmar, 3) if np.isfinite(calmar) else None,
        "avg_holding_bars": round(avg_holding_bars, 2),
        "total_transaction_costs": round(float(total_cost_paid), 2),
        "avg_trade_pnl": round(float(np.mean(trade_pnls)), 2) if trade_pnls else 0.0,
    }
    if trades_by_max_lots is not None:
        metrics["trades_by_max_lots"] = trades_by_max_lots
    return metrics, equity_curve


def _run_grid_backtest(strategy: dict, bt_df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Session-anchored, multi-lot grid/fade backtest for the
    afternoon_grid_fade strategy (engine="session_grid"). Unlike
    _run_one_backtest (single all-or-nothing position, vol-scaled size),
    this engine tracks a per-day 1pm-anchored ladder of up to
    params['max_lots'] fixed-size lots and an average-entry-price
    take-profit, force-flattening every day at params['day_session_end_utc_hour']
    -- the 3 built-in strategies in strategy_tools.py set this to the day-session
    close so nothing carries into the night session, but a chat-frontend grid
    strategy may set it later (e.g. NIGHT_SESSION_END_UTC_HOUR) to deliberately
    carry a position through the day session and into the night session.

    Stop-loss mode is pluggable via params["stop_mode"]:
      - "ticks_beyond_max" (default, used by the 3 built-in strategies in
        strategy_tools.py): only arms once max_lots are on, closing the
        whole position if price extends stop_ticks_beyond_max past the
        last lot's trigger level.
      - "dollar_loss_from_avg_entry" (chat-frontend grid strategies): closes
        the whole position the moment its unrealized $ loss vs the average
        entry price (across however many lots are currently on) reaches
        params["stop_loss_dollars"] -- checked at every lot count, not only
        once max_lots is reached.
      - "ticks_from_avg_entry" (chat-frontend grid strategies): closes the
        whole position once price has moved params["stop_loss_ticks"]
        against the average entry price (on the same tick-price series the
        entry/add/take-profit triggers use) -- the tick-denominated
        equivalent of "dollar_loss_from_avg_entry", also checked at every
        lot count.

    No-lookahead discipline matches _run_one_backtest: every entry/add/stop
    decided from a bar's CLOSE is filled at the *next* bar's OPEN. The one
    deliberate exception is the daily force-flatten, which fills at that
    day's own last bar's CLOSE (there is no "next bar" left within the same
    day to fill at) - a deviation from the next-open convention, made
    necessary by the end-of-day boundary itself, not a lookahead shortcut.
    """
    sid = strategy["id"]
    p = strategy["params"]
    entry_ticks = p["entry_ticks"]
    add_ticks = p["add_ticks"]
    max_lots = p["max_lots"]
    stop_mode = p.get("stop_mode", "ticks_beyond_max")
    stop_ticks_beyond_max = p.get("stop_ticks_beyond_max")
    stop_loss_dollars = p.get("stop_loss_dollars")
    stop_loss_ticks = p.get("stop_loss_ticks")
    tp_ticks = p["take_profit_ticks"]
    anchor_hour = p["anchor_utc_hour"]
    session_end_hour = p["day_session_end_utc_hour"]

    df = bt_df.copy()
    df["spread_ui_close"] = config.spread_ui_price(df["MHI_Close"], df["HHI_Close"])
    df["spread_ui_open"] = config.spread_ui_price(df["MHI_Open"], df["HHI_Open"])
    hour_of_day = df.index.hour + df.index.minute / 60.0
    df["date"] = df.index.date
    df["in_afternoon"] = (hour_of_day >= anchor_hour) & (hour_of_day < session_end_hour)

    afternoon = df[df["in_afternoon"]]
    anchor_by_date = afternoon.groupby("date")["spread_ui_open"].first()
    df["anchor_price"] = df["date"].map(anchor_by_date)
    last_afternoon_idx = set(afternoon.groupby("date").apply(lambda g: g.index[-1]))
    df["is_session_end"] = df.index.isin(last_afternoon_idx)

    equity = config.INITIAL_CAPITAL
    equity_curve = []
    trade_log = []
    total_cost_paid = 0.0

    lots = 0
    direction = 0          # +1 long spread, -1 short spread (fixed once lot 1 fills)
    breakout_sign = 0      # sign of the initial move away from the anchor that triggered lot 1
    lot_fill_ui_prices: list[float] = []
    lot_fill_dollar_prices: list[float] = []
    total_units = 0.0
    last_mark_dollar = None
    entry_bar_i = None
    current_date = None
    cycle_done_today = False

    pending_action = None   # "enter" / "add" / "exit_tp" / "exit_sl"
    pending_direction = None
    pending_breakout_sign = None

    n = len(df)

    for i in range(n):
        row = df.iloc[i]
        if row["date"] != current_date:
            current_date = row["date"]
            cycle_done_today = False

        # 1. execute any action decided at the previous bar's close, at THIS bar's open
        if pending_action is not None:
            open_dollar = config.spread_dollar_value(row["MHI_Open"], row["HHI_Open"])
            open_ui = row["spread_ui_open"]
            if pending_action in ("enter", "add"):
                if lots > 0:
                    equity += total_units * direction * (open_dollar - last_mark_dollar)
                cost = config.spread_transaction_cost(1.0)
                equity -= cost
                total_cost_paid += cost
                if lots == 0:
                    direction = pending_direction
                    breakout_sign = pending_breakout_sign
                    entry_bar_i = i
                lots += 1
                total_units = float(lots)
                lot_fill_ui_prices.append(open_ui)
                lot_fill_dollar_prices.append(open_dollar)
                last_mark_dollar = open_dollar
            elif pending_action in ("exit_tp", "exit_sl"):
                pnl = total_units * direction * (open_dollar - last_mark_dollar)
                equity += pnl
                cost = config.spread_transaction_cost(total_units)
                equity -= cost
                total_cost_paid += cost
                trade_log.append(
                    {
                        "entry_bar": entry_bar_i,
                        "exit_bar": i,
                        "direction": "long_spread" if direction == 1 else "short_spread",
                        "holding_bars": i - entry_bar_i,
                        "lots": lots,
                        "exit_reason": pending_action,
                        "gross_pnl_last_leg": round(float(pnl), 2),
                    }
                )
                lots = 0
                direction = 0
                breakout_sign = 0
                lot_fill_ui_prices = []
                lot_fill_dollar_prices = []
                total_units = 0.0
                last_mark_dollar = None
                entry_bar_i = None
                cycle_done_today = True
            pending_action = None
            pending_direction = None
            pending_breakout_sign = None

        # 1b. daily force-flatten at the day session's own last bar, at THIS bar's CLOSE
        # (no next bar remains within the same day to fill an exit at)
        if row["is_session_end"] and lots > 0:
            close_dollar = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
            pnl = total_units * direction * (close_dollar - last_mark_dollar)
            equity += pnl
            cost = config.spread_transaction_cost(total_units)
            equity -= cost
            total_cost_paid += cost
            trade_log.append(
                {
                    "entry_bar": entry_bar_i,
                    "exit_bar": i,
                    "direction": "long_spread" if direction == 1 else "short_spread",
                    "holding_bars": i - entry_bar_i,
                    "lots": lots,
                    "exit_reason": "day_session_close_flatten",
                    "gross_pnl_last_leg": round(float(pnl), 2),
                }
            )
            lots = 0
            direction = 0
            breakout_sign = 0
            lot_fill_ui_prices = []
            lot_fill_dollar_prices = []
            total_units = 0.0
            last_mark_dollar = None
            entry_bar_i = None
            cycle_done_today = True

        # 2. mark remaining open position to this bar's close
        if lots > 0:
            close_dollar = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
            equity += total_units * direction * (close_dollar - last_mark_dollar)
            last_mark_dollar = close_dollar

        equity_curve.append(equity)

        # 3. decide the action for the NEXT bar using only this bar's close-based
        # signal - skipped on the session-end bar itself (nothing left to fill same-day)
        if i < n - 1 and row["in_afternoon"] and not row["is_session_end"] and pd.notna(row["anchor_price"]):
            anchor = row["anchor_price"]
            dist = row["spread_ui_close"] - anchor
            if lots == 0:
                if not cycle_done_today and abs(dist) >= entry_ticks:
                    pending_action = "enter"
                    pending_breakout_sign = 1 if dist > 0 else -1
                    pending_direction = -pending_breakout_sign  # fade the move
            else:
                breakout_extension = breakout_sign * dist
                if lots < max_lots and breakout_extension >= entry_ticks + add_ticks * lots:
                    pending_action = "add"
                elif stop_mode == "dollar_loss_from_avg_entry":
                    avg_dollar = float(np.mean(lot_fill_dollar_prices))
                    close_dollar_now = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
                    unrealized = total_units * direction * (close_dollar_now - avg_dollar)
                    if unrealized <= -stop_loss_dollars:
                        pending_action = "exit_sl"
                elif stop_mode == "ticks_from_avg_entry":
                    avg_ui = float(np.mean(lot_fill_ui_prices))
                    adverse_move = -direction * (row["spread_ui_close"] - avg_ui)
                    if adverse_move >= stop_loss_ticks:
                        pending_action = "exit_sl"
                elif lots >= max_lots:
                    stop_level = entry_ticks + add_ticks * (max_lots - 1) + stop_ticks_beyond_max
                    if breakout_extension >= stop_level:
                        pending_action = "exit_sl"
                if pending_action is None:
                    avg_ui = float(np.mean(lot_fill_ui_prices))
                    favorable_move = direction * (row["spread_ui_close"] - avg_ui)
                    if favorable_move >= tp_ticks:
                        pending_action = "exit_tp"

    return _summarize_backtest(sid, df.index, equity_curve, trade_log, total_cost_paid)


def run_backtest_for_strategy(strategy: dict, bt_df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Dispatch to the correct backtest engine for a strategy spec.
    engine="session_grid" (afternoon_grid_fade) uses the multi-lot,
    session-anchored grid engine; engine="dsl_expression" (chat frontend)
    uses the single-position, vol-scaled engine with a DSL-expression signal
    function; every other strategy uses that same engine with its hardcoded
    _signal() logic."""
    engine = strategy.get("engine")
    if engine == "session_grid":
        return _run_grid_backtest(strategy, bt_df)
    if engine == "dsl_expression":
        return _run_one_backtest(strategy, bt_df, signal_fn=_build_dsl_signal_fn(strategy))
    return _run_one_backtest(strategy, bt_df)


def build_backtest_dataframe() -> pd.DataFrame:
    """Shared helper: assemble the out-of-sample signal+price dataframe used
    by both the backtester and the evaluator's stability checks."""
    feats = pd.read_parquet(config.FEATURES_PATH)
    aligned = pd.read_parquet(config.ALIGNED_DATA_PATH)
    preds = pd.read_parquet(config.PREDICTIONS_PATH)

    n_train = int(len(feats) * config.TRAIN_FRACTION)
    oos_index = feats.index[n_train:]

    bt_df = feats.loc[oos_index, [
        "spread_zscore", "spread_rsi", "spread_ma_crossover", "rolling_corr_returns",
        "hedge_ratio", "spread_vol",
    ]].copy()
    bt_df = bt_df.join(
        aligned.loc[oos_index, ["MHI_Open", "MHI_Close", "HHI_Open", "HHI_Close"]]
    )
    bt_df["ml_score"] = preds["ml_score"].reindex(oos_index)
    return bt_df.dropna()


@tool("backtest_all_strategies")
def backtest_all_strategies() -> str:
    """
    Backtest every strategy in outputs/strategies.json over the true
    out-of-sample period (the ~35% of history never used for ML training or
    model selection), using outputs/features.parquet for signals and
    outputs/aligned_data.parquet for OHLC execution prices, joined with the
    ML model's out-of-sample probability from outputs/predictions.parquet.

    Realism controls: signals are computed only from information available
    at a bar's close; the resulting order is filled at the *next* bar's
    open (never the same bar), so there is no lookahead bias. Every entry
    and exit charges real per-lot commission (config.MHI_COMMISSION_PER_LOT /
    HHI_COMMISSION_PER_LOT) plus tick-based slippage on both legs (see
    config.spread_transaction_cost). Position size is volatility-scaled per
    each strategy's position-sizing rule from outputs/strategies.json.

    Computes, per strategy: number of trades, win rate, profit factor,
    total return, annualized Sharpe ratio, max drawdown, Calmar ratio,
    average holding period, and total transaction costs paid. Equity curves
    are saved to outputs/equity_<strategy_id>.csv and all metrics are saved
    to outputs/backtest_results.json.

    Returns a JSON string with the full metrics table for all strategies.
    """
    strategies = json.loads(config.STRATEGIES_PATH.read_text())
    bt_df = build_backtest_dataframe()

    all_results = {}
    for strategy in strategies:
        metrics, equity_curve = run_backtest_for_strategy(strategy, bt_df)
        all_results[strategy["id"]] = metrics
        equity_curve.to_frame("equity").to_csv(config.OUTPUT_DIR / f"equity_{strategy['id']}.csv")

    summary = {
        "oos_period": [str(bt_df.index.min()), str(bt_df.index.max())],
        "n_oos_bars": len(bt_df),
        "initial_capital": config.INITIAL_CAPITAL,
        "cost_assumptions": {
            "mhi_commission_per_lot_per_side": config.MHI_COMMISSION_PER_LOT,
            "hhi_commission_per_lot_per_side": config.HHI_COMMISSION_PER_LOT,
            "slippage_ticks_per_leg_per_side": config.SLIPPAGE_TICKS,
        },
        "results": all_results,
    }
    config.BACKTEST_RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)
