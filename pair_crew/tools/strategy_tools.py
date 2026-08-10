"""Supervisor Agent tool: turn the ML Agent's output into several concrete,
executable pair-trading strategies that combine the ML signal with technical
analysis rules."""
import json

from crewai.tools import tool

from pair_crew import config


@tool("build_candidate_strategies")
def build_candidate_strategies() -> str:
    """
    Build several concrete, executable pair-trading strategy specifications
    that combine the ML Agent's selected model/target (outputs/best_model_meta.json)
    with technical-analysis rules on the engineered features (z-score,
    RSI-on-spread, moving-average crossover, rolling correlation).

    Each strategy fully specifies: entry logic, exit logic, hedge-ratio
    assumption, stop-loss, take-profit/time-stop, and a volatility-scaled
    position-sizing rule, so it can be mechanically backtested with no
    discretionary judgment calls.

    Produces seven strategies of increasing sophistication:
      1. ta_zscore_reversion - pure technical z-score mean reversion (baseline,
         no ML) so ML's marginal value can be measured.
      2. ml_filtered_reversion - the same z-score entries, but only taken when
         the ML model's predicted spread-direction probability confirms the
         reversion call (reduces false signals).
      3. ml_directional - trades directly on high-conviction ML probability
         confirmed by a moving-average trend filter, with position size
         scaled by ML conviction.
      4. corr_regime_reversion - z-score + RSI entries gated by a minimum
         rolling correlation (only trade when the pair relationship looks
         intact) with an early protective exit if correlation breaks down.
      5. afternoon_grid_fade - session-anchored intraday grid/fade (no ML,
         no z-score): anchored to each day's 1pm HKT reopen price, fades a
         200-tick move away from that anchor, adds a lot every further
         50-tick adverse extension up to 3 lots, take-profits 50 ticks in
         favor of the average entry price, and stops the whole position out
         if price extends 50 ticks beyond the 3rd lot. Force-flattened at
         the 4:30pm HKT session close every day.
      6. full_day_grid_fade - identical grid/fade mechanics to
         afternoon_grid_fade, but anchored to the 9:15am HKT day-session
         open instead of the 1pm reopen, and an open position is carried
         through the lunch break into the afternoon (not flattened/re-
         anchored at 1pm) - only stop-loss, take-profit, or the 4:30pm
         session close end it.
      7. morning_grid_fade - identical grid/fade mechanics to
         afternoon_grid_fade, but confined to the morning session:
         anchored to the 9:15am HKT open and force-flattened at the 12:30
         HKT lunch break instead of the 4:30pm close (only if the stop-loss
         hasn't already ended it first).
      8. mid_morning_full_day_grid_fade - identical to full_day_grid_fade,
         but anchored to the 9:30am HKT price (15 minutes into the day
         session) instead of the 9:15am open.
      9. mid_morning_grid_fade - identical to morning_grid_fade, but
         anchored to the 9:30am HKT price instead of the 9:15am open.
    Saves the specifications to outputs/strategies.json and returns them as
    a JSON string.
    """
    meta = json.loads(config.BEST_MODEL_META_PATH.read_text())

    common_costs = {
        "mhi_commission_per_lot_per_side": config.MHI_COMMISSION_PER_LOT,
        "hhi_commission_per_lot_per_side": config.HHI_COMMISSION_PER_LOT,
        "slippage_ticks_per_leg_per_side": config.SLIPPAGE_TICKS,
        "tick_size": config.TICK_SIZE,
        "description": (
            f"Real exchange/broker commission per lot, per side: HKD "
            f"{config.MHI_COMMISSION_PER_LOT} (MHI), HKD {config.HHI_COMMISSION_PER_LOT} "
            f"(HHI). Slippage assumed at {config.SLIPPAGE_TICKS} tick per leg, per side "
            f"({config.TICK_SIZE} index point), converted to $ via each leg's own "
            "contract multiplier - not a bps-of-notional proxy."
        ),
    }
    common_hedge = {
        "source": "fixed_hkex_contract_ratio",
        "description": (
            f"Hedge ratio is the fixed HKEX spread contract ratio: {config.MHI_HEDGE_LOTS} "
            f"lots of MHI (multiplier {config.MHI_CONTRACT_MULTIPLIER}) against "
            f"{config.HHI_HEDGE_LOTS} lot(s) of HHI (multiplier {config.HHI_CONTRACT_MULTIPLIER}) "
            "on the opposite side, per traded unit. This is the exchange-defined contract "
            "spec, not a statistical estimate, so it is constant for every trade and every "
            "bar - there is no rolling re-estimation and no lookahead risk from it."
        ),
    }
    common_sizing = {
        "method": "inverse_volatility_target",
        "risk_fraction_of_equity": config.RISK_FRACTION,
        "max_leverage": config.MAX_LEVERAGE,
        "description": (
            "Number of spread units = min(equity * risk_fraction / spread_$_volatility, "
            "equity * max_leverage / unit_gross_notional), where a 'unit' is one "
            f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot pair at "
            "their contract multipliers, so expected per-bar $ P&L volatility is "
            "targeted at roughly risk_fraction of equity regardless of current spread "
            "volatility, capped by max leverage on gross notional."
        ),
    }

    strategies = [
        {
            "id": "ta_zscore_reversion",
            "name": "Technical Z-Score Mean Reversion (baseline, no ML)",
            "uses_ml": False,
            "entry_logic": (
                f"Flat -> short spread when spread_zscore > {config.ZSCORE_ENTRY}; "
                f"flat -> long spread when spread_zscore < -{config.ZSCORE_ENTRY}."
            ),
            "exit_logic": (
                f"Take-profit: close when |spread_zscore| <= {config.ZSCORE_EXIT} (reversion "
                f"achieved). Time-stop: close after {config.MAX_HOLDING_BARS} bars regardless."
            ),
            "stop_loss": f"Close if |spread_zscore| >= {config.ZSCORE_STOP} (divergence continuing against the position).",
            "take_profit": f"|spread_zscore| <= {config.ZSCORE_EXIT}",
            "hedge_ratio": common_hedge,
            "position_sizing": common_sizing,
            "costs": common_costs,
            "params": {
                "z_entry": config.ZSCORE_ENTRY,
                "z_exit": config.ZSCORE_EXIT,
                "z_stop": config.ZSCORE_STOP,
                "max_holding_bars": config.MAX_HOLDING_BARS,
            },
        },
        {
            "id": "ml_filtered_reversion",
            "name": "ML-Filtered Mean Reversion",
            "uses_ml": True,
            "ml_model": meta["model_name"],
            "ml_target": meta["target"],
            "entry_logic": (
                f"Same z-score triggers as the baseline (|z| > {config.ZSCORE_ENTRY}), but only "
                "taken when the ML model's predicted probability of the spread rising "
                "(ml_score) confirms the reversion direction: short spread requires "
                "ml_score < 0.45 (model agrees spread is more likely to fall); long spread "
                "requires ml_score > 0.55 (model agrees spread is more likely to rise)."
            ),
            "exit_logic": (
                f"Same as baseline: take-profit at |spread_zscore| <= {config.ZSCORE_EXIT}, "
                f"time-stop after {config.MAX_HOLDING_BARS} bars."
            ),
            "stop_loss": f"Close if |spread_zscore| >= {config.ZSCORE_STOP}.",
            "take_profit": f"|spread_zscore| <= {config.ZSCORE_EXIT}",
            "hedge_ratio": common_hedge,
            "position_sizing": common_sizing,
            "costs": common_costs,
            "params": {
                "z_entry": config.ZSCORE_ENTRY,
                "z_exit": config.ZSCORE_EXIT,
                "z_stop": config.ZSCORE_STOP,
                "ml_confirm_short_below": 0.45,
                "ml_confirm_long_above": 0.55,
                "max_holding_bars": config.MAX_HOLDING_BARS,
            },
        },
        {
            "id": "ml_directional",
            "name": "ML-Directional Spread Trading with Trend Confirmation",
            "uses_ml": True,
            "ml_model": meta["model_name"],
            "ml_target": meta["target"],
            "entry_logic": (
                "Flat -> long spread when ml_score > 0.60 AND spread_ma_crossover > 0 "
                "(short-term spread MA above long-term MA confirms an emerging uptrend); "
                "flat -> short spread when ml_score < 0.40 AND spread_ma_crossover < 0."
            ),
            "exit_logic": (
                "Close a long when ml_score drops back below 0.50 (conviction lost); close "
                "a short when ml_score rises back above 0.50. "
                f"Time-stop after {config.MAX_HOLDING_BARS} bars."
            ),
            "stop_loss": f"Close if |spread_zscore| >= {config.ZSCORE_STOP} regardless of ML score.",
            "take_profit": "Conviction-fade exit above (ml_score crossing back through 0.50).",
            "hedge_ratio": common_hedge,
            "position_sizing": {
                **common_sizing,
                "conviction_scaling": (
                    "Additionally multiplies notional by 2*|ml_score - 0.5| (0 to 1), so size "
                    "grows with model confidence, capped by max_leverage."
                ),
            },
            "costs": common_costs,
            "params": {
                "prob_upper": 0.60,
                "prob_lower": 0.40,
                "prob_exit": 0.50,
                "z_stop": config.ZSCORE_STOP,
                "max_holding_bars": config.MAX_HOLDING_BARS,
                "conviction_scaling": True,
            },
        },
        {
            "id": "corr_regime_reversion",
            "name": "Correlation-Regime-Filtered Mean Reversion",
            "uses_ml": True,
            "ml_model": meta["model_name"],
            "ml_target": meta["target"],
            "entry_logic": (
                f"Short spread when spread_zscore > {config.ZSCORE_ENTRY} AND spread_rsi > 70 "
                f"(overbought spread) AND rolling_corr_returns > 0.2 (relationship still intact). "
                f"Long spread when spread_zscore < -{config.ZSCORE_ENTRY} AND spread_rsi < 30 AND "
                "rolling_corr_returns > 0.2. ML score is used only for position sizing "
                "(see position_sizing), not as an entry gate."
            ),
            "exit_logic": (
                f"Take-profit at |spread_zscore| <= {config.ZSCORE_EXIT}. Protective regime exit: "
                "close immediately if rolling_corr_returns drops below 0.0 (pair relationship "
                f"breaking down). Time-stop after {config.MAX_HOLDING_BARS} bars."
            ),
            "stop_loss": f"Close if |spread_zscore| >= {config.ZSCORE_STOP}, or the correlation-breakdown protective exit above.",
            "take_profit": f"|spread_zscore| <= {config.ZSCORE_EXIT}",
            "hedge_ratio": common_hedge,
            "position_sizing": {
                **common_sizing,
                "conviction_scaling": (
                    "Multiplies notional by 2*|ml_score - 0.5| (0 to 1) as a secondary "
                    "confidence overlay on top of the TA-driven entry."
                ),
            },
            "costs": common_costs,
            "params": {
                "z_entry": config.ZSCORE_ENTRY,
                "z_exit": config.ZSCORE_EXIT,
                "z_stop": config.ZSCORE_STOP,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "min_corr_entry": 0.2,
                "min_corr_exit": 0.0,
                "max_holding_bars": config.MAX_HOLDING_BARS,
                "conviction_scaling": True,
            },
        },
        {
            "id": "afternoon_grid_fade",
            "name": "Afternoon Session Grid Fade (1pm-anchored, no ML)",
            "uses_ml": False,
            "engine": "session_grid",
            "entry_logic": (
                f"Each day, anchor = the spread's platform-display price (see hedge_ratio "
                f"note) at the 13:00 HKT reopen. Flat -> enter 1 lot FADING the first "
                f"{config.GRID_ENTRY_TICKS}-tick move away from the anchor in either "
                f"direction (spread rose to anchor+{config.GRID_ENTRY_TICKS} -> short 1 lot "
                f"expecting reversion; spread fell to anchor-{config.GRID_ENTRY_TICKS} -> long "
                "1 lot expecting reversion). At most one such grid cycle is started per day."
            ),
            "exit_logic": (
                f"Adds: while below {config.GRID_MAX_LOTS} lots, add 1 more lot every "
                f"further {config.GRID_ADD_TICKS}-tick extension AGAINST the position (i.e. "
                "the breakout continuing past the anchor in its original direction) - lot 2 "
                f"at anchor +/-{config.GRID_ENTRY_TICKS + config.GRID_ADD_TICKS} ticks, lot 3 "
                f"at anchor +/-{config.GRID_ENTRY_TICKS + 2 * config.GRID_ADD_TICKS} ticks. Take-profit: close "
                f"all lots when price moves {config.GRID_TAKE_PROFIT_TICKS} ticks in favor "
                "of the current average entry price (checked at every lot count, 1 to 3). "
                "Force-flattened at the 16:30 HKT day-session close every day regardless of "
                "P&L (never carried into the night session or to the next day)."
            ),
            "stop_loss": (
                f"Only arms once all {config.GRID_MAX_LOTS} lots are on. Closes the entire "
                f"position if price extends a further {config.GRID_STOP_TICKS_BEYOND_MAX} "
                f"ticks beyond the 3rd lot's trigger level (i.e. anchor +/-"
                f"{config.GRID_ENTRY_TICKS + config.GRID_ADD_TICKS * (config.GRID_MAX_LOTS - 1) + config.GRID_STOP_TICKS_BEYOND_MAX} "
                "ticks) - a fixed, bounded max loss regardless of how the average entry "
                "price has moved as lots were added."
            ),
            "take_profit": f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor of the average entry price across all currently-held lots.",
            "hedge_ratio": {
                **common_hedge,
                "tick_reference": (
                    "Tick-distance triggers (entry/add/stop) are measured on the trading "
                    "platform's own displayed spread price (config.spread_ui_price, using "
                    "MHI_SPREAD_UI_MULTIPLIER / HHI_SPREAD_UI_MULTIPLIER), matching how a "
                    "discretionary trader reads the platform - NOT the $ P&L spread series "
                    "used for backtest accounting (that remains spread_dollar_value)."
                ),
            },
            "position_sizing": {
                "method": "fixed_lots_per_grid_rung",
                "description": (
                    "1 spread unit (the fixed "
                    f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot ratio) per "
                    f"grid rung, up to {config.GRID_MAX_LOTS} units total - NOT "
                    "volatility-scaled like the other strategies, since this strategy's own "
                    "risk control is the lot cap plus the staged stop-loss."
                ),
            },
            "costs": common_costs,
            "params": {
                "anchor_utc_hour": config.AFTERNOON_ANCHOR_UTC_HOUR,
                "day_session_end_utc_hour": config.DAY_SESSION_END_UTC_HOUR,
                "entry_ticks": config.GRID_ENTRY_TICKS,
                "add_ticks": config.GRID_ADD_TICKS,
                "max_lots": config.GRID_MAX_LOTS,
                "stop_ticks_beyond_max": config.GRID_STOP_TICKS_BEYOND_MAX,
                "take_profit_ticks": config.GRID_TAKE_PROFIT_TICKS,
            },
        },
        {
            "id": "full_day_grid_fade",
            "name": "Full-Day Grid Fade (9:15am HKT anchor, spans lunch into the afternoon)",
            "uses_ml": False,
            "engine": "session_grid",
            "entry_logic": (
                f"Identical mechanics to afternoon_grid_fade, but anchored to the 09:15 HKT "
                "day-session open instead of the 13:00 HKT reopen, and using a TIGHTENED "
                f"entry threshold ({config.GRID_ENTRY_TICKS_TIGHT} ticks instead of the "
                "original 100) - at the 100-tick threshold this strategy traded "
                "far more often than afternoon_grid_fade and lost the resulting gross edge to "
                "transaction costs; a parameter sweep over the same OOS window showed it turns "
                f"OOS-profitable from roughly {config.GRID_ENTRY_TICKS_TIGHT - 25}-"
                f"{config.GRID_ENTRY_TICKS_TIGHT} ticks onward (see config.GRID_ENTRY_TICKS_TIGHT "
                "for the full sweep and its selection-bias caveat). Each day, anchor = the "
                f"spread's platform-display price at the 09:15 HKT open. Flat -> enter 1 lot "
                f"FADING the first {config.GRID_ENTRY_TICKS_TIGHT}-tick move away from the "
                "anchor in either direction. At most one such grid cycle is started per day."
            ),
            "exit_logic": (
                f"Adds: while below {config.GRID_MAX_LOTS} lots, add 1 more lot every "
                f"further {config.GRID_ADD_TICKS}-tick extension AGAINST the position - lot 2 "
                f"at anchor +/-{config.GRID_ENTRY_TICKS_TIGHT + config.GRID_ADD_TICKS} ticks, "
                f"lot 3 at anchor +/-{config.GRID_ENTRY_TICKS_TIGHT + 2 * config.GRID_ADD_TICKS} "
                f"ticks. Take-profit: close all lots when price moves "
                f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor "
                "of the current average entry price (checked at every lot count, 1 to 3, at "
                "any time from the 09:15 morning open through the 16:30 afternoon close). "
                "Unlike afternoon_grid_fade, an open position is carried straight through the "
                "12:30-13:00 HKT lunch break into the afternoon session - it is NOT flattened "
                "or re-anchored at 1pm, only monitored continuously against the same 9:15am "
                "anchor and the same lot-count-conditional stop/take-profit as trading resumes. "
                "Force-flattened at the 16:30 HKT day-session close every day regardless of "
                "P&L (never carried into the night session or to the next day)."
            ),
            "stop_loss": (
                f"Only arms once all {config.GRID_MAX_LOTS} lots are on. Closes the entire "
                f"position if price extends a further {config.GRID_STOP_TICKS_BEYOND_MAX} "
                f"ticks beyond the 3rd lot's trigger level (i.e. anchor +/-"
                f"{config.GRID_ENTRY_TICKS_TIGHT + config.GRID_ADD_TICKS * (config.GRID_MAX_LOTS - 1) + config.GRID_STOP_TICKS_BEYOND_MAX} "
                "ticks), checked continuously across the whole day session including through "
                "the lunch break gap."
            ),
            "take_profit": f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor of the average entry price across all currently-held lots, checked any time during the full day session.",
            "hedge_ratio": {
                **common_hedge,
                "tick_reference": (
                    "Tick-distance triggers (entry/add/stop) are measured on the trading "
                    "platform's own displayed spread price (config.spread_ui_price), matching "
                    "how a discretionary trader reads the platform - NOT the $ P&L spread "
                    "series used for backtest accounting (that remains spread_dollar_value)."
                ),
            },
            "position_sizing": {
                "method": "fixed_lots_per_grid_rung",
                "description": (
                    "1 spread unit (the fixed "
                    f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot ratio) per "
                    f"grid rung, up to {config.GRID_MAX_LOTS} units total - NOT "
                    "volatility-scaled, same as afternoon_grid_fade."
                ),
            },
            "costs": common_costs,
            "params": {
                "anchor_utc_hour": config.MORNING_ANCHOR_UTC_HOUR,
                "day_session_end_utc_hour": config.DAY_SESSION_END_UTC_HOUR,
                "entry_ticks": config.GRID_ENTRY_TICKS_TIGHT,
                "add_ticks": config.GRID_ADD_TICKS,
                "max_lots": config.GRID_MAX_LOTS,
                "stop_ticks_beyond_max": config.GRID_STOP_TICKS_BEYOND_MAX,
                "take_profit_ticks": config.GRID_TAKE_PROFIT_TICKS,
            },
        },
        {
            "id": "morning_grid_fade",
            "name": "Morning Session Grid Fade (9:15am HKT anchor, flat by lunch)",
            "uses_ml": False,
            "engine": "session_grid",
            "entry_logic": (
                f"Identical mechanics to afternoon_grid_fade, but confined to the morning "
                "session only: anchored to the 09:15 HKT day-session open, and force-flattened "
                "at 12:30 HKT (the lunch break) instead of the 16:30 close, using a TIGHTENED "
                f"entry threshold ({config.GRID_ENTRY_TICKS_TIGHT} ticks instead of the "
                "original 100) - at the 100-tick threshold this strategy traded "
                "far more often than afternoon_grid_fade and lost the resulting gross edge to "
                "transaction costs; a parameter sweep over the same OOS window showed it turns "
                f"OOS-profitable from roughly {config.GRID_ENTRY_TICKS_TIGHT - 25}-"
                f"{config.GRID_ENTRY_TICKS_TIGHT} ticks onward (see config.GRID_ENTRY_TICKS_TIGHT "
                "for the full sweep and its selection-bias caveat). Each day, anchor = "
                f"the spread's platform-display price at the 09:15 HKT open. Flat -> enter 1 "
                f"lot FADING the first {config.GRID_ENTRY_TICKS_TIGHT}-tick move away from the "
                "anchor in either direction. At most one such grid cycle is started per morning."
            ),
            "exit_logic": (
                f"Adds: while below {config.GRID_MAX_LOTS} lots, add 1 more lot every "
                f"further {config.GRID_ADD_TICKS}-tick extension AGAINST the position - lot 2 "
                f"at anchor +/-{config.GRID_ENTRY_TICKS_TIGHT + config.GRID_ADD_TICKS} ticks, "
                f"lot 3 at anchor +/-{config.GRID_ENTRY_TICKS_TIGHT + 2 * config.GRID_ADD_TICKS} "
                f"ticks. Take-profit: close all lots when price moves "
                f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor "
                "of the current average entry price (checked at every lot count, 1 to 3). "
                "Force-flattened at 12:30 HKT every day regardless of P&L if the stop-loss "
                "has not already triggered - the position is never carried into the lunch "
                "break, the afternoon session, or the next day."
            ),
            "stop_loss": (
                f"Only arms once all {config.GRID_MAX_LOTS} lots are on. Closes the entire "
                f"position if price extends a further {config.GRID_STOP_TICKS_BEYOND_MAX} "
                f"ticks beyond the 3rd lot's trigger level (i.e. anchor +/-"
                f"{config.GRID_ENTRY_TICKS_TIGHT + config.GRID_ADD_TICKS * (config.GRID_MAX_LOTS - 1) + config.GRID_STOP_TICKS_BEYOND_MAX} "
                "ticks) - a fixed, bounded max loss regardless of how the average entry "
                "price has moved as lots were added."
            ),
            "take_profit": f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor of the average entry price across all currently-held lots, checked any time during the morning session.",
            "hedge_ratio": {
                **common_hedge,
                "tick_reference": (
                    "Tick-distance triggers (entry/add/stop) are measured on the trading "
                    "platform's own displayed spread price (config.spread_ui_price), matching "
                    "how a discretionary trader reads the platform - NOT the $ P&L spread "
                    "series used for backtest accounting (that remains spread_dollar_value)."
                ),
            },
            "position_sizing": {
                "method": "fixed_lots_per_grid_rung",
                "description": (
                    "1 spread unit (the fixed "
                    f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot ratio) per "
                    f"grid rung, up to {config.GRID_MAX_LOTS} units total - NOT "
                    "volatility-scaled, same as afternoon_grid_fade."
                ),
            },
            "costs": common_costs,
            "params": {
                "anchor_utc_hour": config.MORNING_ANCHOR_UTC_HOUR,
                "day_session_end_utc_hour": config.MORNING_SESSION_END_UTC_HOUR,
                "entry_ticks": config.GRID_ENTRY_TICKS_TIGHT,
                "add_ticks": config.GRID_ADD_TICKS,
                "max_lots": config.GRID_MAX_LOTS,
                "stop_ticks_beyond_max": config.GRID_STOP_TICKS_BEYOND_MAX,
                "take_profit_ticks": config.GRID_TAKE_PROFIT_TICKS,
            },
        },
        {
            "id": "mid_morning_full_day_grid_fade",
            "name": "Mid-Morning Full-Day Grid Fade (9:30am HKT anchor, spans lunch into the afternoon)",
            "uses_ml": False,
            "engine": "session_grid",
            "entry_logic": (
                f"Identical mechanics to full_day_grid_fade, but anchored to the 09:30 HKT "
                "price (15 minutes into the day session) instead of the 09:15 HKT open, using "
                f"an entry threshold ({config.MID_MORNING_FULL_DAY_ENTRY_TICKS} ticks) tuned "
                "specifically for this anchor via tune_tools.py's tune/test split - validated "
                "on the untouched TEST half with a positive but thin-sample edge (20 trades, "
                "+1.96% return, Sharpe 0.409; see config.MID_MORNING_FULL_DAY_ENTRY_TICKS for "
                "the full sweep and its caveats). Each day, anchor = the spread's "
                f"platform-display price at the 09:30 HKT mark. Flat -> enter 1 lot FADING the "
                f"first {config.MID_MORNING_FULL_DAY_ENTRY_TICKS}-tick move away from the "
                "anchor in either direction. At most one such grid cycle is started per day."
            ),
            "exit_logic": (
                f"Adds: while below {config.GRID_MAX_LOTS} lots, add 1 more lot every "
                f"further {config.GRID_ADD_TICKS}-tick extension AGAINST the position - lot 2 "
                f"at anchor +/-{config.MID_MORNING_FULL_DAY_ENTRY_TICKS + config.GRID_ADD_TICKS} ticks, "
                f"lot 3 at anchor +/-{config.MID_MORNING_FULL_DAY_ENTRY_TICKS + 2 * config.GRID_ADD_TICKS} "
                f"ticks. Take-profit: close all lots when price moves "
                f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor "
                "of the current average entry price (checked at every lot count, 1 to 3, at "
                "any time from the 09:30 anchor through the 16:30 afternoon close). An open "
                "position is carried straight through the 12:30-13:00 HKT lunch break into the "
                "afternoon session - it is NOT flattened or re-anchored at 1pm, only monitored "
                "continuously against the same 9:30am anchor and the same lot-count-conditional "
                "stop/take-profit as trading resumes. Force-flattened at the 16:30 HKT "
                "day-session close every day regardless of P&L (never carried into the night "
                "session or to the next day)."
            ),
            "stop_loss": (
                f"Only arms once all {config.GRID_MAX_LOTS} lots are on. Closes the entire "
                f"position if price extends a further {config.GRID_STOP_TICKS_BEYOND_MAX} "
                f"ticks beyond the 3rd lot's trigger level (i.e. anchor +/-"
                f"{config.MID_MORNING_FULL_DAY_ENTRY_TICKS + config.GRID_ADD_TICKS * (config.GRID_MAX_LOTS - 1) + config.GRID_STOP_TICKS_BEYOND_MAX} "
                "ticks), checked continuously across the whole day session including through "
                "the lunch break gap."
            ),
            "take_profit": f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor of the average entry price across all currently-held lots, checked any time during the full day session.",
            "hedge_ratio": {
                **common_hedge,
                "tick_reference": (
                    "Tick-distance triggers (entry/add/stop) are measured on the trading "
                    "platform's own displayed spread price (config.spread_ui_price), matching "
                    "how a discretionary trader reads the platform - NOT the $ P&L spread "
                    "series used for backtest accounting (that remains spread_dollar_value)."
                ),
            },
            "position_sizing": {
                "method": "fixed_lots_per_grid_rung",
                "description": (
                    "1 spread unit (the fixed "
                    f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot ratio) per "
                    f"grid rung, up to {config.GRID_MAX_LOTS} units total - NOT "
                    "volatility-scaled, same as afternoon_grid_fade."
                ),
            },
            "costs": common_costs,
            "params": {
                "anchor_utc_hour": config.MID_MORNING_ANCHOR_UTC_HOUR,
                "day_session_end_utc_hour": config.DAY_SESSION_END_UTC_HOUR,
                "entry_ticks": config.MID_MORNING_FULL_DAY_ENTRY_TICKS,
                "add_ticks": config.GRID_ADD_TICKS,
                "max_lots": config.GRID_MAX_LOTS,
                "stop_ticks_beyond_max": config.GRID_STOP_TICKS_BEYOND_MAX,
                "take_profit_ticks": config.GRID_TAKE_PROFIT_TICKS,
            },
        },
        {
            "id": "mid_morning_grid_fade",
            "name": "Mid-Morning Grid Fade (9:30am HKT anchor, flat by lunch)",
            "uses_ml": False,
            "engine": "session_grid",
            "entry_logic": (
                f"Identical mechanics to morning_grid_fade, but anchored to the 09:30 HKT "
                "price (15 minutes into the day session) instead of the 09:15 HKT open, using "
                f"an entry threshold ({config.MID_MORNING_ENTRY_TICKS} ticks) tuned specifically "
                "for this anchor via tune_tools.py's tune/test split -- NOTE: this was the best "
                "candidate on the TUNE half but did NOT validate on the untouched TEST half "
                "(21 trades, -8.33% return, Sharpe -0.373); kept at the TUNE-selected value "
                "rather than cherry-picked, but treat this strategy as NOT out-of-sample "
                "validated (see config.MID_MORNING_ENTRY_TICKS for the full sweep). Each day, "
                f"anchor = the spread's platform-display price at the 09:30 HKT mark. Flat -> "
                f"enter 1 lot FADING the first {config.MID_MORNING_ENTRY_TICKS}-tick move away "
                "from the anchor in either direction. At most one such grid cycle is started "
                "per morning."
            ),
            "exit_logic": (
                f"Adds: while below {config.GRID_MAX_LOTS} lots, add 1 more lot every "
                f"further {config.GRID_ADD_TICKS}-tick extension AGAINST the position - lot 2 "
                f"at anchor +/-{config.MID_MORNING_ENTRY_TICKS + config.GRID_ADD_TICKS} ticks, "
                f"lot 3 at anchor +/-{config.MID_MORNING_ENTRY_TICKS + 2 * config.GRID_ADD_TICKS} "
                f"ticks. Take-profit: close all lots when price moves "
                f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor "
                "of the current average entry price (checked at every lot count, 1 to 3). "
                "Force-flattened at 12:30 HKT every day regardless of P&L if the stop-loss "
                "has not already triggered - the position is never carried into the lunch "
                "break, the afternoon session, or the next day."
            ),
            "stop_loss": (
                f"Only arms once all {config.GRID_MAX_LOTS} lots are on. Closes the entire "
                f"position if price extends a further {config.GRID_STOP_TICKS_BEYOND_MAX} "
                f"ticks beyond the 3rd lot's trigger level (i.e. anchor +/-"
                f"{config.MID_MORNING_ENTRY_TICKS + config.GRID_ADD_TICKS * (config.GRID_MAX_LOTS - 1) + config.GRID_STOP_TICKS_BEYOND_MAX} "
                "ticks) - a fixed, bounded max loss regardless of how the average entry "
                "price has moved as lots were added."
            ),
            "take_profit": f"{config.GRID_TAKE_PROFIT_TICKS} ticks in favor of the average entry price across all currently-held lots, checked any time during the morning session.",
            "hedge_ratio": {
                **common_hedge,
                "tick_reference": (
                    "Tick-distance triggers (entry/add/stop) are measured on the trading "
                    "platform's own displayed spread price (config.spread_ui_price), matching "
                    "how a discretionary trader reads the platform - NOT the $ P&L spread "
                    "series used for backtest accounting (that remains spread_dollar_value)."
                ),
            },
            "position_sizing": {
                "method": "fixed_lots_per_grid_rung",
                "description": (
                    "1 spread unit (the fixed "
                    f"{config.MHI_HEDGE_LOTS}:{config.HHI_HEDGE_LOTS} MHI:HHI lot ratio) per "
                    f"grid rung, up to {config.GRID_MAX_LOTS} units total - NOT "
                    "volatility-scaled, same as afternoon_grid_fade."
                ),
            },
            "costs": common_costs,
            "params": {
                "anchor_utc_hour": config.MID_MORNING_ANCHOR_UTC_HOUR,
                "day_session_end_utc_hour": config.MORNING_SESSION_END_UTC_HOUR,
                "entry_ticks": config.MID_MORNING_ENTRY_TICKS,
                "add_ticks": config.GRID_ADD_TICKS,
                "max_lots": config.GRID_MAX_LOTS,
                "stop_ticks_beyond_max": config.GRID_STOP_TICKS_BEYOND_MAX,
                "take_profit_ticks": config.GRID_TAKE_PROFIT_TICKS,
            },
        },
    ]

    config.STRATEGIES_PATH.write_text(json.dumps(strategies, indent=2))
    return json.dumps(strategies, indent=2)
