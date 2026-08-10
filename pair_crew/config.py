"""Shared paths and constants for the pair-trading crew."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Historical 1-min data is supplied via the Streamlit uploader (app.py sidebar),
# which writes/merges uploads into these two files -- there is no bundled
# source CSV in this repo the way the PBES/FCPO template has one checked in.
MHI_CSV = DATA_DIR / "MHI_1min.csv"
HHI_CSV = DATA_DIR / "HHI_1min.csv"

ALIGNED_DATA_PATH = OUTPUT_DIR / "aligned_data.parquet"
DATA_SUMMARY_PATH = OUTPUT_DIR / "data_preprocessing_summary.json"
FEATURES_PATH = OUTPUT_DIR / "features.parquet"
FEATURE_SUMMARY_PATH = OUTPUT_DIR / "feature_engineering_summary.json"
PAIR_STATS_PATH = OUTPUT_DIR / "pair_stats.json"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.json"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
BEST_MODEL_META_PATH = OUTPUT_DIR / "best_model_meta.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.parquet"
STRATEGIES_PATH = OUTPUT_DIR / "strategies.json"
BACKTEST_RESULTS_PATH = OUTPUT_DIR / "backtest_results.json"
BACKTEST_GRID_VARIANT_PATH = OUTPUT_DIR / "backtest_grid_variant.json"
EVALUATION_PATH = OUTPUT_DIR / "evaluation.json"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.md"

# Which entry point is currently running the shared tool functions -- set to
# "crewai_agents" by main.py before kickoff(); left at this default when run
# via app.py's deterministic pipeline (no LLM). compile_final_report reads
# this to title the report accurately, since both entry points call the
# exact same tool functions and write to the exact same FINAL_REPORT_PATH.
ENTRY_POINT = "deterministic_pipeline"

# --- Fixed pair contract spec (HKEX MHI/HHI spread) ---
# Exchange-defined lot ratio and per-leg contract multipliers, NOT a
# statistical estimate -- replaces the rolling OLS-estimated hedge ratio
# everywhere (spread calculation, cointegration test, backtest P&L, position
# sizing). For every MHI_HEDGE_LOTS lots of MHI the spread trades
# HHI_HEDGE_LOTS lots of HHI on the opposite side.
MHI_HEDGE_LOTS = 2
HHI_HEDGE_LOTS = 1

# Real $-per-index-point contract multiplier for each leg (HKEX spec: Mini-HSI
# futures = HKD 10/pt, HSCEI ("HHI") futures = HKD 50/pt). This is the number
# used in every $ calculation below (spread_dollar_value, spread_unit_notional,
# cointegration residuals, feature "spread" column, backtest P&L) -- exactly
# the role PBES_CONTRACT_MULTIPLIER/FCPO_CONTRACT_MULTIPLIER play in the template.
MHI_CONTRACT_MULTIPLIER = 10.0
HHI_CONTRACT_MULTIPLIER = 50.0

# Reference only -- the "2 / 5" figures given for this pair are the multiplier
# the user's own trading platform uses to display a normalized spread *price*
# on its UI. They are NOT a $ P&L multiplier and are not used in any
# calculation in this pipeline (which uses MHI_CONTRACT_MULTIPLIER /
# HHI_CONTRACT_MULTIPLIER throughout instead). Kept only for reference/cross-
# check against the platform's own display, never read by any tool module.
MHI_SPREAD_UI_MULTIPLIER = 2
HHI_SPREAD_UI_MULTIPLIER = 5

# HKEX spec: minimum price fluctuation is 1 index point for both MHI and HHI.
TICK_SIZE = 1.0

# FCPO-leg $ notional per PBES-leg $ notional at the fixed lot ratio -- kept
# for reporting/back-compat where a single scalar "hedge ratio" is expected.
FIXED_HEDGE_RATIO = (HHI_HEDGE_LOTS * HHI_CONTRACT_MULTIPLIER) / (MHI_HEDGE_LOTS * MHI_CONTRACT_MULTIPLIER)


def spread_dollar_value(mhi_price, hhi_price):
    """$ value of one fixed-ratio MHI/HHI spread unit: long MHI_HEDGE_LOTS lots
    of MHI minus HHI_HEDGE_LOTS lots of HHI, each leg valued at its real
    contract multiplier. Linear in both prices, so a diff of this series is
    exactly the $ P&L of holding one unit across that price move."""
    return (
        MHI_HEDGE_LOTS * MHI_CONTRACT_MULTIPLIER * mhi_price
        - HHI_HEDGE_LOTS * HHI_CONTRACT_MULTIPLIER * hhi_price
    )


def spread_unit_notional(mhi_price, hhi_price):
    """Gross $ notional of one fixed-ratio MHI/HHI spread unit (sum of both
    legs' absolute value) -- used for leverage calculations."""
    return (
        MHI_HEDGE_LOTS * MHI_CONTRACT_MULTIPLIER * mhi_price
        + HHI_HEDGE_LOTS * HHI_CONTRACT_MULTIPLIER * hhi_price
    )


# --- Real per-leg transaction costs ---
# Real exchange/broker commission schedules, not a bps-of-notional proxy:
# flat HKD per lot, per side, per leg. Slippage is expressed in ticks per
# leg, per side (1 tick = TICK_SIZE index points), converted to $ via each
# leg's own contract multiplier -- so MHI and HHI slippage are not the same
# dollar amount even though both assume "1 tick" of adverse fill.
MHI_COMMISSION_PER_LOT = 6.0    # HKD, per side
HHI_COMMISSION_PER_LOT = 7.5    # HKD, per side
SLIPPAGE_TICKS = 0              # per leg, per side -- slippage disabled; costs are commission-only


def spread_transaction_cost(n_units: float = 1.0) -> float:
    """$ cost of putting on (or taking off) `n_units` of the fixed-ratio
    spread on ONE side (entry or exit) -- commission + slippage, both legs.
    Called twice per round-trip trade (once at entry, once at exit)."""
    mhi_lots = MHI_HEDGE_LOTS * n_units
    hhi_lots = HHI_HEDGE_LOTS * n_units
    commission = mhi_lots * MHI_COMMISSION_PER_LOT + hhi_lots * HHI_COMMISSION_PER_LOT
    slippage = (
        mhi_lots * SLIPPAGE_TICKS * TICK_SIZE * MHI_CONTRACT_MULTIPLIER
        + hhi_lots * SLIPPAGE_TICKS * TICK_SIZE * HHI_CONTRACT_MULTIPLIER
    )
    return commission + slippage


# --- Trading session ---
# No session filter: both MHI and HHI share the same HKEX trading hours, so
# every bar in the aligned data is eligible for signals/trades. session_mask
# is kept as a pass-through (all True) so tools that call it don't need a
# special no-filter code path.
def session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """No session filter for this pair -- MHI and HHI share identical trading
    hours, so every timestamp in the aligned data is eligible."""
    return np.ones(len(index), dtype=bool)


# --- Data engineering knobs ---
RESAMPLE_FREQ = "1min"        # both legs are liquid HKEX index futures; kept at native 1-min bar size (no resampling)
MAX_FFILL_BARS = 5            # max consecutive bars to forward-fill a stale last price across
ROLLING_WINDOW_SHORT = 20     # ~20 minutes
ROLLING_WINDOW_LONG = 120     # ~2 hours
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 3.5

# --- Feature engineering knobs ---
ROC_WINDOW_BARS = 10          # rate-of-change lookback, ~10 minutes

# --- ML knobs ---
N_CV_SPLITS = 5
EMBARGO_BARS = 15             # bars purged between train/test fold boundaries to prevent leakage
LABEL_HORIZON = 10            # bars ahead used to define forward-looking targets, ~10 minutes

# --- Backtest knobs ---
TRAIN_FRACTION = 0.65         # fraction of aligned history used for ML training; remainder is true out-of-sample
INITIAL_CAPITAL = 100_000.0   # HKD
RISK_FRACTION = 0.15          # target fraction of equity at risk (vol-scaled) per pair position
MAX_LEVERAGE = 3.0            # cap on one spread unit's gross notional / equity
MIN_VOL_FLOOR = 1.0           # floor on $ spread volatility used in position sizing to avoid blow-ups
MAX_HOLDING_BARS = 240        # ~4 hours; re-check against observed mean-reversion half-life once pair_stats.json exists

# --- Afternoon session grid/fade strategy knobs (strategy_tools.py id
# "afternoon_grid_fade", backtest_tools.py _run_grid_backtest) ---
# HKEX day session reopens at 13:00 HKT after the lunch break and closes
# ~16:30 HKT; both converted to UTC (HKT = UTC+8 year-round, no DST) since
# aligned_data.parquet is indexed in the source CSVs' "Timestamp (UTC)".
# Verified against the actual data: bars resume at exactly 05:00 UTC after
# the 04:xx UTC lunch-break gap, and the day session's last bars are ~08:29
# UTC before a gap to the 09:15 UTC night/T+1 session.
AFTERNOON_ANCHOR_UTC_HOUR = 5.0     # 13:00 HKT reopen -- the daily grid anchor price (afternoon_grid_fade)
MORNING_ANCHOR_UTC_HOUR = 1.25      # 09:15 HKT day-session open -- the daily grid anchor price (full_day_grid_fade, morning_grid_fade);
                                     # confirmed against the data: bars begin exactly at 01:15 UTC each day
MID_MORNING_ANCHOR_UTC_HOUR = 1.5   # 09:30 HKT -- 15 minutes into the day session, offered as a chat-frontend
                                     # grid_anchor option (chat_app/translate.py) alongside the 9:15am open and
                                     # 1pm reopen anchors above
DAY_SESSION_END_UTC_HOUR = 8.5      # 16:30 HKT close -- position is force-flattened here, never carried into the night session
MORNING_SESSION_END_UTC_HOUR = 4.5  # 12:30 HKT -- force-flatten point for morning_grid_fade. The data's actual
                                     # last morning bar is ~03:59 UTC (11:59am HKT, i.e. the real lunch break is
                                     # 12:00-13:00 HKT, not 12:30-13:00) -- 4.5 is kept at the requested 12:30 HKT
                                     # anyway since it's harmless: no bars exist in either the 12:00-12:30 or the
                                     # 12:30-13:00 stretch, so the actual force-flatten bar is identical either way.
NIGHT_SESSION_END_UTC_HOUR = 15.0   # 23:00 HKT -- force-flatten point for chat-frontend grid strategies that
                                     # explicitly want to carry a position from the morning open through the day
                                     # session and into the night/T+1 session, rather than flattening at the
                                     # 16:30 HKT day close (DAY_SESSION_END_UTC_HOUR). 23:00 HKT falls well within
                                     # the night session's real bar range (~09:15-17:00 UTC, see the note above),
                                     # so this is a genuine mid-session cutoff, not a boundary edge case.
GRID_ENTRY_TICKS = 200               # distance from the 1pm anchor that triggers the first lot (afternoon_grid_fade).
                                     # Originally 100; raised to 200 after tune_tools.py's tune/test split (see
                                     # outputs/tune_test_validation.json) independently picked 200 as best on the TUNE
                                     # half (Dec 2025-Mar 2026) of the OOS window, validated on the untouched TEST half
                                     # (Mar-Jun 2026): 7 trades, +3.82% return, Sharpe 0.53, PF 2.09 -- better risk-
                                     # adjusted than the original 100-tick default, though on a thin (7-trade) sample.
GRID_ENTRY_TICKS_TIGHT = 250        # tightened entry distance for morning_grid_fade / full_day_grid_fade -- these two
                                     # traded far more often than afternoon_grid_fade at the 100-tick threshold and lost
                                     # money to transaction costs despite a decent gross edge. Validated via
                                     # tune_tools.py's tune/test split: the TUNE half (Dec 2025-Mar 2026) of the OOS
                                     # window independently picked 250 for both strategies, and the untouched TEST half
                                     # (Mar-Jun 2026) confirmed a positive but modest edge (full_day: 10 trades, +1.95%,
                                     # Sharpe 0.45; morning: 6 trades, +1.47%, Sharpe 0.48) -- smaller than the
                                     # originally-reported full-window numbers (+9.0%/+5.3%), which were inflated by
                                     # tuning against the same window used to report them. Directionally consistent
                                     # (positive Sharpe both halves, same threshold picked independently on the tune
                                     # half) but still a thin sample (6-10 trades) -- treat as corroborated but not yet
                                     # statistically decisive; revisit once more OOS history is available.
MID_MORNING_FULL_DAY_ENTRY_TICKS = 175  # entry threshold for mid_morning_full_day_grid_fade (9:30am anchor),
                                     # tuned via tune_tools.py's tune/test split: TUNE half (Dec 2025-Mar 2026)
                                     # independently picked 175, and the untouched TEST half (Mar-Jun 2026)
                                     # confirmed a positive edge: 20 trades, +1.96% return, Sharpe 0.409, PF 1.29
                                     # -- thin sample, treat as corroborated but not statistically decisive.
MID_MORNING_ENTRY_TICKS = 125       # entry threshold for mid_morning_grid_fade (9:30am anchor, flat by lunch),
                                     # tuned via the same tune/test split: TUNE half picked 125 as the best
                                     # candidate (composite_score 1.02), but the untouched TEST half did NOT
                                     # validate it -- 21 trades, -8.33% return, Sharpe -0.373, PF 0.36. This is
                                     # exactly the overfitting scenario the tune/test split exists to catch: kept
                                     # at the TUNE-selected value rather than hand-picking a better-looking TEST
                                     # number, but this strategy should be treated as NOT validated out-of-sample.
GRID_ADD_TICKS = 50                 # further adverse distance for each additional lot
GRID_MAX_LOTS = 3
GRID_STOP_TICKS_BEYOND_MAX = 50     # once at max lots, further adverse distance beyond the last rung that stops the whole position out
GRID_TAKE_PROFIT_TICKS = 50         # favorable distance from the average entry price that closes the whole position

# --- Tune/test split for session-grid entry-tick selection (tune_tools.py) ---
# GRID_ENTRY_TICKS_TIGHT above was picked by sweeping candidates against the
# ENTIRE OOS window that then evaluated it -- real selection bias. This splits
# that same OOS window chronologically into an earlier TUNE sub-period (used
# only to sweep candidates and pick one) and a later, untouched TEST
# sub-period (the locked-in choice is backtested there exactly once).
TUNE_TEST_SPLIT_FRACTION = 0.5      # fraction of the OOS window used as the TUNE sub-period; remainder is TEST
TUNE_ENTRY_TICKS_CANDIDATES = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
TUNE_MIN_TRADES = 4                 # minimum trades on the TUNE sub-period for a candidate to be scored at all
                                     # (avoids picking a single-trade fluke); deliberately below
                                     # MIN_TRADES_FOR_SIGNIFICANCE since the tune half is itself only ~3 months
                                     # and this strategy style already trades rarely


def spread_ui_price(mhi_price, hhi_price):
    """Platform-displayed MHI/HHI spread price using the trading platform's
    own UI convention (MHI_SPREAD_UI_MULTIPLIER / HHI_SPREAD_UI_MULTIPLIER) --
    NOT the same series as spread_dollar_value (real $ P&L). This is what a
    discretionary trader watching the platform's spread quote calls a "tick":
    one whole point of this series (TICK_SIZE = 1 index point, same as both
    legs). Used only for the afternoon_grid_fade strategy's tick-distance
    trigger levels; every $ P&L calculation still uses spread_dollar_value."""
    return MHI_SPREAD_UI_MULTIPLIER * mhi_price - HHI_SPREAD_UI_MULTIPLIER * hhi_price
