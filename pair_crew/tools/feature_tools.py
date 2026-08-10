"""Pair-trading feature engineering: hedge ratio, spread, z-score, rolling
correlation/volatility, simple TA indicators, and forward-looking ML targets."""
import json

import numpy as np
import pandas as pd
from crewai.tools import tool

from pair_crew import config


def _rolling_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure feature-engineering step. df must have MHI_Close/HHI_Close
    columns (e.g. outputs/aligned_data.parquet). Returns the full engineered
    feature table (including the three forward-looking targets) with NO
    warm-up/lookahead rows dropped -- the offline pipeline drops them below."""
    out = pd.DataFrame(index=df.index)
    out["MHI_Close"] = df["MHI_Close"]
    out["HHI_Close"] = df["HHI_Close"]

    log_m = np.log(df["MHI_Close"])
    log_h = np.log(df["HHI_Close"])
    out["ret_mhi"] = log_m.diff()
    out["ret_hhi"] = log_h.diff()

    w_short = config.ROLLING_WINDOW_SHORT
    w_long = config.ROLLING_WINDOW_LONG

    # Fixed HKEX contract ratio (2 MHI lots : 1 HHI lot, each at its own real
    # contract multiplier) -- a constant, not a rolling estimate, so unlike a
    # rolling-regression beta it needs no shift to avoid lookahead.
    out["hedge_ratio"] = config.FIXED_HEDGE_RATIO
    out["spread"] = config.spread_dollar_value(df["MHI_Close"], df["HHI_Close"])

    out["rolling_corr_returns"] = out["ret_mhi"].rolling(w_short).corr(out["ret_hhi"])

    spread_mean = out["spread"].rolling(w_long).mean()
    spread_std = out["spread"].rolling(w_long).std()
    out["spread_zscore"] = (out["spread"] - spread_mean) / spread_std

    out["spread_ma_short"] = out["spread"].rolling(w_short).mean()
    out["spread_ma_long"] = out["spread"].rolling(w_long).mean()
    out["spread_ma_crossover"] = out["spread_ma_short"] - out["spread_ma_long"]

    out["spread_rsi"] = _rolling_rsi(out["spread"], w_short)

    out["vol_mhi"] = out["ret_mhi"].rolling(w_short).std()
    out["vol_hhi"] = out["ret_hhi"].rolling(w_short).std()
    out["spread_vol"] = out["spread"].diff().rolling(w_short).std()

    out["spread_roc"] = out["spread"].diff(config.ROC_WINDOW_BARS)

    # --- forward-looking targets (labels only; never used as features) ---
    h = config.LABEL_HORIZON
    future_z = out["spread_zscore"].shift(-h)
    out["target_reversion"] = (future_z.abs() < out["spread_zscore"].abs()).astype(float)

    future_spread = out["spread"].shift(-h)
    out["target_direction"] = (future_spread > out["spread"]).astype(float)

    forward_realized_vol = out["spread"].diff().rolling(h).std().shift(-h)
    trailing_median_vol = out["spread_vol"].rolling(w_long).median()
    out["target_regime"] = (forward_realized_vol > trailing_median_vol).astype(float)

    return out


@tool("engineer_pair_features")
def engineer_pair_features() -> str:
    """
    Engineer pair-trading features and ML targets from outputs/aligned_data.parquet
    (produced by load_clean_align_pair_data). All rolling statistics use only
    trailing/past data so no feature at time t uses information from after
    time t - safe for causal, no-lookahead modeling.

    Features produced: the fixed 2-MHI:1-HHI-lot-ratio $ spread (each leg at
    its own real contract multiplier -- NOT a statistically estimated hedge
    ratio), both legs' log returns, rolling return correlation, spread
    z-score, short/long spread moving averages and their crossover, a spread
    RSI, rolling realized volatility of both legs and of the spread, and the
    spread's rate of change.

    Three forward-looking classification targets are also produced (labels
    only look forward from t, features only look backward from t):
      - target_reversion: 1 if |z-score| shrinks over the next horizon (mean
        reversion), else 0.
      - target_direction: 1 if the spread rises over the next horizon, else 0.
      - target_regime: 1 if realized spread volatility over the next horizon
        is elevated vs its trailing median (high-vol/risk-off regime), else 0.

    Saves the full feature table to outputs/features.parquet and returns a
    JSON summary of the feature list, target definitions, class balance, and
    row counts before/after dropping rolling-window warm-up and label
    look-ahead rows.
    """
    df = pd.read_parquet(config.ALIGNED_DATA_PATH)
    n0 = len(df)

    out = compute_pair_features(df)

    label_cols = ["target_reversion", "target_direction", "target_regime"]
    n_before_dropna = len(out)
    feature_cols = [c for c in out.columns if c not in label_cols + ["MHI_Close", "HHI_Close"]]
    out_clean = out.dropna(subset=feature_cols + label_cols)

    out_clean.to_parquet(config.FEATURES_PATH)

    class_balance = {c: round(float(out_clean[c].mean()), 4) for c in label_cols}

    summary = {
        "rows_before_feature_engineering": n0,
        "rows_after_dropping_warmup_and_label_horizon": len(out_clean),
        "rows_dropped": n_before_dropna - len(out_clean),
        "rolling_window_short_bars": config.ROLLING_WINDOW_SHORT,
        "rolling_window_long_bars": config.ROLLING_WINDOW_LONG,
        "label_horizon_bars": config.LABEL_HORIZON,
        "feature_columns": feature_cols,
        "target_columns": label_cols,
        "target_positive_class_rate": class_balance,
        "target_definitions": {
            "target_reversion": f"1 if |spread z-score| shrinks over the next {config.LABEL_HORIZON} bars (mean reversion), else 0",
            "target_direction": f"1 if hedge-ratio-adjusted spread is higher {config.LABEL_HORIZON} bars ahead, else 0",
            "target_regime": f"1 if realized spread volatility over the next {config.LABEL_HORIZON} bars exceeds its trailing median (high-vol regime), else 0",
        },
        "no_lookahead_design_notes": (
            "hedge_ratio is the fixed HKEX contract ratio (a constant, not a "
            "rolling estimate), so it carries no lookahead risk by construction. All "
            "other features use rolling windows ending at t (current-bar close, "
            "causal). All three targets are shifted forward and are only ever used "
            "as training labels, never as model inputs."
        ),
        "features_path": str(config.FEATURES_PATH),
    }
    config.FEATURE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)
