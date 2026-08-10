"""Statistical tests for pair-trading suitability: correlation, cointegration,
mean-reversion half-life, and regime stability."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from crewai.tools import tool

from pair_crew import config


def _hedge_ratio_ols(y: pd.Series, x: pd.Series):
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha, beta = model.params[0], model.params[1]
    resid = y.values - (alpha + beta * x.values)
    return alpha, beta, pd.Series(resid, index=y.index)


def _half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    X = sm.add_constant(lag.values)
    model = sm.OLS(delta.values, X).fit()
    theta = model.params[1]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2) / theta)


def _hurst_exponent(series: pd.Series, max_lag: int = 40) -> float:
    """Quick variance-scaling Hurst estimate; H<0.5 suggests mean reversion."""
    s = series.dropna().values
    lags = range(2, min(max_lag, len(s) // 2))
    tau = [np.std(s[lag:] - s[:-lag]) for lag in lags]
    tau = [t if t > 0 else 1e-8 for t in tau]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


@tool("test_pair_relationship")
def test_pair_relationship() -> str:
    """
    Test whether MHI and HHI form a statistically sound pair-trading
    relationship. Loads outputs/aligned_data.parquet (must be produced by
    load_clean_align_pair_data first) and, using ONLY the chronological
    training window (first fraction of history, matching the ML train/test
    split so the suitability verdict itself has no lookahead into the
    out-of-sample period), computes: price and return correlation, the
    stationarity (Augmented Dickey-Fuller test) of the ACTUAL tradeable
    spread at the fixed 2-MHI:1-HHI contract ratio, the spread's
    mean-reversion half-life, a Hurst exponent estimate, and rolling-
    correlation stability. A separate Engle-Granger cointegration test (which
    estimates its own optimal hedge ratio internally) is also reported, but
    purely as an informational cross-check against the fixed ratio actually
    traded -- it is not used to set the hedge ratio.

    Full-history versions of the same statistics are also included for
    context/comparison. Saves full results to outputs/pair_stats.json and
    returns a JSON string with all statistics plus a suitability verdict.
    """
    df = pd.read_parquet(config.ALIGNED_DATA_PATH)
    n_train = int(len(df) * config.TRAIN_FRACTION)
    train = df.iloc[:n_train]

    def _stats_block(block: pd.DataFrame) -> dict:
        log_m = np.log(block["MHI_Close"])
        log_h = np.log(block["HHI_Close"])
        ret_m = log_m.diff().dropna()
        ret_h = log_h.diff().dropna()

        price_corr = float(block["MHI_Close"].corr(block["HHI_Close"]))
        return_corr = float(ret_m.corr(ret_h))

        # Actual tradeable spread at the fixed lot ratio + contract
        # multipliers -- this, not an estimated regression residual, is what
        # stationarity/half-life/Hurst are tested against.
        resid = config.spread_dollar_value(block["MHI_Close"], block["HHI_Close"])
        # maxlag is capped explicitly: statsmodels' default autolag search
        # picks maxlag ~= 12*(nobs/100)^0.25 (80+ lags on our ~230k-350k-row
        # 1-min series), which widens the OLS design matrix enough to trigger
        # a known LAPACK gesdd (SVD) pathological-memory-use bug for tall/
        # skinny matrices of that shape -- observed to balloon RSS to ~7GB
        # and get the process OOM-killed. 20 lags (~20 min of 1-min bars) is
        # already generous for autocorrelation this test needs to absorb.
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(resid, maxlag=20, autolag="AIC")
        # Informational only: what ratio would Engle-Granger's own internal
        # regression pick, for comparison against the fixed 2:1 ratio traded.
        eg_stat, eg_p, eg_crit = coint(log_m.values, log_h.values, trend="c", maxlag=20)
        implied_alpha, implied_beta, _ = _hedge_ratio_ols(log_m, log_h)
        half_life = _half_life(resid)
        hurst = _hurst_exponent(resid)

        roll_corr = ret_m.rolling(config.ROLLING_WINDOW_SHORT).corr(ret_h).dropna()

        return {
            "n_bars": len(block),
            "price_correlation": round(price_corr, 4),
            "return_correlation": round(return_corr, 4),
            "fixed_hedge_ratio": {
                "mhi_lots": config.MHI_HEDGE_LOTS,
                "hhi_lots": config.HHI_HEDGE_LOTS,
                "mhi_multiplier": config.MHI_CONTRACT_MULTIPLIER,
                "hhi_multiplier": config.HHI_CONTRACT_MULTIPLIER,
                "ratio": round(config.FIXED_HEDGE_RATIO, 6),
            },
            "log_price_ols_implied_ratio": {
                "alpha": round(float(implied_alpha), 6),
                "beta": round(float(implied_beta), 6),
                "note": "informational only - statistically 'optimal' ratio from an unconstrained OLS regression on log prices, for comparison against the fixed 2:1 contract ratio actually traded",
            },
            "adf_test_on_spread": {
                "statistic": round(float(adf_stat), 4),
                "p_value": round(float(adf_p), 4),
                "critical_values": {k: round(float(v), 4) for k, v in adf_crit.items()},
                "stationary_at_5pct": bool(adf_p < 0.05),
            },
            "engle_granger_cointegration": {
                "statistic": round(float(eg_stat), 4),
                "p_value": round(float(eg_p), 4),
                "cointegrated_at_10pct": bool(eg_p < 0.10),
                "cointegrated_at_5pct": bool(eg_p < 0.05),
            },
            "mean_reversion_half_life_bars": round(half_life, 2) if np.isfinite(half_life) else None,
            "hurst_exponent_of_spread": round(hurst, 4),
            "rolling_return_correlation": {
                "window_bars": config.ROLLING_WINDOW_SHORT,
                "mean": round(float(roll_corr.mean()), 4),
                "std": round(float(roll_corr.std()), 4),
                "pct_time_above_0.3": round(float((roll_corr > 0.3).mean()), 4),
            },
        }

    train_stats = _stats_block(train)
    full_stats = _stats_block(df)

    hl = train_stats["mean_reversion_half_life_bars"]
    reasonable_half_life = hl is not None and 0 < hl <= 300
    is_suitable = (
        train_stats["adf_test_on_spread"]["stationary_at_5pct"]
        and train_stats["engle_granger_cointegration"]["cointegrated_at_10pct"]
        and reasonable_half_life
        and train_stats["return_correlation"] > 0.15
    )
    if is_suitable:
        verdict = "SUITABLE"
        reasoning = (
            "The training-window spread is stationary (ADF) and the pair shows "
            "Engle-Granger cointegration evidence with a tradable mean-reversion "
            f"half-life (~{hl:.1f} bars). Statistical arbitrage / mean-reversion "
            "pair trading is a reasonable approach for this pair."
        )
    elif train_stats["return_correlation"] > 0.15 and (
        train_stats["adf_test_on_spread"]["stationary_at_5pct"]
        or train_stats["engle_granger_cointegration"]["cointegrated_at_10pct"]
    ):
        verdict = "MARGINAL"
        reasoning = (
            "The pair shows positive return co-movement and partial evidence of "
            "mean reversion (either ADF or Engle-Granger, not both, or a borderline "
            "half-life), so pair trading is workable but the relationship should be "
            "treated as unstable and monitored/refit frequently, with conservative "
            "sizing and wider risk controls."
        )
    else:
        verdict = "NOT SUITABLE"
        reasoning = (
            "Neither cointegration test nor return correlation provides strong "
            "support for a stable long-run relationship. A classic mean-reversion "
            "spread strategy is not well justified; any strategy attempted should "
            "be flagged as high risk / exploratory."
        )

    result = {
        "verdict": verdict,
        "reasoning": reasoning,
        "train_window_stats": train_stats,
        "full_history_stats": full_stats,
        "train_fraction_used": config.TRAIN_FRACTION,
    }
    config.PAIR_STATS_PATH.write_text(json.dumps(result, indent=2))
    return json.dumps(result, indent=2)
