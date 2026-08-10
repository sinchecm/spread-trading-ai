"""Data loading, cleaning, and alignment for the MHI/HHI pair."""
import json

import numpy as np
import pandas as pd
from crewai.tools import tool

from pair_crew import config


def _load_raw(path):
    df = pd.read_csv(path)
    ts_col = "Timestamp (UTC)"
    df[ts_col] = pd.to_datetime(df[ts_col])
    n_raw = len(df)
    df = df.sort_values(ts_col)
    n_dupe_ts = int(df[ts_col].duplicated().sum())
    df = df.drop_duplicates(subset=ts_col, keep="last")
    df = df.set_index(ts_col)
    vol_col = [c for c in df.columns if "Volume" in c][0]
    df = df.rename(columns={vol_col: "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    n_bad_ohlc = int(((df["High"] < df["Low"]) | (df["Close"] <= 0)).sum())
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    n_before_session_filter = len(df)
    df = df[config.session_mask(df.index)]
    n_out_of_session = n_before_session_filter - len(df)
    return df, {
        "raw_rows": n_raw,
        "duplicate_timestamps_dropped": n_dupe_ts,
        "invalid_ohlc_rows_dropped": n_bad_ohlc,
        "out_of_session_rows_dropped": n_out_of_session,
        "clean_rows": len(df),
        "raw_start": str(df.index.min()) if len(df) else None,
        "raw_end": str(df.index.max()) if len(df) else None,
    }


def _resample(df, freq, max_ffill_bars):
    """Resample to a regular bar grid over the full continuous history (no
    session blocks to split on -- MHI and HHI share the same HKEX trading
    hours, so config.session_mask never removes anything and there is no
    weekend/inter-session gap to avoid bridging), forward-filling short
    no-trade gaps causally (uses only past prints, never future ones)."""
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if df.empty:
        empty = pd.DataFrame(columns=columns).astype(float)
        return empty, {
            "bars_after_resample": 0,
            "bars_with_no_trade": 0,
            "bars_forward_filled": 0,
            "bars_left_as_gap": 0,
        }

    o = df["Open"].resample(freq).first()
    h = df["High"].resample(freq).max()
    l = df["Low"].resample(freq).min()
    c = df["Close"].resample(freq).last()
    v = df["Volume"].resample(freq).sum()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    n_bars_total = len(out)
    n_bars_missing = int(out["Close"].isna().sum())

    close_ffilled = out["Close"].ffill(limit=max_ffill_bars)
    filled_mask = out["Close"].isna() & close_ffilled.notna()
    out.loc[filled_mask, "Open"] = close_ffilled[filled_mask]
    out.loc[filled_mask, "High"] = close_ffilled[filled_mask]
    out.loc[filled_mask, "Low"] = close_ffilled[filled_mask]
    out.loc[filled_mask, "Close"] = close_ffilled[filled_mask]
    out.loc[filled_mask, "Volume"] = 0.0

    stats = {
        "bars_after_resample": n_bars_total,
        "bars_with_no_trade": n_bars_missing,
        "bars_forward_filled": int(filled_mask.sum()),
        "bars_left_as_gap": int(out["Close"].isna().sum()),
    }
    return out, stats


@tool("load_clean_align_pair_data")
def load_clean_align_pair_data() -> str:
    """
    Load the MHI and HHI 1-minute OHLCV CSV files, clean each series (parse
    timestamps, drop duplicate timestamps and invalid OHLC rows), resample
    both to a common regular bar grid (no trading-session filter is applied
    -- MHI and HHI share the same HKEX trading hours), forward-filling short
    no-trade gaps with the last traded price (causal, no lookahead), and
    inner-align the two instruments on their overlapping timestamp grid.

    Saves the aligned dataset to outputs/aligned_data.parquet with columns
    MHI_Open/High/Low/Close/Volume and HHI_Open/High/Low/Close/Volume.

    Returns a JSON string summarizing the preprocessing steps: raw row
    counts, duplicates/invalid rows dropped, resample frequency, gap
    handling, and the final aligned row count and date range. Use this
    summary as the data preprocessing section of the analysis.
    """
    mhi_raw, mhi_stats = _load_raw(config.MHI_CSV)
    hhi_raw, hhi_stats = _load_raw(config.HHI_CSV)

    mhi_rs, mhi_rs_stats = _resample(mhi_raw, config.RESAMPLE_FREQ, config.MAX_FFILL_BARS)
    hhi_rs, hhi_rs_stats = _resample(hhi_raw, config.RESAMPLE_FREQ, config.MAX_FFILL_BARS)

    mhi_rs = mhi_rs.add_prefix("MHI_")
    hhi_rs = hhi_rs.add_prefix("HHI_")

    aligned = mhi_rs.join(hhi_rs, how="inner")
    overlap_start = max(mhi_raw.index.min(), hhi_raw.index.min())
    overlap_end = min(mhi_raw.index.max(), hhi_raw.index.max())
    aligned = aligned.loc[(aligned.index >= overlap_start) & (aligned.index <= overlap_end)]
    aligned = aligned.dropna()
    price_cols = [c for c in aligned.columns if not c.endswith("Volume")]
    aligned = aligned[(aligned[price_cols] > 0).all(axis=1)]

    aligned.to_parquet(config.ALIGNED_DATA_PATH)

    summary = {
        "resample_freq": config.RESAMPLE_FREQ,
        "max_forward_fill_bars": config.MAX_FFILL_BARS,
        "mhi_raw": mhi_stats,
        "hhi_raw": hhi_stats,
        "mhi_resampled": mhi_rs_stats,
        "hhi_resampled": hhi_rs_stats,
        "overlap_date_range": [str(overlap_start), str(overlap_end)],
        "aligned_rows": len(aligned),
        "aligned_date_range": [str(aligned.index.min()), str(aligned.index.max())],
        "aligned_data_path": str(config.ALIGNED_DATA_PATH),
        "notes": (
            "No trading-session filter is applied -- MHI and HHI share the same "
            "HKEX trading hours. Data is resampled to "
            f"{config.RESAMPLE_FREQ} bars using OHLC aggregation and short "
            "no-trade gaps were forward-filled with the last traded price "
            "(causal, no lookahead) before aligning the two instruments on "
            "their common timestamp grid and overlapping calendar range."
        ),
    }
    config.DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)
