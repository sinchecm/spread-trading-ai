"""Streamlit control panel for the MHI/HHI pair-trading pipeline.

Upload new data, run the deterministic (no-LLM) pipeline, and view the
resulting report - all without touching the CLI.

Usage:
    streamlit run app.py
"""
import copy
import json
import sys
import time
import traceback

import pandas as pd
import streamlit as st

from pair_crew import config
from pair_crew.tools.backtest_tools import (
    backtest_all_strategies,
    build_backtest_dataframe,
    run_backtest_for_strategy,
)
from pair_crew.tools.data_tools import load_clean_align_pair_data
from pair_crew.tools.evaluate_tools import evaluate_strategies
from pair_crew.tools.feature_tools import engineer_pair_features
from pair_crew.tools.ml_tools import train_and_compare_models
from pair_crew.tools.report_tools import compile_final_report
from pair_crew.tools.stats_tools import test_pair_relationship
from pair_crew.tools.strategy_tools import build_candidate_strategies

GRID_STRATEGY_IDS = [
    "afternoon_grid_fade",
    "full_day_grid_fade",
    "morning_grid_fade",
    "mid_morning_full_day_grid_fade",
    "mid_morning_grid_fade",
]
GRID_SWEEP_ENTRY_TICKS = [100, 125, 150, 175, 200, 225, 250]

st.set_page_config(page_title="MHI/HHI Pair Trading Pipeline", layout="wide")
st.title("MHI / HHI Pair Trading — Pipeline Control Panel")
st.caption(
    "Upload data, run the full deterministic pipeline (no LLM/API key needed), "
    "and review the report below."
)

PIPELINE_STEPS = [
    ("Load, clean & align data", load_clean_align_pair_data),
    ("Test pair relationship", test_pair_relationship),
    ("Engineer features", engineer_pair_features),
    ("Train & compare models", train_and_compare_models),
    ("Build candidate strategies", build_candidate_strategies),
    ("Backtest strategies", backtest_all_strategies),
    ("Evaluate strategies", evaluate_strategies),
    ("Compile final report", compile_final_report),
]


def _merge_and_save(uploaded_files, target_path) -> tuple[int, str, str, int]:
    """Concatenate the target CSV (if it exists) with all uploaded CSVs,
    normalize the volume column name, de-duplicate by timestamp, sort
    chronologically, and overwrite the target file. Returns (row_count,
    start, end, rows_with_no_volume_data)."""
    frames = []
    if target_path.exists():
        frames.append(pd.read_csv(target_path))
    for f in uploaded_files:
        frames.append(pd.read_csv(f))

    normalized = []
    n_missing_volume = 0
    for df in frames:
        vol_cols = [c for c in df.columns if "Volume" in c]
        if vol_cols:
            df = df.rename(columns={vol_cols[0]: "Volume"})
        else:
            df = df.copy()
            df["Volume"] = 0.0
            n_missing_volume += len(df)
        normalized.append(df[["Timestamp (UTC)", "Open", "High", "Low", "Close", "Volume"]])

    merged = pd.concat(normalized, ignore_index=True)
    merged["Timestamp (UTC)"] = pd.to_datetime(merged["Timestamp (UTC)"])
    merged = merged.sort_values("Timestamp (UTC)").drop_duplicates(
        subset="Timestamp (UTC)", keep="last"
    )
    merged.to_csv(target_path, index=False)
    return (
        len(merged),
        str(merged["Timestamp (UTC)"].min()),
        str(merged["Timestamp (UTC)"].max()),
        n_missing_volume,
    )


# --- Sidebar: data upload ---
st.sidebar.header("1. Data")

st.sidebar.caption(
    f"**MHI source:** `{config.MHI_CSV.name}`"
    + (f" ({sum(1 for _ in open(config.MHI_CSV)) - 1} rows)" if config.MHI_CSV.exists() else " (missing)")
)
st.sidebar.caption(
    f"**HHI source:** `{config.HHI_CSV.name}`"
    + (f" ({sum(1 for _ in open(config.HHI_CSV)) - 1} rows)" if config.HHI_CSV.exists() else " (missing)")
)

mhi_files = st.sidebar.file_uploader(
    "Add MHI CSV file(s)", type="csv", accept_multiple_files=True, key="mhi_upload"
)
hhi_files = st.sidebar.file_uploader(
    "Add HHI CSV file(s)", type="csv", accept_multiple_files=True, key="hhi_upload"
)
st.sidebar.caption(
    "New files are merged with the existing source CSV (de-duplicated by "
    "timestamp, sorted chronologically), not run standalone."
)

if st.sidebar.button("Save & merge uploaded files", disabled=not (mhi_files or hhi_files)):
    try:
        if mhi_files:
            n, start, end, n_no_vol = _merge_and_save(mhi_files, config.MHI_CSV)
            st.sidebar.success(f"MHI merged: {n} rows, {start} → {end}")
            if n_no_vol:
                st.sidebar.warning(f"{n_no_vol} MHI rows had no Volume column - defaulted to 0.")
        if hhi_files:
            n, start, end, n_no_vol = _merge_and_save(hhi_files, config.HHI_CSV)
            st.sidebar.success(f"HHI merged: {n} rows, {start} → {end}")
            if n_no_vol:
                st.sidebar.warning(f"{n_no_vol} HHI rows had no Volume column - defaulted to 0.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Merge failed: {e}")

st.sidebar.divider()
st.sidebar.header("2. Fixed contract spec")
st.sidebar.caption(
    f"{config.MHI_HEDGE_LOTS} MHI lots (×HKD{config.MHI_CONTRACT_MULTIPLIER}/pt) : "
    f"{config.HHI_HEDGE_LOTS} HHI lot(s) (×HKD{config.HHI_CONTRACT_MULTIPLIER}/pt) "
    f"— ratio {config.FIXED_HEDGE_RATIO:.4f}. Edit `pair_crew/config.py` to change."
)
st.sidebar.caption(
    f"Commission: HKD{config.MHI_COMMISSION_PER_LOT}/lot (MHI), HKD{config.HHI_COMMISSION_PER_LOT}/lot "
    f"(HHI), per side. Slippage: {config.SLIPPAGE_TICKS} tick/leg/side."
)

st.sidebar.divider()
st.sidebar.header("3. Fund size")
fund_size = st.sidebar.number_input(
    "Fund size (HKD)", min_value=1000.0, value=float(config.INITIAL_CAPITAL), step=1000.0
)
st.sidebar.caption("Sets INITIAL_CAPITAL for every strategy in the pipeline run below.")

# --- Main: run pipeline ---
st.header("4. Run pipeline")
run = st.button("Run full pipeline", type="primary")

if run:
    config.INITIAL_CAPITAL = fund_size
    progress = st.progress(0.0)
    status = st.empty()
    ok = True
    n_steps = len(PIPELINE_STEPS)
    for i, (label, fn) in enumerate(PIPELINE_STEPS):
        status.info(f"Running: {label}…")
        print(f"[pipeline] ({i + 1}/{n_steps}) START: {label}", flush=True)
        t0 = time.time()
        try:
            fn.func()
        except Exception:
            tb = traceback.format_exc()
            print(f"[pipeline] ({i + 1}/{n_steps}) FAILED: {label} after {time.time() - t0:.1f}s\n{tb}", file=sys.stderr, flush=True)
            status.error(f"Failed at: {label}")
            st.code(tb)
            ok = False
            break
        print(f"[pipeline] ({i + 1}/{n_steps}) DONE: {label} ({time.time() - t0:.1f}s)", flush=True)
        progress.progress((i + 1) / n_steps)
    if ok:
        status.success("Pipeline complete.")

# --- Results ---
if config.FINAL_REPORT_PATH.exists():
    st.header("5. Results")
    tab_report, tab_equity, tab_grid_sweep = st.tabs(
        ["Final report", "Equity curves", "Grid Threshold Sweep"]
    )

    with tab_report:
        report_text = config.FINAL_REPORT_PATH.read_text()
        st.download_button("Download final_report.md", report_text, file_name="final_report.md")
        st.markdown(report_text)

    with tab_equity:
        if config.BACKTEST_RESULTS_PATH.exists():
            results = json.loads(config.BACKTEST_RESULTS_PATH.read_text())["results"]
            curves = {}
            for sid in results:
                csv_path = config.OUTPUT_DIR / f"equity_{sid}.csv"
                if csv_path.exists():
                    series = pd.read_csv(csv_path, index_col=0, parse_dates=True)["equity"]
                    curves[sid] = series.resample("1D").last().dropna()
            if curves:
                st.caption(
                    "Downsampled to one point per day for chart performance - full "
                    "minute-bar resolution is in outputs/equity_<strategy_id>.csv."
                )
                st.line_chart(pd.DataFrame(curves))
            st.dataframe(pd.DataFrame(results).T)
        else:
            st.info("Run the pipeline to generate backtest results.")

    with tab_grid_sweep:
        st.caption(
            f"Entry-tick threshold sweep (100-250 ticks, 25-tick steps) for the "
            f"{len(GRID_STRATEGY_IDS)} session-anchored grid/fade strategies, backtested over "
            "the full out-of-sample window. The row matching each strategy's current default "
            "threshold is highlighted."
        )
        if not config.STRATEGIES_PATH.exists():
            st.info("Run the pipeline to generate strategies.json before viewing this tab.")
        else:
            all_strategies = {s["id"]: s for s in json.loads(config.STRATEGIES_PATH.read_text())}
            grid_bases = {sid: all_strategies[sid] for sid in GRID_STRATEGY_IDS if sid in all_strategies}
            if not grid_bases:
                st.info("No session-grid strategies found in strategies.json.")
            else:
                if st.button("Run threshold sweep", key="run_grid_sweep"):
                    with st.spinner(f"Backtesting {len(grid_bases) * len(GRID_SWEEP_ENTRY_TICKS)} threshold/strategy combinations..."):
                        bt_df = build_backtest_dataframe()
                        sweep_results = {}
                        for sid, base in grid_bases.items():
                            rows = []
                            for entry_ticks in GRID_SWEEP_ENTRY_TICKS:
                                strat = copy.deepcopy(base)
                                strat["params"]["entry_ticks"] = entry_ticks
                                m, _ = run_backtest_for_strategy(strat, bt_df)
                                rows.append(
                                    {
                                        "entry_ticks": entry_ticks,
                                        "n_trades": m["n_trades"],
                                        "win_rate_%": round(m["win_rate"] * 100, 1),
                                        "profit_factor": m["profit_factor"],
                                        "total_return_%": m["total_return_pct"],
                                        "sharpe": m["sharpe_ratio_annualized"],
                                        "max_dd_%": m["max_drawdown_pct"],
                                        "calmar": m["calmar_ratio"],
                                    }
                                )
                            sweep_results[sid] = pd.DataFrame(rows).set_index("entry_ticks")
                        st.session_state["grid_sweep_results"] = sweep_results

                sweep_results = st.session_state.get("grid_sweep_results")
                if sweep_results is None:
                    st.info("Click **Run threshold sweep** to backtest each strategy across the 100-250 tick range.")
                else:
                    for sid, base in grid_bases.items():
                        if sid not in sweep_results:
                            continue
                        default_ticks = base["params"]["entry_ticks"]
                        st.subheader(f"{base['name']} (`{sid}`)")
                        st.caption(f"Current default: **{default_ticks} ticks**")
                        df = sweep_results[sid]
                        styled = df.style.apply(
                            lambda row: [
                                "background-color: #2e7d32; color: white" if row.name == default_ticks else ""
                                for _ in row
                            ],
                            axis=1,
                        )
                        st.dataframe(styled, width="stretch")

                st.divider()
                st.subheader("Lot-count breakdown (at each strategy's current default threshold)")
                if config.BACKTEST_RESULTS_PATH.exists():
                    backtest_results = json.loads(config.BACKTEST_RESULTS_PATH.read_text())["results"]
                    lot_rows = []
                    for sid in GRID_STRATEGY_IDS:
                        if sid in backtest_results and "trades_by_max_lots" in backtest_results[sid]:
                            by_lots = backtest_results[sid]["trades_by_max_lots"]
                            lot_rows.append(
                                {
                                    "strategy": sid,
                                    "entry_ticks": grid_bases[sid]["params"]["entry_ticks"] if sid in grid_bases else None,
                                    "1 lot": by_lots.get("1", 0),
                                    "2 lots": by_lots.get("2", 0),
                                    "3 lots": by_lots.get("3", 0),
                                }
                            )
                    if lot_rows:
                        st.dataframe(pd.DataFrame(lot_rows).set_index("strategy"), width="stretch")
                    else:
                        st.info("No lot-count breakdown available - run the pipeline's backtest step first.")
                else:
                    st.info("Run the pipeline to generate backtest_results.json.")
else:
    st.info("Run the pipeline to generate a report.")
