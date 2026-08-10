# MHI / HHI Pair Trading — CrewAI Multi-Agent Workflow

[![GitHub](https://img.shields.io/badge/GitHub-spread--trading--ai-181717?logo=github&logoColor=white)](https://github.com/sinchecm/spread-trading-ai)

A three-agent [CrewAI](https://github.com/crewAIInc/crewAI) workflow that discovers and
evaluates a pair-trading strategy for HKEX Mini-Hang Seng Index (MHI) and Hang Seng China
Enterprises Index (HHI) futures, from raw 1-minute CSVs to a final, risk-screened strategy
recommendation.

For system architecture, tech stack, and data flow, see [`ARCHITECTURE.md`](ARCHITECTURE.md)
(also available as a [rendered page](https://claude.ai/code/artifact/cb41be27-2509-4822-9813-0d7d7365eff5)).

## Agents

| Agent | Job | Tools |
|---|---|---|
| **ML Agent** | Clean/align data, test the pair for cointegration, engineer spread features, train & compare ML models with time-series-aware CV | `load_clean_align_pair_data`, `test_pair_relationship`, `engineer_pair_features`, `train_and_compare_models` |
| **Supervisor Agent** | Turn the ML output into 7 concrete strategies (ML + technical analysis rules, plus three session-anchored grid/fade strategies) and backtest them realistically | `build_candidate_strategies`, `backtest_all_strategies` |
| **Evaluator Agent** | Reject overfit/unstable strategies, pick the final one, compile the report | `evaluate_strategies`, `compile_final_report` |

Every tool does the actual numerical work in plain pandas/sklearn/statsmodels (deterministic,
testable, no lookahead) and writes its artifacts to `outputs/`; the LLM agents call these tools
in sequence and narrate/reason over the results rather than doing arithmetic themselves.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY (or OPENAI_API_KEY + CREW_MODEL)
```

`CREW_MODEL` is a `provider/model` string matching one of CrewAI's native LLM
providers (`anthropic`, `openai`, `azure`, `bedrock`, `gemini`, `snowflake`) or
its `openai_compatible` group (`ollama`, `ollama_chat`, `deepseek`, `openrouter`,
`hosted_vllm`, `cerebras`, `dashscope`) — e.g. `anthropic/claude-sonnet-5`
(default), `openai/gpt-4o`, or `ollama/llama3.1` for a local model.

Before running the full crew, sanity check the model can actually call a tool end-to-end:

```bash
python check_llm_connection.py
```

## Data

There is no data checked into this repo. Add MHI/HHI 1-minute OHLCV CSV files via the
Streamlit uploader (see below) — new uploads are merged into `MHI_1min.csv` /
`HHI_1min.csv` at the repo root (de-duplicated by timestamp, sorted chronologically), which
is what `pair_crew/config.py` points at. Both legs share the same HKEX trading hours, so no
session filter is applied.

## Run

```bash
python main.py
```

This runs the full crew and prints the final markdown report, which is also written to
`outputs/final_report.md`.

## Streamlit UI

A lighter-weight alternative to the CLI for iterating on data/results without touching
Python:

```bash
streamlit run app.py
```

- **Upload data:** add MHI/HHI CSV file(s) from the sidebar; they're merged with the
  existing source CSV rather than replacing it.
- **Run pipeline:** runs the deterministic tool chain (same as "Running the pipeline
  without an LLM" below) with a live progress bar - no API key or LLM call involved.
- **Results:** the compiled report and per-strategy equity curves, plus a download
  button for `final_report.md`.

This does not run the CrewAI agent pipeline (`main.py`) - it's a UI over the same
deterministic tools the agents call, for fast, reproducible iteration.

## Chat frontend (chat_app/)

A second, separate frontend for a desk of traders, alongside (not replacing) the CLI/
Streamlit tools above — those stay untouched for direct pipeline use. Each trader logs in
with their own credentials and describes a strategy in plain language; an LLM translates it
into a small, safe expression DSL (`pair_crew/tools/dsl_tools.py`) built only from
whitelisted variables/operators — no `eval()`/`exec()` of LLM-authored text — which the
trader reviews and confirms or edits. Only then does the **exact same deterministic
backtest engine** used by the CLI/Streamlit pipeline (`run_backtest_for_strategy` in
`pair_crew/tools/backtest_tools.py`) actually run it; the LLM never computes a P&L number
itself. Each trader's strategies and results are private, isolated under
`chat_app/user_data/<username>/`.

Setup (in addition to the base `pip install -r requirements.txt` / `.env` above):

```bash
chainlit create-secret          # paste the output into .env as CHAINLIT_AUTH_SECRET
python chat_app/hash_password.py   # once per trader; paste "username:<hash>" pairs,
                                    # comma-separated, into .env as CHAT_TRADERS
```

Run it (needs `outputs/features.parquet`, `aligned_data.parquet`, and `predictions.parquet`
to already exist — run the pipeline once via `streamlit run app.py` or `python main.py`
first if they don't):

```bash
chainlit run chat_app/app.py -h 0.0.0.0
```

`-h 0.0.0.0` binds it for the office-PC-hosting setup: run it on one machine and the other
traders reach it over the LAN at that machine's IP — no cloud hosting cost. The only
recurring cost is LLM API usage for the translation step, which reuses whatever `CREW_MODEL`
is already configured above (including the `ollama/llama3.1` fallback, at the cost of less
reliable DSL generation than a hosted model).

## Running the pipeline without an LLM

Every stage is a plain Python function under `pair_crew/tools/` and can be run directly
(useful for debugging or CI, with no API key required):

```bash
python -c "
from pair_crew.tools.data_tools import load_clean_align_pair_data
from pair_crew.tools.stats_tools import test_pair_relationship
from pair_crew.tools.feature_tools import engineer_pair_features
from pair_crew.tools.ml_tools import train_and_compare_models
from pair_crew.tools.strategy_tools import build_candidate_strategies
from pair_crew.tools.backtest_tools import backtest_all_strategies
from pair_crew.tools.evaluate_tools import evaluate_strategies
from pair_crew.tools.report_tools import compile_final_report

load_clean_align_pair_data.func()
test_pair_relationship.func()
engineer_pair_features.func()
train_and_compare_models.func()
build_candidate_strategies.func()
backtest_all_strategies.func()
evaluate_strategies.func()
print(compile_final_report.func())
"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage currently focuses on `pair_crew/tools/dsl_tools.py` -- the sandboxed expression
DSL the chat frontend's LLM writes into. Since that's the security boundary between
untrusted LLM-authored text and this process, `tests/test_dsl_tools.py` exhaustively
checks that sandbox-escape attempts (attribute/subscript access, `eval`/`exec`/`__import__`,
comprehensions, walrus, disallowed operators, etc.) are all rejected, alongside operator
semantics, the `compile_strategy_params` entry/exit contract, the HKT hour-of-day
conversion (including the midnight-wrap edge case), and the plausibility-warning
heuristics.

## Outputs

All intermediate and final artifacts land in `outputs/`:

- `aligned_data.parquet`, `data_preprocessing_summary.json`
- `pair_stats.json` — cointegration/ADF/half-life/Hurst results
- `features.parquet`, `feature_engineering_summary.json`
- `model_comparison.json`, `best_model.joblib`, `best_model_meta.json`, `predictions.parquet`
- `strategies.json` — the 4 candidate strategy specs
- `backtest_results.json`, `equity_<strategy_id>.csv` — one equity curve per strategy
- `evaluation.json` — rejection reasons, composite scores, final pick
- `final_report.md` — the full deliverable

## Contract spec (fixed, not estimated)

- Hedge ratio: 2 MHI lots : 1 HHI lot (`config.MHI_HEDGE_LOTS` / `config.HHI_HEDGE_LOTS`)
- Contract multiplier: HKD 10/index point (MHI), HKD 50/index point (HHI) —
  `config.MHI_CONTRACT_MULTIPLIER` / `config.HHI_CONTRACT_MULTIPLIER`, used in every $
  calculation (P&L, cointegration residuals, feature spread, position sizing)
- Tick size: 1 index point, both legs (HKEX spec)
- Commission: HKD 6/lot (MHI), HKD 7.5/lot (HHI), per side —
  `config.MHI_COMMISSION_PER_LOT` / `config.HHI_COMMISSION_PER_LOT`
- Slippage: disabled, 0 ticks per leg, per side (`config.SLIPPAGE_TICKS`) — costs are commission-only
- `config.MHI_SPREAD_UI_MULTIPLIER` / `HHI_SPREAD_UI_MULTIPLIER` (2 / 5) are kept only as a
  reference to the trading platform's own spread-price display convention — they are **not**
  used in any $ calculation in this pipeline.

## Design notes / where the rigor lives

- **No lookahead:** ML targets are forward-looking labels only (never features); backtest
  signals are computed on bar closes and filled on the *next* bar's open; embargoed
  walk-forward CV purges the training fold's tail so forward-looking labels can't leak across
  the train/validation split.
- **Time-series-aware validation:** `train_and_compare_models` uses an expanding-window,
  embargoed `TimeSeriesSplit`, not k-fold, and a single untouched out-of-sample holdout
  (last ~35% of history) that is only ever scored once, after model selection.
- **Real contract economics, not a proxy:** the hedge ratio is the fixed HKEX spread contract
  spec (2 MHI lots : 1 HHI lot, each at its own contract multiplier), not a rolling-estimated
  regression beta, so it carries no lookahead risk and needs no shifting. Backtest P&L is
  computed directly in real HKD from that fixed ratio (`config.spread_dollar_value` /
  `spread_unit_notional`), not a log-return notional proxy. Transaction costs use real
  per-lot commission plus tick-based slippage per leg (`config.spread_transaction_cost`),
  not a bps-of-notional proxy.
- **Realistic backtesting:** every entry/exit pays commission + slippage on both legs;
  position size is volatility-scaled (in real $ spread volatility), not fixed lot size.
- **Overfitting screen:** the Evaluator reruns each strategy on the first/second half of the
  OOS window independently and rejects strategies with too few trades, negative Sharpe,
  profit factor < 1, excessive drawdown, or inconsistent sign of returns across the two halves.
- **Tool-use guardrails:** each of the three tasks (`pair_crew/tasks.py`) has a `guardrail`
  that checks, after the agent answers, whether that stage's expected output files were
  actually (re)written during this run. If they weren't, the agent's answer is rejected and
  CrewAI automatically retries the task with an explicit message telling it which tools to
  call, in order — this catches an agent that skips its tools and fabricates a report instead
  of computing it.

See `outputs/final_report.md` (Section 11, Key Limitations) for the honest caveats once a run
has completed.
