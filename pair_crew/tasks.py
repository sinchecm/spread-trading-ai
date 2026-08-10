from pathlib import Path

from crewai import Agent, Task

from pair_crew import config

_NO_FABRICATION_NOTICE = (
    "You have no data and know no statistics until a tool returns them to you. Never "
    "guess, estimate, extrapolate, or recall a number from general knowledge - every "
    "figure in your final answer must be traceable to an actual tool result from this "
    "run. If you have not yet called the required tools, call them before writing "
    "anything else."
)


def _tool_use_guardrail(run_started_at: float, artifact_paths: list[Path], tool_names: list[str]):
    """Guardrail factory: fails the task if its expected output artifacts were not
    actually (re)written after the run started, which means the agent skipped its
    real tool calls and fabricated the answer instead."""

    def guardrail(output):
        stale = [p for p in artifact_paths if not p.exists() or p.stat().st_mtime < run_started_at]
        if stale:
            stale_names = ", ".join(p.name for p in stale)
            return False, (
                "Guardrail check failed: the following expected output files were never "
                f"written or updated during this run: {stale_names}. This means you did not "
                "actually call your tools - you answered from imagination instead of real "
                "computation. " + _NO_FABRICATION_NOTICE + " Call your tools now, in this "
                f"order: {', '.join(tool_names)}."
            )
        return True, output.raw

    return guardrail


def build_ml_task(agent: Agent, run_started_at: float) -> Task:
    return Task(
        description=(
            "Run the full ML research pipeline on the MHI and HHI CSV files, calling your "
            "tools in this exact order (each depends on the previous tool's saved output):\n"
            "1. load_clean_align_pair_data - load, clean, and align both instruments.\n"
            "2. test_pair_relationship - test cointegration/correlation/mean-reversion to "
            "decide if this pair is suitable for pair trading, and why.\n"
            "3. engineer_pair_features - build spread, returns, rolling correlation, "
            "z-score, volatility features and the three candidate ML targets.\n"
            "4. train_and_compare_models - train and compare several ML models with "
            "time-series-aware validation and select the best model by out-of-sample "
            "performance and robustness (not just the best single fold).\n"
            "After running all four tools, write a clear narrative summary that a Supervisor "
            "Agent (who cannot see your tool outputs directly) can act on: what the data "
            "looked like, whether the pair is suitable and why, what features exist, how the "
            "models compared, and which model/target you recommend and why.\n\n" + _NO_FABRICATION_NOTICE
        ),
        expected_output=(
            "A structured narrative covering: (1) data preprocessing summary, (2) pair "
            "relationship verdict with the key statistics, (3) the engineered feature list "
            "and target definitions, (4) the model comparison table highlights, and (5) the "
            "recommended model + target with its out-of-sample metrics and rationale. Every "
            "number quoted must come from an actual tool result, not be invented."
        ),
        agent=agent,
        guardrail=_tool_use_guardrail(
            run_started_at,
            [
                config.ALIGNED_DATA_PATH,
                config.PAIR_STATS_PATH,
                config.FEATURES_PATH,
                config.MODEL_COMPARISON_PATH,
            ],
            [
                "load_clean_align_pair_data",
                "test_pair_relationship",
                "engineer_pair_features",
                "train_and_compare_models",
            ],
        ),
    )


def build_supervisor_task(agent: Agent, context_tasks: list[Task], run_started_at: float) -> Task:
    return Task(
        description=(
            "Using the ML Agent's recommended model/target, call build_candidate_strategies "
            "to generate several fully-specified pair-trading strategies that combine the ML "
            "signal with technical analysis rules, then call backtest_all_strategies to "
            "backtest every one of them under realistic assumptions (transaction costs, "
            "slippage, no lookahead bias). Report each strategy's rules and its full backtest "
            "metrics (P/L, Sharpe ratio, max drawdown, win rate, profit factor).\n\n" + _NO_FABRICATION_NOTICE
        ),
        expected_output=(
            "A structured narrative covering: (1) each candidate strategy's entry/exit logic, "
            "hedge-ratio assumption, stop-loss, take-profit, and position sizing, and (2) a "
            "comparison table of each strategy's backtested P/L and risk metrics, taken "
            "verbatim from the backtest_all_strategies tool result."
        ),
        agent=agent,
        context=context_tasks,
        guardrail=_tool_use_guardrail(
            run_started_at,
            [config.STRATEGIES_PATH, config.BACKTEST_RESULTS_PATH],
            ["build_candidate_strategies", "backtest_all_strategies"],
        ),
    )


def build_evaluator_task(agent: Agent, context_tasks: list[Task], run_started_at: float) -> Task:
    return Task(
        description=(
            "Call evaluate_strategies to independently compare every backtested strategy, "
            "reject any that are overfit, unstable across sub-periods, unprofitable after "
            "costs, or impractical (too few trades to be meaningful), and select the single "
            "best remaining strategy based primarily on out-of-sample performance, risk-"
            "adjusted return, drawdown control, and implementation practicality. Then call "
            "compile_final_report to produce the complete written deliverable, and return its "
            "full text as your final answer.\n\n" + _NO_FABRICATION_NOTICE
        ),
        expected_output=(
            "The complete final markdown report (verbatim output of compile_final_report), "
            "covering data preprocessing summary, pair relationship analysis, model "
            "comparison, recommended ML model, candidate strategies, backtest results, final "
            "selected strategy, and key limitations."
        ),
        agent=agent,
        context=context_tasks,
        guardrail=_tool_use_guardrail(
            run_started_at,
            [config.EVALUATION_PATH, config.FINAL_REPORT_PATH],
            ["evaluate_strategies", "compile_final_report"],
        ),
    )
