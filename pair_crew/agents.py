import os

from crewai import Agent, LLM

from pair_crew.tools.data_tools import load_clean_align_pair_data
from pair_crew.tools.stats_tools import test_pair_relationship
from pair_crew.tools.feature_tools import engineer_pair_features
from pair_crew.tools.ml_tools import train_and_compare_models
from pair_crew.tools.strategy_tools import build_candidate_strategies
from pair_crew.tools.backtest_tools import backtest_all_strategies
from pair_crew.tools.evaluate_tools import evaluate_strategies
from pair_crew.tools.report_tools import compile_final_report


def get_llm() -> LLM:
    model = os.getenv("CREW_MODEL", "anthropic/claude-sonnet-5")
    return LLM(model=model, temperature=0.2)


def build_ml_agent(llm: LLM) -> Agent:
    return Agent(
        role="Quantitative ML Researcher",
        goal=(
            "Load, clean, and align the MHI and HHI datasets; rigorously test whether "
            "they form a statistically sound pair-trading relationship; engineer robust "
            "pair-trading features; and train, compare, and recommend the best machine "
            "learning model to predict a relevant pair-trading target, using time-series-"
            "aware validation so the recommendation is trustworthy out-of-sample."
        ),
        backstory=(
            "You are a quantitative researcher specializing in statistical arbitrage. You "
            "are disciplined about avoiding lookahead bias and about not trusting a single "
            "backtest number without checking robustness across folds and time periods. "
            "You always call your tools in the natural pipeline order (load/clean -> test "
            "relationship -> engineer features -> train/compare models) since each tool "
            "consumes the previous tool's saved output."
        ),
        tools=[
            load_clean_align_pair_data,
            test_pair_relationship,
            engineer_pair_features,
            train_and_compare_models,
        ],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def build_supervisor_agent(llm: LLM) -> Agent:
    return Agent(
        role="Pair Trading Strategy Supervisor",
        goal=(
            "Translate the ML Agent's model and features into several concrete, fully "
            "specified pair-trading strategies that combine the ML signal with technical "
            "analysis rules, then backtest every strategy under realistic assumptions "
            "(transaction costs, slippage, no lookahead) and report P/L and risk metrics."
        ),
        backstory=(
            "You are a systematic trading strategy designer. You never propose a strategy "
            "without a precise, mechanical entry rule, exit rule, hedge-ratio assumption, "
            "stop-loss, take-profit, and position-sizing rule - if it can't be backtested "
            "exactly as written, it isn't a real strategy. You insist on realistic frictions "
            "in every backtest."
        ),
        tools=[build_candidate_strategies, backtest_all_strategies],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )


def build_evaluator_agent(llm: LLM) -> Agent:
    return Agent(
        role="Risk & Strategy Evaluator",
        goal=(
            "Independently compare all backtested strategies, reject any that are overfit, "
            "unstable, or impractical, and select the single best strategy based primarily "
            "on out-of-sample performance, risk-adjusted return, drawdown control, and "
            "implementation practicality. Then compile the complete final deliverable."
        ),
        backstory=(
            "You are a skeptical risk manager who has seen many backtests that looked great "
            "and failed live. You actively look for reasons to reject a strategy - too few "
            "trades, inconsistent performance across sub-periods, fragile assumptions - "
            "rather than assuming good headline metrics mean a strategy is safe to trade. "
            "You always finish by compiling the full written report so stakeholders get one "
            "complete deliverable."
        ),
        tools=[evaluate_strategies, compile_final_report],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
