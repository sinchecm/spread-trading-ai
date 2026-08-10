"""Chat-box strategy frontend for the MHI/HHI pair-trading desk.

A second, separate frontend alongside app.py/main.py (which stay
untouched, for the existing direct-use pipeline). Each of up to 6 traders
logs in with their own credentials (chat_app/auth.py) and describes a
strategy in plain language; the LLM (chat_app/translate.py) translates it
into a safe expression DSL (pair_crew/tools/dsl_tools.py) which the trader
confirms or edits, then the exact same deterministic backtest engine the
main pipeline uses (pair_crew/tools/backtest_tools.py) runs it -- the LLM
never computes a P&L number itself. Results are private to each trader
(chat_app/storage.py).

Usage:
    chainlit run chat_app/app.py
"""
import sys
from pathlib import Path

# chainlit loads this file directly (its own dir on sys.path, not the repo
# root), so package-relative imports below need the repo root added by hand
# -- this makes `chainlit run chat_app/app.py` work regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import chainlit as cl

import chat_app.auth  # noqa: F401  (registers the password_auth_callback)
from chat_app import storage
from chat_app.translate import (
    _GRID_ANCHOR_LABEL,
    StrategyTranslation,
    build_strategy_spec,
    translate_and_validate,
)
from pair_crew import config as pipeline_config
from pair_crew.tools import dsl_tools
from pair_crew.tools.backtest_tools import build_backtest_dataframe, run_backtest_for_strategy

CONFIRM_WORDS = {"confirm", "yes", "y", "go", "run it", "backtest it", "looks good", "confirmed", "do it"}
CANCEL_WORDS = {"cancel", "no", "discard", "nevermind", "never mind", "stop"}

_PIPELINE_OUTPUTS_MISSING = (
    "The shared pipeline outputs (features/model predictions) don't exist yet. "
    "Ask an admin to run the main pipeline once (`streamlit run app.py`, or `python main.py`) "
    "so there's data for chat strategies to backtest against."
)


def _pipeline_outputs_ready() -> bool:
    return (
        pipeline_config.FEATURES_PATH.exists()
        and pipeline_config.ALIGNED_DATA_PATH.exists()
        and pipeline_config.PREDICTIONS_PATH.exists()
    )


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("stage", "idle")
    cl.user_session.set("conversation", [])
    cl.user_session.set("pending_translation", None)

    user = cl.user_session.get("user")
    greeting = (
        f"Welcome, **{user.identifier}**. Describe a pair-trading strategy in plain "
        "language and I'll turn it into a backtestable rule for you to confirm.\n\n"
        "Example: *\"Short the spread when the z-score is above 2 and correlation is "
        "above 0.2, cover when the z-score drops back below 0.5.\"*\n\n"
        "Other commands: `list` (your saved strategies), `help`."
    )
    if not _pipeline_outputs_ready():
        greeting += f"\n\n:warning: {_PIPELINE_OUTPUTS_MISSING}"
    await cl.Message(content=greeting).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()
    if not text:
        return

    user = cl.user_session.get("user")
    username = user.identifier
    stage = cl.user_session.get("stage", "idle")
    lowered = text.lower().strip()

    if stage == "confirming":
        await _handle_confirmation(username, text, lowered)
        return

    if stage == "collecting":
        conversation = cl.user_session.get("conversation", [])
        conversation.append({"role": "user", "content": text})
        await _run_translation(conversation)
        return

    # idle
    if lowered in ("list", "my strategies", "show my strategies", "show strategies"):
        await _list_strategies(username)
        return
    if lowered in ("help", "?"):
        await _send_help()
        return

    await _run_translation([{"role": "user", "content": text}])


async def _run_translation(conversation: list[dict]) -> None:
    async with cl.Step(name="Translating to strategy DSL", type="run"):
        try:
            result = await cl.make_async(translate_and_validate)(conversation)
        except Exception as e:  # LLM/network error -- don't lose the trader's turn
            await cl.Message(
                content=f"Sorry, translation failed ({e}). Please try describing it again."
            ).send()
            return

    conversation = [*conversation, {"role": "assistant", "content": result.model_dump_json()}]
    cl.user_session.set("conversation", conversation)

    if not result.understood:
        cl.user_session.set("stage", "collecting")
        await cl.Message(
            content=result.clarifying_question or "Could you clarify your strategy idea?"
        ).send()
        return

    cl.user_session.set("stage", "confirming")
    cl.user_session.set("pending_translation", result.model_dump())
    await cl.Message(content=_format_draft(result)).send()


async def _handle_confirmation(username: str, text: str, lowered: str) -> None:
    if lowered in CANCEL_WORDS:
        _reset_session()
        await cl.Message(content="Discarded. Describe a new strategy whenever you're ready.").send()
        return

    if lowered in CONFIRM_WORDS:
        pending = cl.user_session.get("pending_translation")
        result = StrategyTranslation.model_validate(pending)
        await _run_backtest(username, result)
        _reset_session()
        return

    # anything else is treated as an edit instruction on the current draft
    conversation = cl.user_session.get("conversation", [])
    conversation.append({"role": "user", "content": f"Revise the previous draft: {text}"})
    await _run_translation(conversation)


def _reset_session() -> None:
    cl.user_session.set("stage", "idle")
    cl.user_session.set("conversation", [])
    cl.user_session.set("pending_translation", None)


def _format_draft(result: StrategyTranslation) -> str:
    lines = [f"**{result.name or 'Draft strategy'}**", "", result.plain_explanation or "", "", "```"]
    if result.is_grid:
        anchor_label = _GRID_ANCHOR_LABEL[result.grid_anchor]
        scope_label = {
            "full_day": "full day (through lunch to 16:30 HKT)",
            "morning_only": "morning only (flat by 12:30 HKT)",
            "through_night": "full day + night session (flat by 23:00 HKT)",
        }[result.grid_session_scope]
        lines.append(f"anchor:       {anchor_label}, {scope_label}")
        lines.append(f"entry:        {result.grid_entry_ticks} ticks from anchor")
        lines.append(f"add:          +1 lot every {result.grid_add_ticks} further ticks, up to {result.grid_max_lots} lots")
        lines.append(f"take_profit:  {result.grid_take_profit_ticks} ticks favorable from avg entry")
        if result.grid_stop_loss_dollars is not None:
            lines.append(f"stop_loss:    HKD {result.grid_stop_loss_dollars:,.0f} unrealized loss from avg entry")
        else:
            lines.append(f"stop_loss:    {result.grid_stop_loss_ticks} ticks adverse from avg entry")
    else:
        if result.entry_long_expr:
            lines.append(f"entry_long:  {result.entry_long_expr}")
        if result.entry_short_expr:
            lines.append(f"entry_short: {result.entry_short_expr}")
        lines.append(f"exit:        {result.exit_expr}")
        if result.stop_expr:
            lines.append(f"stop:        {result.stop_expr}")
        if result.max_holding_bars:
            lines.append(f"max_holding: {result.max_holding_bars} bars (~{result.max_holding_bars} min)")
    lines.append("```")
    lines.append("Reply **confirm** to backtest this, describe a change to edit it, or **cancel** to discard.")
    return "\n".join(lines)


def _run_backtest_sync(spec: dict) -> tuple[dict, pd.Series, list[str]]:
    bt_df = build_backtest_dataframe()
    warnings: list[str] = []
    if spec["engine"] == "dsl_expression":
        compiled = dsl_tools.compile_strategy_params(spec["params"])
        warnings = dsl_tools.plausibility_warnings(compiled, bt_df)
    metrics, equity_curve = run_backtest_for_strategy(spec, bt_df)
    return metrics, equity_curve, warnings


async def _run_backtest(username: str, result: StrategyTranslation) -> None:
    if not _pipeline_outputs_ready():
        await cl.Message(content=_PIPELINE_OUTPUTS_MISSING).send()
        return

    slug = storage.unique_slug(username, result.name or "strategy")
    spec = build_strategy_spec(f"{username}_{slug}", result)

    async with cl.Step(name="Backtesting (out-of-sample)", type="run"):
        try:
            metrics, equity_curve, warnings = await cl.make_async(_run_backtest_sync)(spec)
        except dsl_tools.SafeExpressionError as e:
            await cl.Message(content=f"Couldn't backtest that: {e}").send()
            return
        except Exception as e:
            await cl.Message(content=f"Backtest failed: {e}").send()
            return

    storage.save_strategy(username, slug, spec, result.plain_explanation or "")
    storage.save_backtest_result(username, slug, metrics, equity_curve)

    content = _format_metrics(spec, metrics)
    if warnings:
        content += "\n\n:rotating_light: **Plausibility check flagged this strategy:**\n"
        content += "\n".join(f"- {w}" for w in warnings)
        content += (
            "\n\nThe numbers above are still what this exact rule produced, but they're "
            "likely meaningless -- describe a fix and I'll retranslate it."
        )

    await cl.Message(
        content=content,
        elements=[_equity_chart(equity_curve, spec["name"])],
    ).send()


def _format_metrics(spec: dict, metrics: dict) -> str:
    pf = metrics["profit_factor"]
    calmar = metrics["calmar_ratio"]
    lines = [
        f"**Backtest result: {spec['name']}** (out-of-sample, saved as `{spec['id']}`)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Trades | {metrics['n_trades']} |",
        f"| Win rate | {metrics['win_rate'] * 100:.1f}% |",
        f"| Profit factor | {pf if pf is not None else 'n/a'} |",
        f"| Total return | {metrics['total_return_pct']:.2f}% |",
        f"| Sharpe (annualized) | {metrics['sharpe_ratio_annualized']:.2f} |",
        f"| Max drawdown | {metrics['max_drawdown_pct']:.2f}% |",
        f"| Calmar | {calmar if calmar is not None else 'n/a'} |",
        f"| Avg holding (bars) | {metrics['avg_holding_bars']:.1f} |",
        f"| Total costs paid | HKD {metrics['total_transaction_costs']:,.2f} |",
    ]
    if "trades_by_max_lots" in metrics:
        by_lots = ", ".join(f"{k} lot(s): {v}" for k, v in metrics["trades_by_max_lots"].items())
        lines.append(f"| Cycles by max lots reached | {by_lots} |")
    if metrics["n_trades"] < 8:
        lines.append("")
        lines.append(
            f":warning: Only {metrics['n_trades']} trades in the out-of-sample window -- "
            "too few to draw a statistically reliable conclusion, even if the headline "
            "numbers look good."
        )
    return "\n".join(lines)


def _equity_chart(equity_curve: pd.Series, name: str) -> cl.Pyplot:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    equity_curve.plot(ax=ax)
    ax.set_title(f"{name} -- equity curve (out-of-sample)")
    ax.set_ylabel("Equity (HKD)")
    fig.tight_layout()
    return cl.Pyplot(name="equity_curve", figure=fig)


async def _list_strategies(username: str) -> None:
    records = storage.list_strategies(username)
    if not records:
        await cl.Message(content="You haven't saved any strategies yet.").send()
        return
    lines = [f"**Your saved strategies ({len(records)}):**", ""]
    for r in records:
        strat = r["strategy"]
        lines.append(f"- **{strat['name']}** (`{r['slug']}`) -- {r.get('description') or strat.get('entry_logic', '')}")
    await cl.Message(content="\n".join(lines)).send()


async def _send_help() -> None:
    await cl.Message(
        content=(
            "Describe a strategy in plain language, e.g. *\"go long when the z-score is "
            "below -2 and the ML score is above 0.6, exit when the z-score is back above "
            "-0.5\"*. I'll translate it, you confirm or edit it, then it gets backtested "
            "on the shared out-of-sample data.\n\n"
            "Commands: `list` (your saved strategies), `help`.\n"
            "While reviewing a draft: `confirm` to backtest, `cancel` to discard, or just "
            "describe what to change."
        )
    ).send()
