"""Plain-language -> DSL translation for the chat frontend.

The LLM here only ever produces expression *strings* over dsl_tools'
whitelisted variables -- it never computes a backtest number itself. Every
translation is compiled through pair_crew.tools.dsl_tools before it's
accepted, so a malformed or unsafe expression is rejected (and the LLM
asked to retry) before it ever reaches the backtest engine.

Uses crewai's LLM class (same CREW_MODEL env var as pair_crew/agents.py),
so whichever provider is already configured for the deterministic pipeline
-- including the $0 ollama/llama3.1 fallback -- also powers this chat.
"""
import json
import os
import re
from typing import Literal, Optional

from crewai import LLM
from pydantic import BaseModel, Field

from pair_crew import config
from pair_crew.tools import dsl_tools

_SYSTEM_PROMPT = f"""You are a strategy-translation assistant for a desk of MHI/HHI \
index-futures pair traders. A trader describes a trading idea in plain \
language; your only job is to translate it into a small, safe expression \
DSL so it can be mechanically backtested. You never estimate performance \
or make claims about profitability yourself -- a deterministic backtester \
does that after the trader confirms your translation.

{dsl_tools.describe_dsl_for_prompt()}

Rules:
- If the trader's request is specific enough to fully determine entry, \
exit, and (implicitly or explicitly) risk control, set understood=true and \
fill in the expression fields.
- entry_long_expr / entry_short_expr: at least one must be set. Leave the \
other unset (null) if the trader only described a one-directional strategy.
- exit_expr is REQUIRED -- every strategy must have a concrete exit rule. \
If the trader didn't specify one, propose a sensible default based on what \
they described (e.g. the reversion target implied by their entry rule) and \
say so plainly in plain_explanation, rather than leaving it unset.
- stop_expr is optional but recommended; max_holding_bars is optional (a \
bar is ~1 minute).
- If the request is too vague to translate (e.g. missing a concrete \
threshold, unclear which direction, contradictory), set understood=false \
and ask exactly ONE clarifying_question -- the single most important \
missing detail, in plain language a discretionary trader would use (not \
DSL syntax).
- When understood=true, also fill plain_explanation: 1-2 plain-English \
sentences restating exactly what the expressions will do, for the trader \
to confirm before anything is backtested.
- Only ever use the variables/operators listed above. Do not invent new \
variables or functions.
- If the trader describes a rule in terms of distance from a session anchor \
(e.g. "N points below/above the morning open/reopen") WITHOUT also describing \
a multi-lot ladder (adding lots as price extends further), treat it as a \
single-position DSL request: this DSL's boolean/arithmetic expressions have \
NO anchor-relative variable, so do NOT approximate it by comparing the raw \
`spread` variable to a small constant (its real range is tens of thousands of \
HKD, so a small threshold would fire on essentially every bar). Either ask a \
clarifying_question explaining this limitation and proposing the closest \
available alternative (spread_zscore, spread_roc, or spread_ma_crossover), or \
use one of those directly if it's a reasonable match -- and say so plainly in \
plain_explanation.

GRID MODE -- a separate, structured strategy type (set is_grid=true instead \
of filling the entry/exit expression fields) for session-anchored, multi-lot \
ladder ("grid"/"fade") strategies: enter 1 lot when price first moves \
grid_entry_ticks away from a session anchor price (fading the move, i.e. \
betting on reversion back toward the anchor), add 1 more lot every further \
grid_add_ticks of adverse extension up to grid_max_lots lots total, take-profit \
by closing everything once price moves grid_take_profit_ticks in favor of the \
average entry price, and stop-loss by closing everything once the position's \
loss versus its average entry price reaches a threshold expressed EITHER in \
dollars (grid_stop_loss_dollars, e.g. 4000 for a HKD 4,000 unrealized loss) OR \
in ticks (grid_stop_loss_ticks, the tick-denominated equivalent of \
grid_take_profit_ticks but in the adverse direction) -- use whichever unit the \
trader actually specified; if they said "$X" or "HKDX" use grid_stop_loss_dollars, \
if they said a tick/point count use grid_stop_loss_ticks. Use grid mode whenever the trader \
describes adding/scaling into a position as it moves further from a reference \
point (an opening/reopening price) -- this is NOT expressible as a DSL \
expression and must use grid mode instead.
- Required grid fields when is_grid=true: grid_anchor ("morning_open" for the \
9:15 HKT day-session open, "mid_morning_open" for the 9:30 HKT price 15 minutes \
into the day session, or "afternoon_reopen" for the 13:00 HKT lunch \
reopen), grid_session_scope ("full_day" = carries through the lunch break to \
the 16:30 HKT close, "morning_only" = force-flattened at the 12:30 HKT lunch \
break, or "through_night" = carries through the lunch break AND the day/night \
session gap, force-flattened at 23:00 HKT -- use this whenever the trader asks \
to hold into the night session or gives a flatten time later than 16:30 HKT, \
e.g. "flat by 11pm"), grid_entry_ticks, grid_add_ticks, grid_max_lots (all positive \
integers), grid_take_profit_ticks (positive integer), and EXACTLY ONE of \
grid_stop_loss_dollars (a positive $ amount) or grid_stop_loss_ticks (a \
positive tick count) -- a stop-loss is REQUIRED, ask for it if the trader \
didn't specify one; never leave a grid strategy without one, and never set \
both stop fields at once. When is_grid=true, leave \
entry_long_expr/entry_short_expr/exit_expr/stop_expr/max_holding_bars unset.
"""


class StrategyTranslation(BaseModel):
    understood: bool = Field(
        description="True only if a complete, backtestable strategy (entry + exit) could be produced."
    )
    clarifying_question: Optional[str] = Field(
        default=None, description="Set when understood=false: the one question to ask the trader next."
    )
    name: Optional[str] = Field(default=None, description="Short human-readable strategy name.")
    entry_long_expr: Optional[str] = None
    entry_short_expr: Optional[str] = None
    exit_expr: Optional[str] = None
    stop_expr: Optional[str] = None
    max_holding_bars: Optional[int] = None
    plain_explanation: Optional[str] = Field(
        default=None, description="Plain-English restatement of the rule, for the trader to confirm."
    )

    is_grid: bool = Field(
        default=False,
        description="True for a session-anchored, multi-lot ladder ('grid'/'fade') strategy -- see GRID MODE.",
    )
    grid_anchor: Optional[Literal["morning_open", "mid_morning_open", "afternoon_reopen"]] = None
    grid_session_scope: Optional[Literal["full_day", "morning_only", "through_night"]] = None
    grid_entry_ticks: Optional[int] = None
    grid_add_ticks: Optional[int] = None
    grid_max_lots: Optional[int] = None
    grid_take_profit_ticks: Optional[int] = None
    grid_stop_loss_dollars: Optional[float] = Field(
        default=None, description="Set this OR grid_stop_loss_ticks (exactly one), not both."
    )
    grid_stop_loss_ticks: Optional[int] = Field(
        default=None, description="Set this OR grid_stop_loss_dollars (exactly one), not both."
    )


def _get_llm() -> LLM:
    model = os.getenv("CREW_MODEL", "anthropic/claude-sonnet-5")
    return LLM(model=model, temperature=0.1)


def _params_from_translation(result: StrategyTranslation) -> dict:
    return {
        "entry_long_expr": result.entry_long_expr,
        "entry_short_expr": result.entry_short_expr,
        "exit_expr": result.exit_expr,
        "stop_expr": result.stop_expr,
        "max_holding_bars": result.max_holding_bars,
    }


_GRID_CUE_RE = re.compile(
    r"add\s+(?:1\s+|one\s+|another\s+)?lots?|cap(?:ped)?\s+(?:at\s+)?\d+\s*lots?|"
    r"up\s+to\s+\d+\s*lots?|\bladder\b|\bgrid\b|\bscale\s+(?:in|up)\b|max(?:imum)?\s+\d+\s*lots?",
    re.IGNORECASE,
)

_GRID_HINT = (
    "Note: this describes adding/scaling lots progressively with a lot cap -- that is a "
    "GRID MODE strategy (is_grid=true), not a DSL boolean-expression strategy. Do not ask "
    "about session-anchor DSL limitations for it; use the grid_* fields instead, filling in "
    "your best interpretation of any unstated details (e.g. default session scope to "
    "full_day) rather than asking a clarifying question, unless something is genuinely "
    "ambiguous between two very different strategies."
)


def _looks_like_grid_request(conversation: list[dict]) -> bool:
    """Cheap, deterministic pre-check for grid/ladder language, used to
    proactively steer the LLM (see _GRID_HINT) rather than relying solely on
    its own judgment -- the small/cheap models this chat can be configured
    with (e.g. gemini-flash-lite) are not fully consistent from one call to
    the next at classifying this correctly, as observed in practice: the
    exact same trader request was sometimes correctly routed to grid mode
    and sometimes not."""
    combined = " ".join(m["content"] for m in conversation if m.get("role") == "user")
    return bool(_GRID_CUE_RE.search(combined))


class GridParamError(ValueError):
    """A chat-frontend grid strategy translation was missing/invalid a required grid parameter."""


def _validate_grid_translation(result: StrategyTranslation) -> None:
    required = (
        "grid_anchor", "grid_session_scope", "grid_entry_ticks", "grid_add_ticks",
        "grid_max_lots", "grid_take_profit_ticks",
    )
    missing = [f for f in required if getattr(result, f) is None]
    if missing:
        raise GridParamError(f"grid strategy is missing required field(s): {', '.join(missing)}")
    if result.grid_entry_ticks <= 0 or result.grid_add_ticks <= 0:
        raise GridParamError("grid_entry_ticks and grid_add_ticks must both be positive")
    if result.grid_max_lots < 1:
        raise GridParamError("grid_max_lots must be at least 1")
    if result.grid_take_profit_ticks <= 0:
        raise GridParamError("grid_take_profit_ticks must be positive")

    has_dollars = result.grid_stop_loss_dollars is not None
    has_ticks = result.grid_stop_loss_ticks is not None
    if has_dollars == has_ticks:  # neither set, or both set
        raise GridParamError(
            "grid strategy needs exactly one of grid_stop_loss_dollars / grid_stop_loss_ticks"
        )
    if has_dollars and result.grid_stop_loss_dollars <= 0:
        raise GridParamError("grid_stop_loss_dollars must be a positive $ amount")
    if has_ticks and result.grid_stop_loss_ticks <= 0:
        raise GridParamError("grid_stop_loss_ticks must be a positive tick count")


def _translate_once(conversation: list[dict]) -> StrategyTranslation:
    llm = _get_llm()
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *conversation]
    raw = llm.call(messages, response_model=StrategyTranslation)
    data = json.loads(raw) if isinstance(raw, str) else raw
    return StrategyTranslation.model_validate(data)


def translate_and_validate(conversation: list[dict], max_attempts: int = 2) -> StrategyTranslation:
    """Translate a trader's message (plus prior chat turns) into a
    StrategyTranslation, validating every produced expression against
    dsl_tools before accepting it. If the LLM's first attempt is unsafe or
    malformed, it gets one retry with the specific error appended to the
    conversation; if it still fails, we fall back to asking the trader to
    rephrase rather than silently accepting or guessing.

    Grid/ladder requests get an extra deterministic nudge (see
    _looks_like_grid_request/_GRID_HINT): the LLM doesn't reliably classify
    these as grid mode on every call, so a cue-matched request gets the hint
    injected up front, and -- if it's still misclassified anyway -- one more
    corrective retry before falling back to a clarifying question."""
    messages = list(conversation)
    is_grid_request = _looks_like_grid_request(conversation)
    if is_grid_request:
        messages = [*messages, {"role": "user", "content": _GRID_HINT}]
    grid_retry_used = not is_grid_request  # only spend the extra retry if it was actually a grid cue

    last_error = None
    for _ in range(max_attempts):
        result = _translate_once(messages)
        if not result.understood:
            if not grid_retry_used:
                grid_retry_used = True
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.model_dump_json()},
                    {"role": "user", "content": _GRID_HINT},
                ]
                continue
            return result
        try:
            if result.is_grid:
                _validate_grid_translation(result)
            else:
                dsl_tools.compile_strategy_params(_params_from_translation(result))
            return result
        except (dsl_tools.SafeExpressionError, GridParamError) as e:
            last_error = str(e)
            messages = [
                *messages,
                {"role": "assistant", "content": result.model_dump_json()},
                {
                    "role": "user",
                    "content": (
                        f"That was invalid: {last_error}. Regenerate it correctly per your "
                        "instructions."
                    ),
                },
            ]
    return StrategyTranslation(
        understood=False,
        clarifying_question=(
            "I couldn't turn that into a valid strategy expression "
            f"({last_error}). Could you rephrase the rule in terms of the spread's "
            "z-score, RSI, trend, correlation, or the model's confidence score?"
        ),
    )


_GRID_ANCHOR_HOUR = {
    "morning_open": "MORNING_ANCHOR_UTC_HOUR",
    "mid_morning_open": "MID_MORNING_ANCHOR_UTC_HOUR",
    "afternoon_reopen": "AFTERNOON_ANCHOR_UTC_HOUR",
}
_GRID_ANCHOR_LABEL = {
    "morning_open": "9:15 HKT morning open",
    "mid_morning_open": "9:30 HKT mid-morning price",
    "afternoon_reopen": "1pm HKT reopen",
}
_GRID_SESSION_END_HOUR = {
    "full_day": "DAY_SESSION_END_UTC_HOUR",
    "morning_only": "MORNING_SESSION_END_UTC_HOUR",
    "through_night": "NIGHT_SESSION_END_UTC_HOUR",
}
_GRID_SCOPE_LABEL = {
    "full_day": "full day, carried through lunch to the 16:30 HKT close",
    "morning_only": "morning only, flat by the 12:30 HKT lunch break",
    "through_night": "full day and into the night session, flat by 23:00 HKT",
}


def build_strategy_spec(strategy_id: str, result: StrategyTranslation) -> dict:
    """Assemble the full strategy dict the backtest engine (and per-user
    storage) expect from a confirmed StrategyTranslation. Dispatches to a
    session-anchored grid spec (engine="session_grid") or a DSL-expression
    spec (engine="dsl_expression") depending on result.is_grid."""
    if result.is_grid:
        return _build_grid_spec(strategy_id, result)
    return _build_dsl_spec(strategy_id, result)


def _build_grid_spec(strategy_id: str, result: StrategyTranslation) -> dict:
    anchor_hour = getattr(config, _GRID_ANCHOR_HOUR[result.grid_anchor])
    session_end_hour = getattr(config, _GRID_SESSION_END_HOUR[result.grid_session_scope])
    params = {
        "anchor_utc_hour": anchor_hour,
        "day_session_end_utc_hour": session_end_hour,
        "entry_ticks": result.grid_entry_ticks,
        "add_ticks": result.grid_add_ticks,
        "max_lots": result.grid_max_lots,
        "take_profit_ticks": result.grid_take_profit_ticks,
    }
    if result.grid_stop_loss_dollars is not None:
        params["stop_mode"] = "dollar_loss_from_avg_entry"
        params["stop_loss_dollars"] = result.grid_stop_loss_dollars
        stop_desc = f"HKD {result.grid_stop_loss_dollars:,.0f} unrealized loss vs the average entry price"
    else:
        params["stop_mode"] = "ticks_from_avg_entry"
        params["stop_loss_ticks"] = result.grid_stop_loss_ticks
        stop_desc = f"{result.grid_stop_loss_ticks} ticks adverse from the average entry price"

    anchor_label = _GRID_ANCHOR_LABEL[result.grid_anchor]
    scope_label = _GRID_SCOPE_LABEL[result.grid_session_scope]
    entry_desc = (
        f"Enter 1 lot fading the first {params['entry_ticks']}-tick move away from the "
        f"{anchor_label}; add 1 lot every further {params['add_ticks']} ticks of adverse "
        f"extension, up to {params['max_lots']} lots. Session scope: {scope_label}."
    )
    exit_desc = (
        f"Take-profit: close all lots at {params['take_profit_ticks']} ticks favorable from "
        f"the average entry price. Stop-loss: close all lots if {stop_desc}."
    )
    return {
        "id": strategy_id,
        "name": result.name or strategy_id,
        "engine": "session_grid",
        "source": "chat_frontend",
        "entry_logic": entry_desc,
        "exit_logic": exit_desc,
        "plain_explanation": result.plain_explanation,
        "hedge_ratio": {
            "source": "fixed_hkex_contract_ratio",
            "description": (
                f"Fixed HKEX spread contract ratio: {config.MHI_HEDGE_LOTS} MHI lots vs "
                f"{config.HHI_HEDGE_LOTS} HHI lot(s) per grid rung."
            ),
            "tick_reference": (
                "Tick-distance triggers (entry/add) are measured on the trading platform's own "
                "displayed spread price (config.spread_ui_price), not the $ P&L spread series."
            ),
        },
        "position_sizing": {
            "method": "fixed_lots_per_grid_rung",
            "description": f"1 spread unit per grid rung, up to {params['max_lots']} units total.",
        },
        "costs": {
            "mhi_commission_per_lot_per_side": config.MHI_COMMISSION_PER_LOT,
            "hhi_commission_per_lot_per_side": config.HHI_COMMISSION_PER_LOT,
            "slippage_ticks_per_leg_per_side": config.SLIPPAGE_TICKS,
            "tick_size": config.TICK_SIZE,
        },
        "params": params,
    }


def _build_dsl_spec(strategy_id: str, result: StrategyTranslation) -> dict:
    params = _params_from_translation(result)
    entry_desc = (
        f"Long: {params['entry_long_expr'] or '(none)'} | "
        f"Short: {params['entry_short_expr'] or '(none)'}"
    )
    exit_desc = f"Exit: {params['exit_expr']}"
    if params.get("stop_expr"):
        exit_desc += f" | Stop: {params['stop_expr']}"
    if params.get("max_holding_bars"):
        exit_desc += f" | Time-stop after {params['max_holding_bars']} bars"

    return {
        "id": strategy_id,
        "name": result.name or strategy_id,
        "engine": "dsl_expression",
        "source": "chat_frontend",
        "entry_logic": entry_desc,
        "exit_logic": exit_desc,
        "plain_explanation": result.plain_explanation,
        "hedge_ratio": {
            "source": "fixed_hkex_contract_ratio",
            "description": (
                f"Fixed HKEX spread contract ratio: {config.MHI_HEDGE_LOTS} MHI lots vs "
                f"{config.HHI_HEDGE_LOTS} HHI lot(s), same as every other strategy in this "
                "pipeline."
            ),
        },
        "position_sizing": {
            "method": "inverse_volatility_target",
            "risk_fraction_of_equity": config.RISK_FRACTION,
            "max_leverage": config.MAX_LEVERAGE,
        },
        "costs": {
            "mhi_commission_per_lot_per_side": config.MHI_COMMISSION_PER_LOT,
            "hhi_commission_per_lot_per_side": config.HHI_COMMISSION_PER_LOT,
            "slippage_ticks_per_leg_per_side": config.SLIPPAGE_TICKS,
            "tick_size": config.TICK_SIZE,
        },
        "params": params,
    }
