"""Safe, sandboxed strategy-expression DSL for the chat frontend.

Traders describe strategies in plain language in the chat box; the chat
LLM's only job is to translate that into boolean/arithmetic expressions
over this DSL's whitelisted variables (chat_app/translate.py). Those
expression strings are compiled here into a small AST and evaluated by a
hand-written tree-walking interpreter -- eval()/exec()/compile()-as-code is
never invoked on LLM-authored text, so an expression can only read the
variables below and combine them with the operators/functions below. The
LLM never computes a P&L number itself: once an expression is confirmed,
the exact same deterministic backtest engine (backtest_tools.py) that the
main pipeline uses runs it.
"""
import ast
from dataclasses import dataclass

ALLOWED_VARIABLES: dict[str, str] = {
    "spread": (
        "Fixed-ratio MHI/HHI $ spread at this bar's close (long MHI_HEDGE_LOTS "
        "lots of MHI vs HHI_HEDGE_LOTS lots of HHI, each at its real contract "
        "multiplier). This is an ABSOLUTE dollar level, NOT centered on zero and "
        "NOT a small tick-distance number -- at current MHI/HHI price levels it "
        "typically sits in the tens of thousands of HKD (e.g. ~65,000-90,000), "
        "dominated by the outright price level of both legs, not by how far price "
        "has moved recently. Comparing it to a small constant (e.g. `spread > 100`) "
        "is almost always a mistake -- use spread_zscore, spread_roc, or "
        "spread_ma_crossover instead for anything meant to capture 'how far has the "
        "spread moved / deviated' rather than its raw level. There is no "
        "session/anchor-relative variable (e.g. 'distance from today's opening "
        "price') in this DSL."
    ),
    "spread_zscore": (
        "Z-score of the spread vs its rolling long-window mean/std -- positive "
        "means the spread is unusually high, negative unusually low."
    ),
    "spread_rsi": "RSI (0-100) of the spread over the rolling short window -- >70 overbought, <30 oversold.",
    "spread_ma_crossover": (
        "Short-window spread moving average minus long-window spread moving "
        "average -- positive means the short-term trend is up."
    ),
    "rolling_corr_returns": (
        "Rolling correlation of MHI vs HHI returns over the short window -- how "
        "intact the pair relationship currently looks (near 1 = tightly linked, "
        "near 0 or negative = breaking down)."
    ),
    "spread_vol": "Rolling standard deviation of the bar-to-bar $ change in the spread (short window).",
    "ml_score": (
        "The trained ML model's predicted probability (0-1) of its target -- "
        "higher means the model leans toward the target's positive outcome."
    ),
    "hedge_ratio": "The fixed HKEX MHI:HHI contract lot ratio (a constant, not a rolling estimate).",
    "MHI_Close": "MHI futures close price of this bar.",
    "HHI_Close": "HHI futures close price of this bar.",
    "MHI_Open": (
        "MHI futures open price of this bar -- informational only. Every entry/"
        "exit this DSL triggers fills at the *next* bar's open, never this one, "
        "so there is no lookahead regardless of what an expression reads."
    ),
    "HHI_Open": "HHI futures open price of this bar -- informational only, see MHI_Open.",
    "hour_hkt": (
        "Hong Kong local wall-clock time (HKT, UTC+8, no DST) of this bar's close, "
        "as a float in [0, 24) -- e.g. 23.0 is 11:00pm HKT, 23.5 is 11:30pm HKT. "
        "The source data's timestamps are UTC; this variable is already converted "
        "to HKT so exit/stop rules can reference a trader's normal wall-clock "
        "session cutoffs directly, e.g. `hour_hkt >= 23` to flatten by 11pm HKT "
        "(HKEX's night/T+1 session runs into the small hours HKT, so this is the "
        "usual way to force an exit before it gets that late rather than holding "
        "the position overnight)."
    ),
}

ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max}

_ALLOWED_COMPARE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY_OPS = (ast.Not, ast.USub, ast.UAdd)


class SafeExpressionError(ValueError):
    """A DSL expression string failed to parse, or used anything outside the whitelist."""


def compile_expr(expr: str) -> ast.Expression:
    """Parse + whitelist-validate a DSL expression string. Returns the
    validated AST, ready for repeated evaluation via eval_expr(). Raises
    SafeExpressionError (never a bare SyntaxError) on anything disallowed."""
    if not expr or not expr.strip():
        raise SafeExpressionError("expression is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeExpressionError(f"could not parse expression {expr!r}: {e}") from e
    _validate(tree.body, expr)
    return tree


def _validate(node: ast.AST, expr: str) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL_OPS):
            raise SafeExpressionError(f"disallowed boolean operator in {expr!r}")
        for v in node.values:
            _validate(v, expr)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BIN_OPS):
            raise SafeExpressionError(f"disallowed arithmetic operator in {expr!r}")
        _validate(node.left, expr)
        _validate(node.right, expr)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise SafeExpressionError(f"disallowed unary operator in {expr!r}")
        _validate(node.operand, expr)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, _ALLOWED_COMPARE_OPS):
                raise SafeExpressionError(f"disallowed comparison operator in {expr!r}")
        _validate(node.left, expr)
        for c in node.comparators:
            _validate(c, expr)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
            raise SafeExpressionError(
                f"disallowed function call in {expr!r} -- only {sorted(ALLOWED_FUNCS)} are allowed"
            )
        if node.keywords:
            raise SafeExpressionError(f"keyword arguments not allowed in {expr!r}")
        for a in node.args:
            _validate(a, expr)
    elif isinstance(node, ast.Name):
        if node.id not in ALLOWED_VARIABLES:
            raise SafeExpressionError(
                f"unknown variable {node.id!r} in {expr!r} -- allowed variables: "
                f"{sorted(ALLOWED_VARIABLES)}"
            )
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise SafeExpressionError(f"disallowed constant {node.value!r} in {expr!r}")
    else:
        raise SafeExpressionError(f"disallowed syntax ({type(node).__name__}) in {expr!r}")


def eval_expr(tree: ast.Expression, env: dict) -> object:
    """Evaluate an already-validated expression tree (from compile_expr)
    against a variable environment. Never calls eval()/exec() -- this is a
    hand-written recursive interpreter over the whitelisted node types only."""
    return _eval_node(tree.body, env)


def _eval_node(node: ast.AST, env: dict):
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                result = _eval_node(v, env)
                if not result:
                    return result
            return result
        for v in node.values:
            result = _eval_node(v, env)
            if result:
                return result
        return result
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right  # ast.Div
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        return +val
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, env)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            else:
                ok = left != right
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        func = ALLOWED_FUNCS[node.func.id]
        return func(*(_eval_node(a, env) for a in node.args))
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise SafeExpressionError(f"cannot evaluate node type {type(node).__name__}")


@dataclass
class CompiledStrategy:
    entry_long: "ast.Expression | None"
    entry_short: "ast.Expression | None"
    exit: ast.Expression
    stop: "ast.Expression | None"
    max_holding_bars: "int | None"


def compile_strategy_params(params: dict) -> CompiledStrategy:
    """Validate + compile every DSL expression in a strategy's `params`
    dict (as produced by the chat frontend's LLM translation step). Raises
    SafeExpressionError if anything is invalid. Requires at least one of
    entry_long_expr / entry_short_expr plus a mandatory exit_expr -- a
    strategy that can enter but never exit isn't backtestable."""
    entry_long_s = params.get("entry_long_expr")
    entry_short_s = params.get("entry_short_expr")
    exit_s = params.get("exit_expr")
    stop_s = params.get("stop_expr")

    if not entry_long_s and not entry_short_s:
        raise SafeExpressionError("strategy needs at least one of entry_long_expr / entry_short_expr")
    if not exit_s:
        raise SafeExpressionError("strategy needs exit_expr")

    return CompiledStrategy(
        entry_long=compile_expr(entry_long_s) if entry_long_s else None,
        entry_short=compile_expr(entry_short_s) if entry_short_s else None,
        exit=compile_expr(exit_s),
        stop=compile_expr(stop_s) if stop_s else None,
        max_holding_bars=params.get("max_holding_bars"),
    )


def _hour_hkt(ts) -> float:
    """Convert a (UTC, tz-naive) bar timestamp to Hong Kong wall-clock hour-
    of-day as a float in [0, 24) -- same `.hour + .minute/60` convention
    already used for UTC session anchors in config.py, offset +8h for HKT
    (UTC+8 year-round, no DST) and wrapped past midnight with `% 24`."""
    return (ts.hour + ts.minute / 60.0 + 8) % 24


def row_env(row) -> dict:
    """Build the DSL variable environment for one backtest bar (a pandas
    Series, or any mapping with the same keys) -- includes the derived
    `spread` and `hour_hkt` variables alongside the raw feature/price
    columns. `row.name` is the bar's timestamp index label (set by
    build_backtest_dataframe / bt_df.iloc[i] or .iterrows())."""
    from pair_crew import config

    env = {
        name: row[name]
        for name in ALLOWED_VARIABLES
        if name not in ("spread", "hour_hkt")
    }
    env["spread"] = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
    env["hour_hkt"] = _hour_hkt(row.name)
    return env


def _trigger_rate(expr: "ast.Expression | None", bt_df, sample_n: int, random_state: int) -> "float | None":
    if expr is None:
        return None
    sample = bt_df.sample(sample_n, random_state=random_state) if len(bt_df) > sample_n else bt_df
    hits = sum(bool(eval_expr(expr, row_env(row))) for _, row in sample.iterrows())
    return hits / len(sample)


def plausibility_warnings(compiled: CompiledStrategy, bt_df, sample_n: int = 3000) -> list[str]:
    """Quick pre-backtest sanity check, run against a sample of real
    historical bars: flag any entry/exit/stop expression that is (almost)
    always or never true. A condition that fires on ~100% (or ~0%) of real
    bars almost always means a threshold or variable's scale was
    misunderstood -- e.g. comparing `spread` (the raw $ mark-to-market
    value, tens of thousands of HKD at current MHI/HHI price levels)
    against a small tick-sized number intended for an anchor-relative
    distance the DSL doesn't actually expose -- rather than a genuine
    trading rule, and left unflagged it silently churns a position in and
    out every bar, bleeding the whole account to transaction costs."""
    warnings = []
    fields = {
        "entry_long_expr": compiled.entry_long,
        "entry_short_expr": compiled.entry_short,
        "exit_expr": compiled.exit,
        "stop_expr": compiled.stop,
    }
    for label, expr in fields.items():
        rate = _trigger_rate(expr, bt_df, sample_n, random_state=0)
        if rate is None:
            continue
        if rate >= 0.98:
            warnings.append(
                f"`{label}` is true on about {rate * 100:.0f}% of historical bars -- it "
                "would fire almost continuously, which usually means a threshold or "
                "variable's scale doesn't match what was intended."
            )
        elif rate <= 0.02 and label in ("entry_long_expr", "entry_short_expr"):
            warnings.append(
                f"`{label}` is true on only about {rate * 100:.1f}% of historical bars -- "
                "this entry may rarely or never trigger."
            )
    return warnings


def describe_dsl_for_prompt() -> str:
    """Human-readable variable/function reference for the chat LLM's system
    prompt, so it knows exactly what it may write."""
    lines = ["Allowed variables:"]
    for name, desc in ALLOWED_VARIABLES.items():
        lines.append(f"  - {name}: {desc}")
    lines.append("Allowed functions: " + ", ".join(sorted(ALLOWED_FUNCS)))
    lines.append(
        "Allowed operators: + - * / (arithmetic), < <= > >= == != (comparison), "
        "and / or / not (boolean, lowercase Python keywords). No other syntax is "
        "permitted: no assignments, attribute access, indexing, string literals, "
        "imports, or any function beyond the list above."
    )
    return "\n".join(lines)
