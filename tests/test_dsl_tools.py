"""Tests for pair_crew/tools/dsl_tools.py -- the sandboxed expression DSL
that the chat frontend's LLM is allowed to write into.

Priorities, in order:
  1. The sandbox cannot be escaped (no attribute/subscript access, no
     disallowed builtins, no way to reach eval/exec/import/__class__ etc.)
     -- this is the security boundary between untrusted LLM-authored text
     and the Python process, so it gets the most exhaustive coverage.
  2. Every allowed operator/function evaluates with correct semantics.
  3. compile_strategy_params enforces the "must have exit, at least one
     entry" contract the backtest engine relies on.
  4. row_env/_hour_hkt build the evaluation environment correctly,
     including the HKT-conversion + midnight-wrap edge case.
  5. plausibility_warnings' near-always/near-never heuristics.
"""
import datetime
import math

import pandas as pd
import pytest

from pair_crew.tools.dsl_tools import (
    ALLOWED_FUNCS,
    ALLOWED_VARIABLES,
    CompiledStrategy,
    SafeExpressionError,
    _hour_hkt,
    compile_expr,
    compile_strategy_params,
    describe_dsl_for_prompt,
    eval_expr,
    plausibility_warnings,
    row_env,
)


def ev(expr: str, env: dict | None = None):
    """Compile + evaluate in one step against a minimal default env."""
    base_env = {name: 1.0 for name in ALLOWED_VARIABLES}
    if env:
        base_env.update(env)
    return eval_expr(compile_expr(expr), base_env)


# --------------------------------------------------------------------------
# 1. Sandbox escape attempts -- every one of these MUST raise
#    SafeExpressionError (not silently succeed, not raise some other
#    exception that a caller might not be catching).
# --------------------------------------------------------------------------

ESCAPE_ATTEMPTS = [
    "__import__('os').system('id')",
    "().__class__",
    "().__class__.__bases__[0]",
    "(1).__class__",
    "spread.__class__",
    "spread.__init__",
    "[].__class__",
    "{}.__class__",
    "getattr(spread, '__class__')",
    "spread.real",  # attribute access on a float, still disallowed
    "eval('1')",
    "exec('1')",
    "compile('1', '', 'eval')",
    "open('/etc/passwd')",
    "globals()",
    "locals()",
    "vars()",
    "__builtins__",
    "spread[0]",
    "[spread][0]",
    "spread if True else 0",  # ternary / IfExp
    "lambda x: x",
    "(lambda: spread)()",
    "[x for x in range(3)]",  # list comprehension
    "{x for x in range(3)}",  # set comprehension
    "{x: x for x in range(3)}",  # dict comprehension
    "(x for x in range(3))",  # generator expression
    "[1, 2, 3]",
    "(1, 2, 3)",
    "{1, 2, 3}",
    "{'a': 1}",
    "*spread,",
    "spread; spread",
    "import os",
    "1 or import os",
    "spread := 5",  # walrus
    "f'{spread}'",  # f-string / JoinedStr
    "'a string'",
    "b'bytes'",
    "1j",  # complex constant
    "None",
    "spread ** 2",  # power not whitelisted
    "spread // 2",  # floor div not whitelisted
    "spread % 2",  # modulo not whitelisted
    "spread & 1",  # bitwise
    "spread | 1",
    "spread ^ 1",
    "spread << 1",
    "spread >> 1",
    "spread is spread",
    "spread is not spread",
    "spread in [1, 2]",
    "spread not in [1, 2]",
    "not_a_real_variable",
    "spread(1)",  # calling a variable as if it were a function
    "abs(spread, extra=1)",  # keyword arg
    "max(spread, key=abs)",  # keyword arg on an allowed func
    "print(spread)",
    "type(spread)",
    "isinstance(spread, float)",
    "",
    "   ",
    "def f(): pass",
    "class C: pass",
    "assert spread",
    "yield spread",
    "return spread",
]


@pytest.mark.parametrize("expr", ESCAPE_ATTEMPTS)
def test_disallowed_expressions_are_rejected(expr):
    with pytest.raises(SafeExpressionError):
        compile_expr(expr)


def test_rejected_expressions_never_raise_bare_syntax_error():
    # compile_expr's contract is "SafeExpressionError, never a bare
    # SyntaxError" -- callers should only need to catch one exception type.
    with pytest.raises(SafeExpressionError):
        compile_expr("def f(:")


def test_disallowed_expression_does_not_leak_python_exception_types():
    for expr in ["__import__('os')", "spread.__class__", "spread ** 2"]:
        try:
            compile_expr(expr)
        except SafeExpressionError:
            pass
        except Exception as e:  # pragma: no cover - failure path
            pytest.fail(f"{expr!r} raised {type(e).__name__} instead of SafeExpressionError: {e}")
        else:
            pytest.fail(f"{expr!r} should have raised SafeExpressionError")


# --------------------------------------------------------------------------
# 2. Allowed syntax: compiles and evaluates with correct semantics.
# --------------------------------------------------------------------------

def test_allowed_variables_all_compile_alone():
    for name in ALLOWED_VARIABLES:
        compile_expr(name)  # should not raise


@pytest.mark.parametrize(
    "expr,env,expected",
    [
        ("spread_zscore > 2", {"spread_zscore": 3}, True),
        ("spread_zscore > 2", {"spread_zscore": 1}, False),
        ("spread_zscore >= 2", {"spread_zscore": 2}, True),
        ("spread_zscore < 2", {"spread_zscore": 1}, True),
        ("spread_zscore <= 2", {"spread_zscore": 2}, True),
        ("spread_zscore == 2", {"spread_zscore": 2}, True),
        ("spread_zscore != 2", {"spread_zscore": 3}, True),
        ("1 < spread_zscore < 3", {"spread_zscore": 2}, True),  # chained compare
        ("1 < spread_zscore < 3", {"spread_zscore": 5}, False),
    ],
)
def test_comparison_semantics(expr, env, expected):
    assert ev(expr, env) is expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 3", 5),
        ("2 - 3", -1),
        ("2 * 3", 6),
        ("6 / 4", 1.5),
        ("-spread_zscore", None),  # handled separately below (needs env)
        ("+spread_zscore", None),
    ],
)
def test_arithmetic_semantics(expr, expected):
    if expected is None:
        return
    assert ev(expr) == pytest.approx(expected)


def test_unary_operators():
    assert ev("-spread_zscore", {"spread_zscore": 4}) == -4
    assert ev("+spread_zscore", {"spread_zscore": 4}) == 4
    assert ev("not (spread_zscore > 0)", {"spread_zscore": -1}) is True
    assert ev("not (spread_zscore > 0)", {"spread_zscore": 1}) is False


def test_division_by_zero_propagates_as_python_would():
    # dsl_tools does not special-case this -- pin down the actual behavior
    # (a raised ZeroDivisionError, not a silently swallowed NaN/None) so a
    # future change to this is a deliberate decision, not an accident.
    with pytest.raises(ZeroDivisionError):
        ev("spread_zscore / 0", {"spread_zscore": 1})


def test_boolean_and_or_short_circuit_and_return_python_truthy_value():
    # `and`/`or` return the actual operand value (Python semantics), not a
    # coerced bool -- confirm eval_expr preserves that rather than forcing
    # everything through bool().
    assert ev("0 and 5") == 0
    assert ev("3 and 5") == 5
    assert ev("0 or 5") == 5
    assert ev("3 or 5") == 3


def test_allowed_functions():
    assert ev("abs(spread_zscore)", {"spread_zscore": -3}) == 3
    assert ev("min(spread_zscore, spread_rsi)", {"spread_zscore": 1, "spread_rsi": 2}) == 1
    assert ev("max(spread_zscore, spread_rsi)", {"spread_zscore": 1, "spread_rsi": 2}) == 2
    assert set(ALLOWED_FUNCS) == {"abs", "min", "max"}


def test_constants_int_float_bool():
    assert ev("1") == 1
    assert ev("1.5") == 1.5
    assert ev("True") is True
    assert ev("False") is False


def test_realistic_composite_expression():
    expr = "spread_zscore > 2 and rolling_corr_returns > 0.2 and not (spread_rsi > 90)"
    env = {"spread_zscore": 2.5, "rolling_corr_returns": 0.3, "spread_rsi": 50}
    assert ev(expr, env) is True
    env["spread_rsi"] = 95
    assert ev(expr, env) is False


def test_unknown_variable_message_lists_allowed_variables():
    with pytest.raises(SafeExpressionError) as exc_info:
        compile_expr("totally_made_up_var > 1")
    msg = str(exc_info.value)
    assert "totally_made_up_var" in msg
    assert "spread_zscore" in msg  # a real allowed variable should be listed as a hint


# --------------------------------------------------------------------------
# 3. compile_strategy_params contract.
# --------------------------------------------------------------------------

def test_compile_strategy_params_requires_an_entry():
    with pytest.raises(SafeExpressionError, match="entry_long_expr / entry_short_expr"):
        compile_strategy_params({"exit_expr": "spread_zscore < 1"})


def test_compile_strategy_params_requires_exit():
    with pytest.raises(SafeExpressionError, match="exit_expr"):
        compile_strategy_params({"entry_long_expr": "spread_zscore > 2"})


def test_compile_strategy_params_happy_path_long_only():
    compiled = compile_strategy_params(
        {
            "entry_long_expr": "spread_zscore < -2",
            "exit_expr": "spread_zscore > 0",
            "stop_expr": "spread_zscore < -4",
            "max_holding_bars": 100,
        }
    )
    assert isinstance(compiled, CompiledStrategy)
    assert compiled.entry_long is not None
    assert compiled.entry_short is None
    assert compiled.exit is not None
    assert compiled.stop is not None
    assert compiled.max_holding_bars == 100


def test_compile_strategy_params_stop_and_max_holding_are_optional():
    compiled = compile_strategy_params(
        {"entry_short_expr": "spread_zscore > 2", "exit_expr": "spread_zscore < 0"}
    )
    assert compiled.entry_long is None
    assert compiled.entry_short is not None
    assert compiled.stop is None
    assert compiled.max_holding_bars is None


def test_compile_strategy_params_propagates_invalid_expression():
    with pytest.raises(SafeExpressionError):
        compile_strategy_params(
            {"entry_long_expr": "spread.__class__", "exit_expr": "spread_zscore < 0"}
        )


# --------------------------------------------------------------------------
# 4. row_env / _hour_hkt.
# --------------------------------------------------------------------------

def test_hour_hkt_basic_conversion():
    ts = pd.Timestamp("2026-01-01 01:00:00")  # 01:00 UTC -> 09:00 HKT
    assert _hour_hkt(ts) == pytest.approx(9.0)


def test_hour_hkt_wraps_past_midnight():
    ts = pd.Timestamp("2026-01-01 16:30:00")  # 16:30 UTC + 8h = 24:30 -> wraps to 00:30
    assert _hour_hkt(ts) == pytest.approx(0.5)


def test_hour_hkt_handles_minutes():
    ts = pd.Timestamp("2026-01-01 23:45:00")  # 23:45 UTC + 8h = 31:45 -> wraps to 07:45
    assert _hour_hkt(ts) == pytest.approx(7.75)


def test_row_env_includes_derived_spread_and_hour_hkt():
    row = pd.Series(
        {
            "spread_zscore": 1.0,
            "spread_rsi": 50.0,
            "spread_ma_crossover": 0.1,
            "rolling_corr_returns": 0.5,
            "spread_vol": 2.0,
            "ml_score": 0.6,
            "hedge_ratio": 2.0,
            "MHI_Close": 20000.0,
            "HHI_Close": 8000.0,
            "MHI_Open": 19990.0,
            "HHI_Open": 7990.0,
        },
        name=pd.Timestamp("2026-01-01 01:00:00"),
    )
    env = row_env(row)
    assert env["hour_hkt"] == pytest.approx(9.0)
    from pair_crew import config

    expected_spread = config.spread_dollar_value(row["MHI_Close"], row["HHI_Close"])
    assert env["spread"] == pytest.approx(expected_spread)
    # every other allowed variable should be carried straight through
    for name in ALLOWED_VARIABLES:
        if name in ("spread", "hour_hkt"):
            continue
        assert env[name] == row[name]


def test_row_env_result_is_directly_usable_by_eval_expr():
    row = pd.Series(
        {name: 1.0 for name in ALLOWED_VARIABLES if name not in ("spread", "hour_hkt")}
        | {"MHI_Close": 20000.0, "HHI_Close": 8000.0},
        name=pd.Timestamp("2026-01-01 01:00:00"),
    )
    env = row_env(row)
    tree = compile_expr("spread_zscore > 0 and hour_hkt < 24")
    assert eval_expr(tree, env) is True


# --------------------------------------------------------------------------
# 5. plausibility_warnings.
# --------------------------------------------------------------------------

def _make_bt_df(n=500):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "spread_zscore": [((-1) ** i) * (i % 5) for i in range(n)],
            "spread_rsi": [50.0] * n,
            "spread_ma_crossover": [0.0] * n,
            "rolling_corr_returns": [0.5] * n,
            "spread_vol": [1.0] * n,
            "ml_score": [0.5] * n,
            "hedge_ratio": [2.0] * n,
            "MHI_Close": [20000.0] * n,
            "HHI_Close": [8000.0] * n,
            "MHI_Open": [20000.0] * n,
            "HHI_Open": [8000.0] * n,
        },
        index=idx,
    )


def test_plausibility_warnings_flags_always_true_expression():
    bt_df = _make_bt_df()
    compiled = compile_strategy_params(
        {"entry_long_expr": "spread_rsi > -1000", "exit_expr": "spread_rsi < 100000"}
    )
    warnings = plausibility_warnings(compiled, bt_df)
    assert any("entry_long_expr" in w and "almost continuously" in w for w in warnings)
    assert any("exit_expr" in w for w in warnings)


def test_plausibility_warnings_flags_rarely_true_entry():
    bt_df = _make_bt_df()
    compiled = compile_strategy_params(
        {"entry_long_expr": "spread_zscore > 1000", "exit_expr": "spread_rsi < 100000"}
    )
    warnings = plausibility_warnings(compiled, bt_df)
    assert any("entry_long_expr" in w and "rarely or never" in w for w in warnings)


def test_plausibility_warnings_silent_for_balanced_expression_and_none_stop():
    bt_df = _make_bt_df()
    compiled = compile_strategy_params(
        {
            "entry_long_expr": "spread_zscore > 2",
            "exit_expr": "spread_zscore < -2",
        }
    )
    # entry_short_expr and stop_expr are None -> must not error or warn on them
    warnings = plausibility_warnings(compiled, bt_df)
    assert not any("entry_short_expr" in w or "stop_expr" in w for w in warnings)


def test_plausibility_warnings_handles_sample_n_larger_than_df():
    bt_df = _make_bt_df(n=10)
    compiled = compile_strategy_params(
        {"entry_long_expr": "spread_zscore > 0", "exit_expr": "spread_zscore < 0"}
    )
    # sample_n (3000 default) > len(bt_df) -- must not crash, just use whole df
    warnings = plausibility_warnings(compiled, bt_df, sample_n=3000)
    assert isinstance(warnings, list)


# --------------------------------------------------------------------------
# 6. describe_dsl_for_prompt -- cheap smoke test that the LLM-facing
#    reference text doesn't silently drop a variable/function.
# --------------------------------------------------------------------------

def test_describe_dsl_for_prompt_mentions_every_variable_and_function():
    text = describe_dsl_for_prompt()
    for name in ALLOWED_VARIABLES:
        assert name in text
    for name in ALLOWED_FUNCS:
        assert name in text
