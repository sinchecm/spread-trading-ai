"""Per-trader isolated storage: confirmed strategies + their backtest
results + equity curves, all scoped under chat_app/user_data/<username>/ so
one trader's book is never visible to another (see chat_app/config.py's
user_dir, which only accepts a username already authenticated by auth.py)."""
import json
import re
import time
from pathlib import Path

import pandas as pd

from chat_app.config import user_dir


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or f"strategy_{int(time.time())}"


def _strategies_dir(username: str) -> Path:
    d = user_dir(username) / "strategies"
    d.mkdir(exist_ok=True, parents=True)
    return d


def _results_dir(username: str) -> Path:
    d = user_dir(username) / "results"
    d.mkdir(exist_ok=True, parents=True)
    return d


def unique_slug(username: str, name: str) -> str:
    base = slugify(name)
    slug = base
    n = 2
    while (_strategies_dir(username) / f"{slug}.json").exists():
        slug = f"{base}_{n}"
        n += 1
    return slug


def save_strategy(username: str, slug: str, strategy: dict, description: str) -> Path:
    record = {
        "slug": slug,
        "description": description,
        "strategy": strategy,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _strategies_dir(username) / f"{slug}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def save_backtest_result(username: str, slug: str, metrics: dict, equity_curve: pd.Series) -> tuple[Path, Path]:
    metrics_path = _results_dir(username) / f"{slug}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    equity_path = _results_dir(username) / f"{slug}_equity.csv"
    equity_curve.to_frame("equity").to_csv(equity_path)
    return metrics_path, equity_path


def list_strategies(username: str) -> list[dict]:
    records = []
    for path in sorted(_strategies_dir(username).glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def load_equity_curve(username: str, slug: str) -> "pd.Series | None":
    path = _results_dir(username) / f"{slug}_equity.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["equity"]
