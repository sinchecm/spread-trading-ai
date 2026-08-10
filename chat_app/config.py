"""Shared config for the chat frontend: trader roster, per-user storage paths.

Kept separate from pair_crew/config.py -- this file is chat_app-specific
(auth roster, per-trader storage isolation) and is never touched by the
deterministic pipeline (app.py/main.py), matching the "second, separate
frontend, alongside the existing pipeline" design.
"""
import os
from pathlib import Path

CHAT_APP_ROOT = Path(__file__).resolve().parent
USER_DATA_DIR = CHAT_APP_ROOT / "user_data"
USER_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Reuse the same deterministic-pipeline outputs (features/aligned data/
# predictions) the main app.py/main.py entry points already produce --
# the chat frontend only ever reads these, never writes them.
from pair_crew import config as pipeline_config  # noqa: E402

FEATURES_PATH = pipeline_config.FEATURES_PATH
ALIGNED_DATA_PATH = pipeline_config.ALIGNED_DATA_PATH
PREDICTIONS_PATH = pipeline_config.PREDICTIONS_PATH
BEST_MODEL_META_PATH = pipeline_config.BEST_MODEL_META_PATH


def _load_traders() -> dict[str, str]:
    """Parse CHAT_TRADERS="username:bcrypt_hash,username2:bcrypt_hash2,..."
    from the environment. Each trader's bcrypt hash is generated once with
    `python chat_app/hash_password.py` and pasted into .env -- plaintext
    passwords are never stored anywhere."""
    raw = os.getenv("CHAT_TRADERS", "")
    traders: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        username, _, pw_hash = entry.partition(":")
        username, pw_hash = username.strip(), pw_hash.strip()
        if username and pw_hash:
            traders[username] = pw_hash
    return traders


TRADERS = _load_traders()


def user_dir(username: str) -> Path:
    """Isolated storage directory for one trader's strategies/results.
    Only ever called with a username that has already been checked against
    TRADERS by chat_app/auth.py, so this can't be used to escape into
    another trader's directory."""
    if username not in TRADERS:
        raise ValueError(f"unknown trader {username!r}")
    d = USER_DATA_DIR / username
    d.mkdir(exist_ok=True, parents=True)
    return d
