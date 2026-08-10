"""Per-trader password auth for the chat frontend.

Registered as chainlit's password_auth_callback (wired up in app.py via the
import below). Each of the 6 traders logs in with their own username/
password; chainlit's session is then scoped to cl.User.identifier, which
chat_app/storage.py uses to keep every trader's strategies and results
private to them.
"""
import bcrypt
import chainlit as cl

from chat_app.config import TRADERS


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> "cl.User | None":
    pw_hash = TRADERS.get(username)
    if pw_hash is None:
        return None
    try:
        ok = bcrypt.checkpw(password.encode(), pw_hash.encode())
    except ValueError:
        # malformed hash in CHAT_TRADERS -- fail closed, not open
        return None
    if not ok:
        return None
    return cl.User(identifier=username, metadata={"role": "trader"})
