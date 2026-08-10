"""One-off CLI helper: hash a trader's password for CHAT_TRADERS in .env.

Usage:
    python chat_app/hash_password.py

Prompts for a password (hidden input) and prints the bcrypt hash to paste
into .env as `username:<hash>` inside CHAT_TRADERS -- the plaintext
password itself is never written anywhere.
"""
import getpass

import bcrypt


def main() -> None:
    pw = getpass.getpass("Password to hash: ")
    confirm = getpass.getpass("Confirm password: ")
    if pw != confirm:
        raise SystemExit("Passwords did not match.")
    if not pw:
        raise SystemExit("Password cannot be empty.")
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    print("\nAdd this to CHAT_TRADERS in .env as  username:<hash>  :\n")
    print(hashed)


if __name__ == "__main__":
    main()
