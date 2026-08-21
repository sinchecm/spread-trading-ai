#!/usr/bin/env python3
"""Headless-browser REPL driver for the chat_app Chainlit frontend.

Reads one command per line from stdin (or a heredoc), drives a real
Chromium tab, and prints results to stdout. Screenshots land in
./screenshots/ next to wherever you run it from (pass an absolute path
to `screenshot` to control that).

Why this exists instead of chromium-cli: chromium-cli isn't installed
in this environment. Playwright is (via the project's conda env), but
its headless_shell binary is missing libnspr4/libnss3/libnssutil3/
libasound.so.2 on this container and there's no passwordless sudo to
`apt-get install` them. The full (non-headless-shell) `chromium`
browser download under ~/.cache/ms-playwright/ happens to already have
all its shared-lib deps satisfied EXCEPT those same NSS/ASound libs --
and this machine's conda env ships matching .so files in its lib/, so
LD_LIBRARY_PATH is enough to bridge the gap without touching apt.
See SKILL.md Gotchas for the full story.

Commands (one per line):
    nav <url>
    fill <selector> <text...>      use for <input> login fields only -- see `type` below
                                    for the chat box, `fill` leaves its submit button
                                    disabled (React state bug, see SKILL.md Gotchas)
    type <selector> <text...>      click + settle + type-with-delay; use this for
                                    #chat-input (chainlit's textarea needs real
                                    keyboard events, and the very first ~1-2
                                    chars are dropped without a settle delay)
    click <selector...>            selector may be a Playwright text= / css selector,
                                    or plain text (matched via :has-text())
    press <key>                    e.g. Enter
    wait-for <selector-or-text=...> [timeout_ms]
    wait-text <substring> [timeout_ms]   poll page body innerText for a substring
                                    (careful: matches ANY occurrence, including in
                                    chainlit's own static prose -- see Gotchas)
    screenshot [path]              default: screenshots/<n>.png, full page
    text                           print page body innerText
    sleep <ms>
    quit

Usage:
    python3 .claude/skills/run-spread-trading-ai/driver.py <<'EOF'
    nav http://127.0.0.1:8010
    fill input[name="email"] trader1
    fill input[name="password"] hunter2
    click Sign In
    wait-for textarea
    screenshot
    EOF
"""
import glob
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def find_chromium_binary() -> str:
    """Prefer a full `chromium` build over `chromium_headless_shell` --
    on this container the headless_shell binary is missing NSS/ASound
    libs that the full build also needs, but LD_LIBRARY_PATH (set by
    the caller, see SKILL.md) covers both the same way. Either works
    once the env is right; the full build is what was verified."""
    cache = Path.home() / ".cache" / "ms-playwright"
    candidates = sorted(glob.glob(str(cache / "chromium-*" / "chrome-linux64" / "chrome")))
    if not candidates:
        raise SystemExit(
            f"No chromium binary found under {cache}. Run: playwright install chromium"
        )
    return candidates[-1]


def resolve_selector(page, raw: str):
    """Try raw as a CSS/Playwright selector first; fall back to
    :has-text() for plain button/link labels like `Sign In`."""
    loc = page.locator(raw)
    try:
        if loc.count() > 0:
            return loc.first
    except Exception:
        pass
    return page.locator(f":text('{raw}')").first


def main() -> None:
    executable_path = find_chromium_binary()
    shot_n = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox"],
            executable_path=executable_path,
        )
        page = browser.new_page()

        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            try:
                if cmd == "nav":
                    page.goto(rest, wait_until="networkidle", timeout=20000)
                    print(f"OK nav {rest} -> title={page.title()!r}")

                elif cmd == "fill":
                    sel, _, text = rest.partition(" ")
                    resolve_selector(page, sel).fill(text)
                    print(f"OK fill {sel!r}")

                elif cmd == "type":
                    sel, _, text = rest.partition(" ")
                    loc = resolve_selector(page, sel)
                    loc.click()
                    page.wait_for_timeout(500)  # settle -- see Gotchas
                    loc.type(text, delay=10)
                    print(f"OK type {sel!r}")

                elif cmd == "click":
                    resolve_selector(page, rest).click()
                    print(f"OK click {rest!r}")

                elif cmd == "press":
                    page.keyboard.press(rest)
                    print(f"OK press {rest}")

                elif cmd == "wait-for":
                    sel_parts = rest.rsplit(" ", 1)
                    if len(sel_parts) == 2 and sel_parts[1].isdigit():
                        sel, timeout = sel_parts[0], int(sel_parts[1])
                    else:
                        sel, timeout = rest, 20000
                    page.wait_for_selector(sel, timeout=timeout)
                    print(f"OK wait-for {sel!r}")

                elif cmd == "wait-text":
                    sel_parts = rest.rsplit(" ", 1)
                    if len(sel_parts) == 2 and sel_parts[1].isdigit():
                        needle, timeout_ms = sel_parts[0], int(sel_parts[1])
                    else:
                        needle, timeout_ms = rest, 60000
                    needle = needle.strip().strip('"')
                    page.wait_for_function(
                        "needle => document.body.innerText.toLowerCase().includes(needle.toLowerCase())",
                        arg=needle,
                        timeout=timeout_ms,
                    )
                    print(f"OK wait-text {needle!r}")

                elif cmd == "screenshot":
                    shot_n += 1
                    path = rest.strip() or str(SCREENSHOT_DIR / f"{shot_n}.png")
                    page.screenshot(path=path, full_page=True)
                    print(f"OK screenshot -> {path}")

                elif cmd == "text":
                    print("---TEXT---")
                    print(page.inner_text("body"))
                    print("---END---")

                elif cmd == "sleep":
                    page.wait_for_timeout(int(rest))
                    print(f"OK sleep {rest}ms")

                elif cmd in ("quit", "exit"):
                    break

                else:
                    print(f"ERR unknown command: {cmd}")

            except Exception as e:
                print(f"ERR {cmd}: {e}")

        browser.close()


if __name__ == "__main__":
    main()
