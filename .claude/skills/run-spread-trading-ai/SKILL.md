---
name: run-spread-trading-ai
description: Build, launch, and drive the MHI/HHI pair-trading platform's web frontends -- the Chainlit trader chat app (chat_app/) and the Streamlit pipeline control panel (app.py). Use when asked to run, start, screenshot, or smoke-test either app, or to confirm a change works end-to-end (login, DSL translation, backtest) rather than just passing tests.
---

Two separate Chainlit/Streamlit web apps, both driven the same way: launch
in the background, then drive a real headless Chromium tab with
`.claude/skills/run-spread-trading-ai/driver.py` (a small Playwright REPL --
`chromium-cli` isn't installed in this environment, see Gotchas).

All paths below are relative to the repo root (`/home/ming/Spread_Trading_AI`).

## Prerequisites

Playwright's Python package and a downloaded Chromium build must already be
present (they were, in this environment: `pip show playwright` -> 1.61.0,
`~/.cache/ms-playwright/chromium-1234/` present). If missing:

```bash
pip install playwright
playwright install chromium
```

No `apt-get` or sudo needed even though the container has no passwordless
sudo -- see Gotchas for why.

## Setup

Both apps read `outputs/*.parquet` (produced by the deterministic pipeline)
and `.env` (API keys, Chainlit auth). If `outputs/` doesn't exist yet, run
the Streamlit app once and click "Run pipeline", or `python main.py`.

The chat app additionally needs, in `.env`:

```
CHAINLIT_AUTH_SECRET=...     # chainlit create-secret
CHAT_TRADERS=user:bcrypt_hash,...   # python chat_app/hash_password.py
```

## Run (agent path)

### Chat frontend (chat_app/)

```bash
nohup chainlit run chat_app/app.py --headless --host 0.0.0.0 --port 8010 \
  > /tmp/chat_app.log 2>&1 &
timeout 30 bash -c 'until curl -sf http://127.0.0.1:8010 >/dev/null; do sleep 1; done'
```

Drive it with the REPL driver (reads commands from stdin, one per line):

```bash
LD_LIBRARY_PATH=/home/ming/miniconda3/lib python3 \
  .claude/skills/run-spread-trading-ai/driver.py <<'EOF'
nav http://127.0.0.1:8010
fill input[name="email"] <username>
fill input[name="password"] <password>
click Sign In
wait-for textarea 20000
screenshot /tmp/login.png
type #chat-input Short the spread when the z-score is above 2, cover when it drops below 0.5.
click #chat-submit
wait-text "Reply confirm" 30000
screenshot /tmp/proposed_dsl.png
type #chat-input confirm
click #chat-submit
wait-text "Sharpe (annualized)" 60000
screenshot /tmp/backtest_result.png
quit
EOF
```

`type` (not `fill`) is required for `#chat-input` -- see Gotchas. Screenshots
are written to whatever path you pass; omit the path to get
`screenshots/<n>.png` under the current directory.

Stop it: `lsof -ti:8010 -sTCP:LISTEN | xargs -r kill`.

### Streamlit pipeline UI (app.py)

```bash
nohup streamlit run app.py --server.headless true --server.port 8502 \
  > /tmp/streamlit.log 2>&1 &
timeout 30 bash -c 'until curl -sf http://127.0.0.1:8502 >/dev/null; do sleep 1; done'
```

```bash
LD_LIBRARY_PATH=/home/ming/miniconda3/lib python3 \
  .claude/skills/run-spread-trading-ai/driver.py <<'EOF'
nav http://127.0.0.1:8502
wait-text "Pipeline Control Panel" 20000
screenshot /tmp/streamlit.png
quit
EOF
```

The driver is generic (`nav`/`fill`/`type`/`click`/`wait-for`/`wait-text`/
`screenshot`/`text`) -- it isn't chat_app-specific, and this same script
drove the Streamlit page above without modification.

Stop it: `lsof -ti:8502 -sTCP:LISTEN | xargs -r kill`.

**Ports:** this machine already runs unrelated services on 8000, 8501, and
8501's actual owner is a *different* project's Streamlit app
(`streamlit_app/app.py`, not this repo's `app.py`) -- don't kill it. Use
8010 / 8502 (or any other free port) instead, and always check
`ss -ltnp | grep <port>` before binding.

## Driver command reference

| command | what it does |
|---|---|
| `nav <url>` | navigate, waits for networkidle |
| `fill <selector> <text>` | set an `<input>` value directly (login fields only) |
| `type <selector> <text>` | click, settle 500ms, type with per-key delay (required for chat_app's `#chat-input`) |
| `click <selector-or-text>` | CSS/Playwright selector, or plain text via `:text()` |
| `press <key>` | keyboard key, e.g. `Enter` |
| `wait-for <selector> [timeout_ms]` | wait for an element |
| `wait-text <substring> [timeout_ms]` | poll `body.innerText` for a substring (quote multi-word needles) |
| `screenshot [path]` | full-page PNG |
| `text` | print `body.innerText` |
| `sleep <ms>` | fixed wait |
| `quit` | close the browser |

## Run (human path)

`streamlit run app.py` / `chainlit run chat_app/app.py -h 0.0.0.0` (see
README.md) -- opens/serves normally; useless headless, only for a human at
a real browser.

---

## Gotchas

- **Headless Chromium is missing shared libs, and there's no passwordless
  sudo to fix it the normal way.** `playwright install chromium` downloads
  two binaries: `chromium_headless_shell-*` (what `headless=True` uses by
  default) and the full `chromium-*` browser. Both are missing
  `libnspr4.so`, `libnss3.so`, `libnssutil3.so`, `libasound.so.2` on this
  container, and `apt-get install libnspr4 libnss3 ...` needs a sudo
  password we don't have. Fix: this machine's conda env
  (`/home/ming/miniconda3/lib/`) happens to already ship matching `.so`
  files (it depends on `nspr`/`nss` for unrelated reasons) -- pointing
  `LD_LIBRARY_PATH` there satisfies the missing symbols without touching
  apt. The driver launches the full `chromium-*` build (not
  `headless_shell`) with `executable_path` set explicitly; run it with
  `LD_LIBRARY_PATH=/home/ming/miniconda3/lib` in front, every time.

- **`page.fill()` on `#chat-input` leaves the send button (`#chat-submit`)
  permanently `disabled`.** It's a React-controlled autosize textarea;
  `fill()`'s synthetic value-set doesn't trigger whatever internal state
  update enables the button. Use `.type()` (real keystroke events) instead
  -- the driver's `type` command does this.

- **The first 1-2 characters typed into `#chat-input` get dropped** if you
  `type()` immediately after `click()` -- something in the component's
  mount/focus transition swallows early keystrokes. The driver's `type`
  command waits 500ms after the click before typing; don't remove that
  wait when adapting the script.

- **`wait-text "confirm"` (lowercase, unquoted) matches instantly and
  falsely.** Chainlit's own static welcome message already contains the
  word "confirm" ("...turn it into a backtestable rule for you to
  **confirm**."). Wait for a longer, response-specific phrase instead --
  `"Reply confirm to backtest"` or `"Sharpe (annualized)"` -- not a bare
  keyword that might appear in the app's boilerplate copy.

- **Ports 8000 and 8501 are already occupied by other processes on this
  machine** (8501 in particular is a *different* project's Streamlit app,
  `streamlit_app/app.py` -- not this repo's). `curl` against a "busy" port
  can still return 200 from the other service, which looks like success
  but proves nothing about this app. Always launch on a free port you
  verified with `ss -ltnp`, and check the log file for
  `Port <N> is not available` if a curl-based readiness check ever passes
  suspiciously fast.

- **Testing the chat app's full LLM -> DSL -> backtest round trip needs a
  trader login you don't have a plaintext password for** (`CHAT_TRADERS`
  in `.env` stores bcrypt hashes only). Don't reuse real trader accounts.
  Generate a disposable one, test, then remove it:

  ```bash
  python3 -c "import bcrypt; print(bcrypt.hashpw(b'TestPass123', bcrypt.gensalt()).decode())"
  # append username:hash to CHAT_TRADERS in .env, restart chainlit, test,
  # then restore .env and rm -rf chat_app/user_data/<test-username>/
  ```

  Restart the server after editing `CHAT_TRADERS` -- it's read once at
  startup via `os.getenv`, not live-reloaded.

## Troubleshooting

- **`error while loading shared libraries: libnspr4.so: cannot open shared
  object file`**: see the Gotchas entry above -- set
  `LD_LIBRARY_PATH=/home/ming/miniconda3/lib`.
- **`BrowserType.launch: Failed to launch chromium because executable
  doesn't exist at .../chrome-linux/chrome`**: wrong path -- the full
  build's directory is `chrome-linux64/`, not `chrome-linux/`.
- **`Locator.click: Timeout ... element is not enabled` on
  `#chat-submit`**: you used `fill` instead of `type` on `#chat-input`.
- **Chainlit CLI: `Error: No such option '-p'`**: current chainlit version
  uses `--port`/`--host`/`-h` (`-h` is `--headless`, a boolean flag -- it
  does NOT take a host argument, unlike some older docs/READMEs suggest).
  Use `--host 0.0.0.0 --port <N>`.
- **`ERROR: [Errno 98] address already in use`**: something else already
  bound that port. `ss -ltnp | grep <port>` to confirm, pick another port.
