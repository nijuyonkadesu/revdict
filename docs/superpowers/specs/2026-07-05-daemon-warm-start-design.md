# Background Daemon for Warm-Start Queries — Design

Date: 2026-07-05
Status: approved

## Problem

Every `revdict` invocation is a fresh process: it loads `embeddings.npy`
(~2GB), parses `metadata.jsonl` (~436k records), and constructs the
bi-encoder, cross-encoder, and (lazily) the emotion classifier from scratch.
For a tool meant to be used repeatedly throughout the day (the user's
original use case: keyboard-shortcut-triggered lookups), this cold-start
cost on every single query is the main remaining usability friction.

## Design

A new `revdict daemon` process loads the index and models exactly once and
serves queries over a local Unix domain socket
(`~/.cache/rev_dictionary/daemon.sock`, PID file alongside it at
`daemon.pid`). It reuses the existing `search.search()` function unchanged —
the daemon is purely a transport wrapper around already-tested logic, not a
reimplementation.

`revdict word` becomes daemon-aware:
1. Try connecting to the socket and sending the query.
2. If nothing's listening or the connection is refused (including a stale
   socket left behind by a crashed daemon — e.g. `kill -9`), clean up any
   stale socket file, spawn a detached `revdict daemon start` process in the
   background, and poll briefly for the socket to become ready.
3. If the daemon fails to start within a timeout, fall back to today's
   direct in-process search — the tool never hard-fails because of daemon
   problems, worst case it's just as slow as before.

The daemon stays running until explicitly stopped (`revdict daemon stop`) or
the machine reboots — no idle timeout. `revdict daemon status` reports
whether it's currently running.

**Critical performance detail:** today, `cli.py` imports `revdict.search` at
module load time, which transitively imports `sentence_transformers`/`torch`
— so even with a warm daemon, the *client* process would still pay to load
those libraries before ever reaching the socket code. That import must
become lazy: `cli.py` only imports `revdict.search` inside the fallback
branch (daemon unavailable and failed to start). In the common case (daemon
already running), the client process never imports those heavy libraries at
all — it's just `socket` + `json` + `rich` + spawning `fzf`, about as fast as
a CLI can start.

The `HF_HUB_OFFLINE` / quiet-progress-bar environment setup (added for the
non-daemon cold-start case) moves from `cli.py` to `daemon.py` — that's now
the process that actually constructs the models, with the same
before-any-other-import ordering requirement.

## Module: `src/revdict/daemon.py`

- `run_server() -> None` — binds the socket, blocking accept loop; per
  connection, reads one line of JSON (`{"query": str, "top_n": int}`), calls
  `search.search(query, top_n=top_n)`, writes back one line of JSON (the
  same result shape `_print_static_results`/`run_picker` already consume, or
  `{"error": str}` if `search()` raised), closes the connection. Single
  request at a time — acceptable for a personal tool.
- `send_query(query: str, top_n: int) -> dict | None` — client-side: connects
  with a short timeout, sends the request, reads the response. Returns
  `None` (never raises) if the socket is absent, refuses connection, or
  times out — the caller's signal that no daemon is available.
- `ensure_daemon_running() -> bool` — called when `send_query` first returns
  `None`: removes a stale socket file if present, spawns
  `revdict daemon start` as a detached background process (own session,
  stdout/stderr redirected to a log file under the cache dir, not waited on
  in the foreground), then polls the socket path briefly until it accepts
  connections or a timeout elapses. Returns whether the daemon became
  reachable in time.
- `stop_daemon() -> bool` — reads the PID file, signals the process to shut
  down cleanly (removing its own socket/PID files on exit), returns whether
  a running daemon was found to stop.
- `daemon_status() -> str` — human-readable running/not-running report,
  used by the `daemon status` subcommand.

## `cli.py` changes

- New subcommand routing: `revdict daemon start|stop|status`.
- `_run_query`'s search step becomes: try `daemon.send_query(...)`; if
  `None`, call `daemon.ensure_daemon_running()` and retry once; if still
  `None`, fall back to a lazily-imported `revdict.search.search(...)` call
  (today's direct in-process path).
- The top-level `from revdict import search as search_mod` import is removed
  from module scope and moved inside the fallback branch.

## Error handling / edge cases

- **Daemon crashes** (e.g. OOM, `kill -9`): socket file may be left behind.
  The client's connection attempt fails with connection-refused; treated
  identically to "not running" — stale socket removed, fresh daemon spawned.
- **Two near-simultaneous first queries, no daemon yet**: both may race to
  spawn a daemon; whichever process loses the `bind()` race on the socket
  path exits immediately, and its client ends up waiting for (and using) the
  daemon the *other* process successfully started. No separate lock file
  needed.
- **Daemon fails to start** (e.g. genuine error loading a model): client's
  poll times out, falls back to in-process search — daemon problems never
  block a query from completing.
- **`revdict daemon stop` with nothing running**: prints a clear
  "not running" message rather than erroring.
- **`build-index` while a daemon is running**: no automatic coordination —
  `build-index` detects a running daemon (via the PID file) when it
  finishes and prints a reminder to run `revdict daemon stop` so the next
  query picks up the refreshed index. The daemon does not watch for index
  changes on disk.

## Testing

- `daemon.py`'s request/response JSON framing and the
  stale-socket-vs-refused-connection classification get unit tests using a
  real local socket pair in-process (the test server side echoes a canned
  response — no real model loading), following the existing pattern used for
  `picker.py`'s subprocess-mocking tests.
- `ensure_daemon_running`'s spawn-and-poll logic gets tests via monkeypatched
  `subprocess.Popen` and socket-connect attempts, not a real spawned process.
- The daemon's use of `search.search()` itself is not re-tested — already
  covered by existing tests; this task only tests the transport layer.
- Manual validation (this task, not automated): start a daemon, run several
  queries confirming they're fast and correct, check `daemon status`, stop
  it, confirm the next query transparently auto-spawns a fresh one again.

## Out of scope

- Handling concurrent queries in parallel (sequential is fine for personal
  use).
- Watching the index files for changes and auto-restarting the daemon.
- An idle-timeout auto-shutdown (explicitly declined by the user — the
  daemon runs until stopped or reboot).
