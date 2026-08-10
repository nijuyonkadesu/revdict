# Process-Aware Daemon Startup Coordination

Date: 2026-08-10
Status: approved for implementation

## Problem

Automatic startup from a query and explicit startup through
`revdict daemon start` currently take different paths. Automatic startup uses
the startup lock, while explicit startup calls the unsynchronized spawn helper
directly. Both paths also stop waiting after a fixed 20 seconds, even when the
daemon process is still alive and legitimately loading the index and models.
This can create redundant child processes and cause clients to fall back to a
second, memory-intensive local model load while the daemon is still starting.

## Design

All external callers will use `ensure_daemon_running()`. The low-level detached
spawn operation will remain an implementation detail and will return the child
process handle instead of deciding readiness itself.

The daemon child will publish its PID immediately after it acquires the
lifetime server lock and before it begins expensive imports. Socket creation
will remain the readiness signal: a live PID without a reachable socket means
"starting," while a reachable socket means "ready."

`ensure_daemon_running()` will:

1. Return immediately when the daemon socket is reachable.
2. If a published daemon PID is alive, wait for its socket while that process
   remains alive.
3. Otherwise acquire the startup lock, recheck the socket and PID state, and
   spawn exactly one detached child when needed.
4. Wait without an arbitrary startup deadline. Succeed when the socket becomes
   reachable and fail only when the observed child or published daemon PID
   exits before readiness.
5. Release the startup lock after readiness or confirmed process failure so a
   later caller may make a fresh attempt.

A caller waiting on another launcher will periodically retry acquisition and
re-evaluate socket/PID state. If the original launcher exits unexpectedly but
its daemon child remains alive, the early PID allows subsequent callers to
adopt and wait for that same startup rather than spawning another child.

The detached child remains independent of the invoking terminal. Interrupting
a waiting CLI process will not terminate the daemon child.

## CLI Behavior

`revdict daemon start` and query-triggered startup will both call
`ensure_daemon_running()`. Explicit startup will print `Daemon started.` after
the socket becomes reachable. If the daemon exits before readiness, it will
print a failure message that points to the daemon log.

`daemon_status()` will distinguish three states:

- ready: PID alive and socket reachable;
- starting: PID alive but socket not yet reachable;
- stopped: no live published PID and no reachable socket.

## Failure and Cleanup

The daemon will install its termination handlers before expensive loading so
`revdict daemon stop` can stop a starting daemon cleanly. A child that fails
during startup will remove its PID and socket files in `run_server()` cleanup.
Unexpected hard termination may leave stale files; launchers will treat a PID
whose process is no longer alive as stale and clean it before a new launch.

There is deliberately no automatic retry loop after a confirmed child failure:
deterministic model or index errors should be surfaced instead of spawning
forever. A later invocation may attempt startup again.

## Tests

Tests will prove that:

- explicit and automatic startup use the same synchronized path;
- concurrent callers spawn only one child;
- a startup lasting longer than the former 20-second deadline continues
  waiting while the daemon PID is alive;
- readiness is reported only after the socket accepts connections;
- child exit before socket readiness returns failure;
- status reports `starting` for a live PID without a socket;
- termination during startup cleans up published state;
- detached process configuration remains unchanged.

Tests will use temporary PID/socket/lock paths and lightweight stand-in
processes or in-process socket servers; they will not load the real models.

## Scope

This change does not alter query handling, daemon request concurrency, daemon
idle lifetime, search fallback after a confirmed startup failure, or the Unix
socket protocol.
