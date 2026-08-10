# Lock-Authoritative Daemon Startup Coordination

Date: 2026-08-10
Status: approved for implementation

## Problem

Automatic startup and `revdict daemon start` used different launch paths. They
also treated a missing socket or an elapsed 20-second deadline as proof that no
daemon was running. During the daemon's multi-gigabyte model load, however, the
socket does not exist yet. A second client could therefore launch another
daemon or fall back to an equally expensive local load.

PID and socket sidecars are not safe ownership signals: either can be stale or
deleted while its process remains alive. Startup coordination must instead be
based on state whose ownership the operating system releases when the process
exits.

## Authoritative state

`daemon.lock` is the daemon's lifetime lock and the sole authority for daemon
ownership. The lock file remains on disk. While holding its advisory lock, the
daemon writes one JSON record into it:

```json
{"pid":1234,"identity":"987654","phase":"starting"}
```

`identity` is Linux's process start-time tick from `/proc/<pid>/stat`. Checking
both PID and start identity prevents a stale record from identifying a reused
PID. `phase` is `starting` until initialization finishes and `ready` only after
the Unix socket is bound and listening.

The PID file remains a compatibility sidecar. Neither it nor the socket path
decides whether a daemon process exists.

## One startup coordinator

Every external launch route calls `ensure_daemon_running()`, including query
startup, native-UI startup, and `revdict daemon start`. Only the daemon child
entered with `REVDICT_DAEMON_CHILD=1` calls `run_server()` directly.

The coordinator:

1. returns immediately for a reachable socket;
2. takes `daemon.start.lock` with a blocking advisory lock so concurrent
   callers serialize without each running their own timeout loop;
3. rechecks readiness;
4. if `daemon.lock` is held, adopts that owner and waits while its `starting`
   record remains current;
5. if `daemon.lock` is free, holds it and waits for any validated prior record
   to finish process teardown, then clears the record and stale socket/PID
   sidecars before spawning exactly one detached child;
6. waits until the socket becomes reachable or the spawned process actually
   exits.

The detached child inherits the launcher's locked startup-file descriptor.
After acquiring `daemon.lock` and publishing `starting`, the child closes that
descriptor. This hands ownership directly from startup lock to lifetime lock:
even if the launcher is interrupted or killed immediately after spawning,
another caller cannot enter the gap and create a second child.

Waiting on the previous record after its lock is released matters because
Python can spend seconds unloading the model after `run_server()` returns. It
prevents any caller—not only `daemon stop`—from overlapping that teardown with
a new multi-gigabyte load.

There is no elapsed-time startup failure. A slow but live daemon remains the
one startup in progress. A `ready` lock owner whose socket is unreachable is
unhealthy: callers do not kill it, delete its files, or start a competing
process.

If a waiting coordinator dies, its startup lock is released by the kernel.
The detached daemon retains its lifetime lock; the next caller adopts that
same process.

## Daemon lifecycle

`run_server()` acquires `daemon.lock` before expensive imports, immediately
publishes the `starting` record, and installs termination handlers. After
loading the index, binding the socket, and calling `listen()`, it updates the
same locked record to `ready`.

On exit, only the lock-owning server removes the socket and PID sidecars it
created. The lifetime lock is released automatically on any process exit, even
when ordinary cleanup cannot run. The lock file itself is never deleted.

## Status and stop

Status derives from the lifetime lock and its validated record:

- free lifetime lock: not running;
- held lock with `starting`: starting;
- held lock with `ready` and a reachable socket: running;
- held lock with `ready` and no reachable socket: unhealthy.

`daemon stop` signals only the PID whose start identity matches the record in
the currently held lifetime lock. It then waits for that exact process
identity to exit, even if the process releases its lock earlier during Python
teardown, so a restart cannot overlap two multi-gigabyte processes. It never
signals a PID taken only from a sidecar.

## Test isolation

Every pytest test replaces all five daemon runtime paths—socket, PID, lifetime
lock, startup lock, and log—with paths under that test's temporary directory.
This prevents a failing test from deleting or probing a user's live daemon
state.

Regression tests cover concurrent single-spawn behavior, adoption of a slow
live startup without a deadline, concrete child failure, stale record races,
unhealthy owners, cleanup lock ownership, shared CLI routing, and runtime-path
isolation.
