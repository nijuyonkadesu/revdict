# Lock-Authoritative Daemon Startup Implementation Plan

**Goal:** Use one synchronized startup path that follows the actual daemon
process rather than a PID/socket sidecar or fixed elapsed-time deadline.

**Architecture:** `daemon.start.lock` serializes launchers. `daemon.lock` is
held for the server process lifetime and contains its PID, process-start
identity, and `starting`/`ready` phase. The socket signals request readiness.

## Constraints

- Never terminate or abandon a daemon merely because initialization is slow.
- Never spawn a competing daemon while the lifetime lock is held.
- Never remove runtime sidecars unless both startup and lifetime locks are
  held by the cleanup path.
- Preserve detached execution and the existing socket protocol.
- Route explicit and automatic startup through the same coordinator.
- Isolate every daemon runtime path in every test.

## Tasks

- [x] Add an autouse fixture for socket, PID, lifetime-lock, startup-lock, and
  log paths.
- [x] Add a validated lifetime-lock record with PID, `/proc` start identity,
  and startup phase.
- [x] Make status distinguish stopped, starting, running, and unhealthy.
- [x] Publish `starting` before expensive imports and `ready` after the socket
  listens.
- [x] Replace the 20-second startup deadline with process/lock-driven waiting.
- [x] Serialize concurrent callers, hand the startup-lock descriptor to the
  child, and adopt an already-starting lock owner.
- [x] Restrict stale-sidecar cleanup to the serialized, lifetime-lock-held
  path.
- [x] Wait for a validated prior process record to finish teardown before
  clearing it or spawning a replacement.
- [x] Route query, native UI, and explicit daemon startup through the
  coordinator.
- [x] Make stop validate the lifetime-lock PID identity and wait for that
  process to finish teardown before allowing restart.
- [x] Run focused and full test suites, exercise the real CLI lifecycle, and
  stage the reviewed replacement without committing or pushing.
