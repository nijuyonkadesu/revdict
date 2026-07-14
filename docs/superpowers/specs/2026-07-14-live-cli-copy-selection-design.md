# Live CLI: Copy Selection on Enter — Design

Date: 2026-07-14
Status: approved

## Intent

Bare `revdict`'s live interactive session (see
`docs/superpowers/specs/2026-07-11-live-interactive-cli-design.md`) already
binds Enter to `execute-silent(echo {q} >> HISTORY_FILE)+clear-query` —
committing the typed query to session history and clearing the query box.
The user wants Enter to *also* copy the currently-highlighted candidate's
headword to the clipboard, on the same keypress, unconditionally — not a
new flag or opt-in, just the default shipped behavior of every live
session from now on.

Access pattern this needs to work under: the user runs `revdict` inside
tmux, itself over SSH, and wants the copy to land in the clipboard of the
device they're physically sitting at (their SSH client), not the remote
host's own system clipboard.

## Mechanism, empirically verified before writing this spec

Three things were tested directly against the real `fzf`/`tmux` binaries
on the development machine (not assumed from documentation) before this
design was finalized:

1. **`execute-silent`'s child process does not reach the pty tmux
   monitors for OSC 52.** Confirmed by writing an OSC 52 sequence to
   stdout from an `execute-silent` binding and checking tmux's own paste
   buffer — no new buffer entry appeared. Writing the same sequence
   directly to `/dev/tty` instead, still from `execute-silent`, *did*
   produce a new tmux buffer entry with the exact expected content. This
   is the difference between the feature working and silently no-op'ing.
2. **A plain (non-DCS-wrapped) OSC 52 sequence written to `/dev/tty` is
   sufficient** given `set-clipboard on` (confirmed set on this machine,
   and a common, frequently-recommended tmux configuration specifically
   for this purpose) — tmux intercepts it and forwards it up the chain
   correctly. The alternative DCS-passthrough wrapping
   (`\ePtmux;...\e\\`, gated by tmux's separate `allow-passthrough`
   option) was also tested, but its success can't be locally verified —
   passthrough deliberately bypasses tmux's own buffer, which is the one
   observable signal available without a real terminal emulator on the
   other end. Given the plain form is fully verified and DCS-wrapping
   isn't, this design uses the plain form.
3. **fzf's `{1}` placeholder substitutes safely as a single shell
   argument even when the field contains an apostrophe** — tested with
   "in someone's eyes", a real headword in this corpus. Same safety
   property already established for `{q}` during the original live-CLI
   work.

## The new CLI entry point

`revdict --copy-selection "MARKED_TEXT"`, added to `cli.py` alongside the
existing `--query-only`/`--jsonl-query` additive entry points, same
dispatch pattern (`argv[0] == "--copy-selection"`).

`format_candidate_line`'s field 1 is always exactly `f"{marker}
{headword}"`, where `marker` is either `"★"` (exact match) or `" "`
(regular candidate) — both one character, so the field is always a
2-character marker+separator prefix followed by the headword. Python
strings are Unicode-aware, so `text[2:]` correctly strips this prefix
regardless of which marker was used, with no special multi-byte handling
needed (unlike the equivalent byte-vs-character problem solved in
`revdict.nvim`'s Lua code, which doesn't have this luxury).

Dispatch logic:

```
remote = bool($TMUX or $SSH_TTY or $SSH_CONNECTION or $SSH_CLIENT)
if remote:
    write OSC 52 (base64-encoded headword) directly to /dev/tty
else:
    pipe the headword to whichever of wl-copy / xclip / xsel / pbcopy
    is found first on PATH
```

Order is deliberate, not arbitrary: `wl-copy` first, since a pure Wayland
session has no X11 server for `xclip`/`xsel` to talk to at all, while an
XWayland compatibility layer (making `xclip`/`xsel` "work" via
translation) is the less direct path when a native Wayland tool is
available; `xclip`/`xsel` next for X11; `pbcopy` last since it only
exists on macOS, where the other three are never present anyway, so its
position doesn't functionally matter but keeps the list in a sensible
"most to least likely on Linux" reading order.

Exits quietly either way — no stdout, since it's invoked via
`execute-silent` and any output would be silently discarded anyway (and
would defeat the "silent" intent if it weren't).

## The fzf binding change

`build_live_session_args`'s existing `enter` binding gains a third
chained action, appended after the two that already exist:

```
enter:execute-silent(echo {q} >> HISTORY_FILE)+clear-query+execute-silent({python_executable} -u -m revdict.cli --copy-selection {1})
```

Enter's existing behavior (commit query to history, clear the box) is
unchanged — the copy is additive, chained onto the same keypress, not a
replacement. This is unconditional: no new parameter on
`build_live_session_args`, no config flag, no opt-in. Every live session
gets this behavior by default from the moment this ships.

## Error handling

If neither OSC 52 nor a local clipboard tool is usable (no controlling
tty, no clipboard tool on `PATH`), `--copy-selection` fails silently —
consistent with `execute-silent`'s own silent-by-design nature, and with
the fact that Enter's other two actions (history commit, clear query)
succeed regardless, so a clipboard failure never blocks or visibly
disrupts the live session.

## Testing

The marker-stripping and environment-detection dispatch logic
(`remote = bool(...)` and which branch it takes) is unit-tested with the
two actual copy mechanisms mocked out, matching how `_get_search_result`
is already mocked in this codebase's existing CLI tests. The two copy
mechanisms themselves (`/dev/tty` writes, shelling out to a real
clipboard tool) are not meaningfully unit-testable — consistent with how
`run_live_session`'s own interactive pieces are already treated in this
codebase — and are validated manually instead, the same tmux-driven
technique already used and proven for this project's earlier live-CLI
work.

## Out of scope

- Any change to `--query-only` or `--jsonl-query` (revdict.nvim's entry
  point) — this is specific to the fzf-based live session's own Enter
  binding.
- DCS-passthrough wrapping — considered, tested, not used (see Mechanism
  section above).
- A visible confirmation/toast that the copy happened — `execute-silent`
  is silent by design; adding visible feedback would require `execute`
  (which switches to the alternate screen, causing a visible flicker)
  and wasn't asked for.
