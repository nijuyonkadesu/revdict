# revdict

A local, offline reverse-dictionary CLI. Give it a word and it shows the
standard definition; give it a phrase describing a meaning and it suggests
matching words — every result tagged with an emotion/connotation badge.
Runs entirely on-device (WordNet, Wiktionary, and small local ML models),
no API keys, no per-query network calls.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [`fzf`](https://github.com/junegunn/fzf) for the interactive picker (optional — falls back to a plain printed list if absent)

## Install

```bash
uv sync --all-extras
```

This creates `.venv/` and installs everything, including a CPU-only PyTorch
build (no CUDA download, regardless of platform). Symlink the entry point
onto your `PATH` so you can drop the `.venv/bin/` prefix everywhere below:

```bash
ln -sf "$(pwd)/.venv/bin/revdict" ~/.local/bin/revdict
```

## First-time setup

Build the local search index once (downloads WordNet, a Wiktionary extract,
and a few small ML models; takes on the order of 30 minutes depending on
your machine):

```bash
revdict build-index
```

Re-run this any time you want to refresh the underlying data.

## Usage

```bash
# Interactive picker (fzf) — arrow keys + live preview, ? to toggle preview,
# Enter to print the selected word
revdict "happy"
revdict "feeling of intense annoyance"

# One-shot, plain-text output (no fzf) — good for scripting
revdict "happy" --no-interactive

# Show more/fewer candidates (default 30)
revdict "happy" --no-interactive -n 10
```

The first query when no daemon is running starts a background daemon that keeps the index
and models warm in memory, so subsequent queries are fast:

```bash
revdict daemon status   # is it running?
revdict daemon stop     # stop it (e.g. before rebuilding the index)
```

If you rebuild the index while a daemon is running, it keeps serving the old
data until you stop it — `build-index` will remind you if this applies.

## Clipboard copy on Enter

In the interactive picker, pressing Enter on a highlighted candidate copies
it to your clipboard, in addition to selecting it. Over SSH and/or inside
tmux, this goes through the terminal's OSC 52 escape sequence — reaching the
clipboard of the device you're physically using, not the remote host's own
clipboard — provided your terminal emulator and tmux's `set-clipboard`
support it. Otherwise it falls back to whichever of `wl-copy`, `xclip`,
`xsel`, or `pbcopy` is available locally.

## Optional: stress-marked pronunciation

If you also have the [`emphasis`/`stressmark`](https://github.com/nijuyonkadesu/emphasis)
project cloned locally, installing it into revdict's own venv adds a
"Stress" column/line to results (e.g. `HAPpy`) showing primary/secondary
syllable stress:

```bash
uv pip install -e /path/to/emphasis
```

This is a fully optional plugin — `stressmark` is never a declared
dependency of `revdict`, so nothing changes for anyone who doesn't install
it. Since the daemon loads it once at startup, run `revdict daemon stop`
after installing or uninstalling it so the next query picks up the change.
