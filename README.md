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
build (no CUDA download, regardless of platform).

## First-time setup

Build the local search index once (downloads WordNet, a Wiktionary extract,
and a few small ML models; takes on the order of 30 minutes depending on
your machine):

```bash
.venv/bin/revdict build-index
```

Re-run this any time you want to refresh the underlying data.

## Usage

```bash
# Interactive picker (fzf) — arrow keys + live preview, ? to toggle preview,
# Enter to print the selected word
.venv/bin/revdict "happy"
.venv/bin/revdict "feeling of intense annoyance"

# One-shot, plain-text output (no fzf) — good for scripting
.venv/bin/revdict "happy" --no-interactive

# Show more/fewer candidates (default 30)
.venv/bin/revdict "happy" --no-interactive -n 10
```

The first query in a while starts a background daemon that keeps the index
and models warm in memory, so subsequent queries are fast:

```bash
revdict daemon status   # is it running?
revdict daemon stop     # stop it (e.g. before rebuilding the index)
```

If you rebuild the index while a daemon is running, it keeps serving the old
data until you stop it — `build-index` will remind you if this applies.
