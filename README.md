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

## Query syntax

Beyond plain word lookups and free-text meaning search, `revdict` understands
a small pattern-matching DSL, typed directly into the same prompt (works in
both the live session and one-shot `revdict "..."` queries):

| Query | Matches |
|---|---|
| `bluebird` | Exact word lookup / free-text meaning search (unchanged default) |
| `blue*` | Words starting with "blue" |
| `*bird` | Words ending with "bird" |
| `bl????rd` | Starts with "bl", ends with "rd", 4 letters between |
| `?????` | Any 5-letter word |
| `*y*` | Words containing "y" anywhere |
| `?????,*y*` | Combine clauses with a comma (AND): 5 letters AND contains "y" |
| `//fuljyo` or `//fuljyo//` | Anagram/unscramble: words using exactly these letters |
| `-abcd` | Words that don't contain any of these letters (one-shot CLI use needs `revdict -- -abcd` — the leading `-` otherwise looks like a flag to argparse; unaffected in the live session) |
| `+abcd` | Words built only from these letters |
| `bl*:snow` | Starts with "bl" AND related in meaning to "snow" |
| `:snow` | Meaning search, explicit form (same as typing `snow` directly) |
| `**winter**` | Multi-word phrases containing the whole word "winter" |
| `expand:nasa` | Phrases whose initials spell "nasa" |

Note: `*`, `?`, `#`, `@`, `//` (anywhere in the string), and a leading
`+`/`-` are pattern-syntax triggers, so a free-text meaning query
containing one of those (e.g. "a word for asking a question?") will be
parsed as a pattern instead. A `:` anywhere in a meaning query has the
same effect -- it splits the query into a pattern part (before the colon)
and a meaning part (after), so e.g. "note: a written record" is parsed as
a pattern search for the literal word "note" combined with a meaning
search for "a written record," not a single meaning query. Prefix the
query with `:` (with nothing before it) to force plain meaning search
explicitly.

## Sort order

By default, results are ordered by relevance ("most similar" to your
query). Override this with `--sort`:

| `--sort` value | Order |
|---|---|
| `relevance` (default) | Most similar first (semantic match quality) |
| `alpha` | A → Z |
| `alpha_desc` | Z → A |
| `shortest` | Shortest word first |
| `longest` | Longest word first |
| `most_common` | Most common in modern published fiction first |
| `least_common` | Least common in modern published fiction first |

```bash
revdict "happy" --sort alpha --no-interactive
revdict "blue*" --sort longest --no-interactive
```

`most_common`/`least_common` reuse the same literary-frequency data that
already nudges the default relevance ranking — a word with no frequency
data at all (very rare hyphenated/multi-word entries) sorts as if it had
zero frequency.

## Category filter

Results default to matching any part of speech or register. Narrow them with `--category`:

| `--category` value | Matches |
|---|---|
| `all` (default) | Everything |
| `noun` | Nouns only |
| `adjective` | Adjectives only |
| `verb` | Verbs only |
| `adverb` | Adverbs only |
| `idiom_slang` | Idiomatic phrases, slang, vulgar, and colloquial senses |
| `old` | Archaic, dated, obsolete, and historical senses |

```bash
revdict "feeling of intense annoyance" --category adjective --no-interactive
```

`noun`/`adjective`/`verb`/`adverb`/`all` work with any existing index. `old` relies entirely on Wiktionary's register tags, which are only captured starting with this version — it comes back empty on an older index (not error) until you run `revdict build-index` to rebuild. `idiom_slang` also uses those tags, but additionally matches on part of speech (`phrase`/`proverb`), a field that's always been in the metadata — so it already returns those pos-based matches on an old index, and simply gains the extra slang/idiomatic/vulgar/colloquial tag-based matches once you reindex.

## Phonetic filters

Five filters based on pronunciation, computed from a `revdict build-index` reindex (see below) — combine any of them, and combine them with `--category`/`--sort` too:

| Flag | Matches |
|---|---|
| `--syllables N` | Headwords with exactly N syllables |
| `--primary-vowel VOWEL` | Headwords whose stressed syllable's vowel is VOWEL (an ARPAbet vowel symbol — AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW) |
| `--rhymes-with WORD` | Headwords that rhyme with WORD |
| `--sounds-like WORD` | Headwords that are phonetically close to WORD (not just spelled similarly) |
| `--meter PATTERN` | Headwords whose stressed/unstressed syllable pattern matches PATTERN — a string of `/` (stressed) and `x` (unstressed), one character per syllable, e.g. `/x` (trochee, like "happy"), `x/` (iamb, like "record" the verb), `/xx` (dactyl, like "elephant") |

```bash
revdict "feeling of intense annoyance" --syllables 2 --no-interactive
revdict "small carnivore" --rhymes-with hat --no-interactive
```

**Requires a reindex.** Unlike category filtering, none of these five work at all on an index built before this feature shipped — run `revdict build-index` to rebuild. Phonetic data is only computed for single-word headwords with no internal hyphen (multi-word phrases and hyphenated compounds are skipped — the underlying `stressmark` library doesn't reliably syllabify either yet); those headwords simply never match any phonetic filter, on any index.

`--rhymes-with`/`--sounds-like` additionally need the `stressmark` library installed and importable at query time (not just at index-build time) — they resolve your target word's pronunciation live, since it's not something a reindex could have precomputed. If `stressmark` isn't installed, these two flags fail with a clear error rather than silently returning no results.
