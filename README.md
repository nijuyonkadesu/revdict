# revdict

A local, offline reverse-dictionary CLI. Give it a word and it shows the
standard definition; give it a phrase describing a meaning and it suggests
matching words — every result tagged with an emotion/connotation badge.
Runs entirely on-device (WordNet, Wiktionary, and small local ML models),
no API keys, no per-query network calls.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- [`fzf`](https://github.com/junegunn/fzf) for the optional one-shot picker and legacy live mode

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
# Native live terminal UI — query syntax, sort/category controls, phonetic
# filters, results, and preview remain on screen together
revdict

# One-shot fzf picker — Enter prints the selected word
revdict "happy"
revdict "feeling of intense annoyance"

# One-shot, plain-text output (no fzf) — good for scripting
revdict "happy" --no-interactive

# Show more/fewer candidates (default 30)
revdict "happy" --no-interactive -n 10

# Optional legacy fzf live session
revdict --fzf
```

### Native live UI

Bare `revdict` opens the native terminal UI. Type ordinary reverse-dictionary
queries or any form from the [query syntax](#query-syntax) table. It shows up
to 50 results, with the list, filters, preview, and a one-line live search
progress status that disappears when idle. It shows the completed percentage, current phase, phase
number, and live details such as the model or index currently loading. `F2` opens every supported filter:
Sort and Category are arrow-key/mouse radio lists, while phonetic filters are
validated text fields. `F1` is generated from those control definitions and
the keybinding registry, so it always reflects the UI.

The UI uses your terminal's ANSI palette rather than a bundled theme. Chrome
(section rules, titles, part-of-speech tags, and scrollbars) stays in the
terminal's default foreground, using only bold or dim attributes. Colour is
reserved for headwords, positive/negative sentiment, match confidence, and the
focused footer action. Headwords default to green; set `REVDICT_ACCENT` to
`green`, `yellow`, or `magenta` to choose another terminal-palette accent. It automatically uses 24-bit output only when
`COLORTERM` advertises `truecolor` or `24bit`; `revdict --truecolor` makes the
same explicit request but never overrides a terminal that does not advertise
support. `NO_COLOR` disables every colour, including stress, while preserving
readability attributes. Stress marks otherwise retain a fixed gold accent that
does not depend on a terminal theme's yellow slot. Selection reverses only the
terminal's own foreground/background across every wrapped result line. Results
use fixed headword and part-of-speech columns, with definitions aligned at a
stable hanging indent. Results and previews have proportional, arrow-free
scrollbars: mouse-wheel scrolling works in either pane, and words wrap only at
word boundaries. Footer actions use bracketed key codes such as `[F1] Help`, so
they remain visibly interactive even when colours are disabled.

| Key | Action |
|---|---|
| `F1` | Toggle generated query, filter, and keybinding help |
| `F2` | Open/close search controls (`↑` / `↓` choose Sort and Category) |
| `F3` | Toggle the preview pane |
| `F4` | Toggle writing chat; Enter sends while its input has focus |
| `F5` | Open/save the active chat provider’s local settings |
| `Ctrl-R` | Cycle sort order |
| `Ctrl-N` / `Ctrl-P` | Select next/previous result |
| `Enter` | Copy the selected headword |
| `Esc` | Clear the query; press again on an empty query to quit |
| `Ctrl-C` | Quit immediately |

The native live UI does not require fzf. `revdict --fzf` retains the earlier
fzf session for users who prefer it; one-shot interactive queries continue to
use fzf when it is installed.

### Writing chat

`F4` opens the writing-assistant pane. It pre-fills a request using the active
search query plus the highlighted word and definition; press `Enter` to send
it, then continue the conversation in the same pane. `F5` opens editable
provider settings and saves them locally.

Configure a provider before first use. This stores endpoints, model choices,
and any supplied API key in `~/.config/revdict/chat.json` with `0600`
permissions; no credential is ever stored in the repository. `--test` makes a
single inexpensive models request and never asks a model to generate text:

```bash
# OpenAI-compatible servers, including Ollama-compatible endpoints.
revdict chat-config --provider ollama --endpoint http://localhost:11434 --model my-model --test

# Gemini: --test also discovers and saves compatible text-chat models.
revdict chat-config --provider gemini --api-key "$REVDICT_GEMINI_API_KEY" --model gemini-3.6-flash --test

# OpenAI and Anthropic use their ordinary default endpoints; endpoints remain editable.
revdict chat-config --provider openai --model gpt-4.1-mini --api-key "$OPENAI_API_KEY" --test
revdict chat-config --provider anthropic --model claude-sonnet-4-5 --api-key "$ANTHROPIC_API_KEY" --test
```

An API key can instead be supplied by the provider-specific environment
variable shown above. Ollama's key is optional (`REVDICT_OLLAMA_API_KEY`).
The provider test only lists models, so it does not load or spam a slow local
generation model.

The first query when no daemon is running starts a background daemon that keeps the index
and models warm in memory, so subsequent queries are fast. The native UI uses
the daemon's progress protocol; if it finds an older daemon, it restarts it
once automatically before replaying the current query. This keeps exactly one
daemon running while enabling truthful progress details:

```bash
revdict daemon status   # is it running?
revdict daemon stop     # stop it (e.g. before rebuilding the index)
```

If you rebuild the index while a daemon is running, it keeps serving the old
data until you stop it — `build-index` will remind you if this applies.

## Clipboard copy on Enter

In the native live UI, pressing Enter on a highlighted candidate copies it to
your clipboard. The legacy fzf live session does the same. Over SSH and/or
inside tmux, this goes through the terminal's OSC 52 escape sequence —
reaching the clipboard of the device you're physically using, not the remote
host's own clipboard — provided your terminal emulator and tmux's
`set-clipboard` support it. Otherwise it falls back to whichever of `wl-copy`,
`xclip`, `xsel`, or `pbcopy` is available locally.

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
| `most_formal` | Most formal-register first (e.g. "lavatory" before "toilet" before "khazi") |
| `oldest` | Most archaic/dated/obsolete/historical-tagged first |
| `most_modern` | Least archaic/dated/obsolete/historical-tagged first |
| `most_lyrical` | Smoothest-sounding (fewest/shortest consonant clusters) first -- experimental |

```bash
revdict "happy" --sort alpha --no-interactive
revdict "blue*" --sort longest --no-interactive
revdict "toilet" --sort most_formal --no-interactive
```

`most_common`/`least_common` reuse the same literary-frequency data that
already nudges the default relevance ranking — a word with no frequency
data at all (very rare hyphenated/multi-word entries) sorts as if it had
zero frequency.

`most_formal`/`oldest`/`most_modern` reuse the same Wiktionary register
tags `--category old`/`--category idiom_slang` are built on (see below) —
they need the same reindex those categories need on an older index.
`most_formal` ranks by the formal ↔ informal spectrum only, not by
subject-matter domain — a legal term like "writ" is not specifically
detected as legal, only as (in this case) untagged/neutral register; true
topic/domain detection (e.g. distinguishing legal terms specifically) is
not yet implemented. These three sorts rank by whichever single sense of
a word actually matched your query, not by whether the word has *any*
tagged sense anywhere — a word like "glad" can have separate obsolete and
informal senses in Wiktionary, but if your query matches its plain,
untagged sense, it sorts as neutral/formal-tied rather than as old or
informal. Untagged/tied candidates keep their original relevance order
rather than moving to an arbitrary position.

`most_lyrical` is an experimental approximation of "smooth/euphonious
sounding" based on average consonant-cluster length in the word's
pronunciation — it needs the same `revdict build-index` reindex
`--syllables`/`--meter`/etc. need (see "Phonetic filters" below), and
words without precomputed phonetics data (multi-word/hyphenated
headwords, or an un-reindexed older index) sort last rather than being
excluded.

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

`--rhymes-with`/`--sounds-like` resolve their target word's pronunciation as a **noun** by default, since the CLI doesn't collect a part of speech for the target word — this matters for a heteronym like "record" (`--rhymes-with record` uses the noun pronunciation "REH-kerd", not the verb "ri-KORD"), which can produce a different rhyme key than you might expect.

Rhyme/sounds-like matches for obscure words not in the CMU Pronouncing Dictionary rely on a machine-predicted (G2P) pronunciation, which isn't always linguistically correct — so an occasional odd match (e.g. a rare compound word matching a target it doesn't actually rhyme with) is expected, not necessarily a bug.
