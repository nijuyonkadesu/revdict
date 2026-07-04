# Reverse Dictionary CLI — Design

Status: **awaiting user review**
Date: 2026-07-04

## Problem

The user has a bash script bound to a keyboard shortcut that looks up a word's
definition via `api.dictionaryapi.dev` and shows it as a desktop notification.
That's fine for "define this exact word," but it can't do the opposite: given a
*description of a meaning* (a word that doesn't quite fit, or a whole phrase),
suggest better words. It's also API-dependent (rate limits, network required).

The user wants a **local, offline, long-term-use tool** that:

1. Takes a word or a phrase/sentence as input.
2. If the input is itself a real word, shows its standard dictionary entry
   (definitions, part of speech) — parity with the existing script.
3. Always additionally runs a **reverse-dictionary search**: given the input as
   a description of intended meaning, surfaces a ranked list of candidate
   words that better fit.
4. For every word shown (exact match and candidates), also shows a detailed
   definition and an **emotional/sentiment tag** (positive/negative/neutral,
   plus richer emotion categories).
5. Is a CLI (not a notification popup) — the user explicitly wants this to
   live in the terminal, as a separate tool from the existing shortcut script
   (which can keep working as-is for quick popups).
6. Runs entirely locally, no API rate limits, CPU-only (no discrete GPU
   available — AMD integrated graphics only; 16 cores / 60GB RAM / ~760GB free
   disk).

Explicitly out of scope for this project: phonetic/IPA stress-mark annotation
(mentioned by the user as prior, unrelated work — not a requirement here).

## Data pipeline (offline, one-time build)

A one-time `revdict build-index` command builds a local corpus + vector index;
everyday use only ever *reads* this prebuilt index — no network calls, no
re-embedding, at query time.

1. **WordNet** (via the `wn` Python package, ~35MB download) — curated
   definitions, part-of-speech, synonyms/hypernyms for ~150k word-senses.
   This is the structured, reliable core.
2. **Wiktionary supplement** — pulled from **kaikki.org**, a project that
   publishes the entire English Wiktionary pre-parsed into clean JSON Lines
   (one sense per line: word, POS, gloss, examples). This avoids writing a
   MediaWiki wikitext parser ourselves, while still getting Wiktionary's
   larger vocabulary (slang, modern usage, more conversational definitions,
   example sentences). The raw extract is filtered before use: drop
   non-English-target entries, drop pure inflected/"form of" entries (plurals,
   verb conjugations that just point back to a lemma — these would otherwise
   flood the corpus with near-duplicate senses), but keep multi-word
   entries/idioms, since a phrase like "green with envy" is a legitimate
   reverse-dictionary answer.
3. **Sentiment/emotion lexicons** — **SentiWordNet** (positive/negative/
   objective scores *per WordNet synset* — free, tiny, and aligns directly
   with the WordNet senses already being loaded) and the **NRC Emotion
   Lexicon (EmoLex)** (word-level association with 8 emotions + 2 sentiments,
   ~14k English words). These are looked up directly, no model inference
   needed, and are the *primary* source of the emotion tag (see Query flow
   step 5 for why).
4. **Merge**: combine WordNet + Wiktionary into one corpus of `(headword, part
   of speech, definition, source, examples)` records, deduplicating
   near-identical senses between the two sources for the same word.
5. **Embed**: every definition/gloss is embedded with a small sentence-embedding
   model, **BAAI/bge-small-en-v1.5** (~130MB, 384-dim, CPU-fast). This model is
   chosen specifically because it's tuned for *asymmetric* retrieval (a short
   query describing a meaning → a longer passage/definition that matches),
   which is exactly the reverse-dictionary task — as opposed to symmetric
   similarity models tuned for comparing two sentences of similar length.
   Passages (definitions) are encoded as-is; queries are encoded with the
   model's recommended instruction prefix prepended (see Query flow step 1) —
   asymmetric models like this one need that prefix on the query side to
   retrieve well.
6. **Store**: the resulting embedding matrix + corpus metadata is saved to
   `~/.cache/rev_dictionary/index/` (a `.npy` matrix + a small metadata table).
   No FAISS/hnswlib dependency is needed — brute-force numpy cosine similarity
   over a few hundred thousand rows runs in well under 100ms on this hardware,
   so we keep the dependency footprint minimal.

**Build time is not yet known and won't be committed to as a number here** —
bge-small's throughput on CPU for short definition text, and the actual size
of the corpus after kaikki filtering, both need to be measured. The
implementation plan should benchmark encoding throughput on a small sample
(~1k definitions) first, then extrapolate to the full corpus; if that
projects to too long a build, the fallback options are tighter kaikki
filtering, ONNX export, or int8 quantization of the bi-encoder. This build
step is a one-time cost (or refresh-on-demand), not part of the per-query
path.

## Query flow

Everyday use loads the prebuilt index + three small local models into memory
(a few seconds) and then serves queries with no network access:

1. **Embed** the user's input (word or phrase) with the same bi-encoder used
   to build the index, with the model's recommended query-instruction prefix
   prepended (e.g. "Represent this sentence for searching relevant passages: ")
   — definitions were embedded without this prefix at index-build time, so
   the asymmetry is what makes retrieval work well.
2. **Retrieve** the top ~75 candidates by cosine similarity against the whole
   corpus.
3. **Rerank** those ~75 with a small cross-encoder,
   **cross-encoder/ms-marco-MiniLM-L-6-v2** (~90MB), which scores the
   (query, definition) pair jointly for a more precise final ordering than
   embedding similarity alone gives.
4. **Dedupe** by headword (a word may have multiple matching senses; keep its
   best-scoring sense) and take the top 10 (`-n` to override).
5. **Emotion tag** each shown word using a **lexicon-first, classifier-fallback**
   approach — not a classifier run on definition text alone. The reasoning:
   the goal is the word's own *connotation* (e.g. "stingy" reads negative,
   "frugal" reads neutral-to-positive, even though both could share a similar
   dictionary definition), not the emotional content of its dictionary gloss —
   a classifier run on the *definition* often can't tell those apart, since it
   only sees denotative text. So:
   - If the word (for WordNet senses, the specific synset) has a **SentiWordNet**
     score, use that for the positive/negative/neutral polarity.
   - If the word is in the **NRC EmoLex** list, use its associated emotions for
     the 7-category-style label (anger, anticipation, disgust, fear, joy,
     sadness, surprise, trust, plus the lexicon's own positive/negative flags).
   - Only for words in neither lexicon (mostly Wiktionary-only entries, e.g.
     slang or very obscure terms) fall back to running
     **j-hartmann/emotion-english-distilroberta-base** (~330MB) on the
     definition text, as a best-effort approximation, clearly still useful but
     acknowledged as weaker signal than the lexicons.
   - The badge shown to the user always displays both an emotion/category label
     and a derived positive/negative/neutral summary, e.g. "Negative (stingy)"
     or "Joy · positive".
6. If the literal input string is itself a real headword in the corpus, its
   full entry (all definitions across all part-of-speech senses, synonyms) is
   pulled separately as an **"exact match"** and pinned first.

**Why bi-encoder + cross-encoder rerank, over the alternatives considered:**
a bi-encoder-only search is simpler and marginally faster, but ranks close
synonyms less precisely. A dedicated research reverse-dictionary model (e.g.
WantWords) is trained specifically for description→word mapping and can be
sharper on descriptive queries, but carries real risks for a tool intended for
long-term use: stale/hard-to-fetch pretrained checkpoints, unclear ongoing
maintenance, and — critically — a **fixed output vocabulary baked in at
training time**, meaning words only present in our Wiktionary supplement could
never surface as candidates. The bi-encoder+cross-encoder hybrid keeps an
open vocabulary (anything in our corpus is retrievable) while still getting
most of the ranking-quality benefit, at a small latency cost (~100-300ms for
the rerank step, on top of a near-instant initial retrieval).

## CLI / interaction

**Interactive picker** (`revdict` with no args, or `revdict <query>` to seed
the first search): candidates are piped into the real **`fzf`** binary
(already installed on the user's system) rather than a custom-built TUI. This
was chosen because the user explicitly wants an fzf-style interface, and
shelling out to real fzf gets fuzzy filtering, arrow-key navigation, and a
live preview pane for free, battle-tested, with far less custom UI code than
reimplementing similar behavior in a Python TUI library.

- The candidate list (headword + one-line gloss + emotion badge + a relative
  relevance indicator) is fed to `fzf` as input lines. The "exact match" entry,
  if the input is a real word, is pinned as the first line and visually
  marked. The relevance indicator is a **rank-based/relative** signal (e.g.
  min-max scaled within the current result set, or a simple bar/star display)
  rather than a calibrated percentage — cross-encoder scores are logits, not
  probabilities, so presenting them as "92%" would overstate precision that
  isn't there. It communicates "this one is closer than that one," not
  "92% confidence."
- `fzf --preview` shows, live as the user arrows through candidates, the full
  detail for the highlighted word: all definitions grouped by part-of-speech
  (WordNet + Wiktionary merged), synonyms/related words, the emotion/connotation
  detail from step 5 above, and an example sentence when available. **Open
  question for review:** the user described wanting to "select one to see
  further details, expand and then collapse" — this spec implements that as
  an always-visible live preview pane (updates automatically as you move the
  selection). `fzf` can alternatively bind a key (e.g. `?`) to toggle the
  preview pane on/off on demand, which may match "expand then collapse" more
  literally. Confirm which behavior is wanted, or whether both (live-by-default,
  toggleable) is fine.
- Typing into fzf fuzzy-filters the candidate list by word or gloss text.
- Pressing Enter on a highlighted candidate prints that word to stdout and
  exits — this also makes the tool pipeable/scriptable, e.g.
  `revdict "feeling of intense annoyance" | wl-copy`.

**One-shot mode**: `revdict <word or phrase> --no-interactive` (or piped/
non-tty stdout) prints the same ranked list + exact match once and exits,
without launching fzf — for scripting contexts where an interactive picker
doesn't make sense.

**`revdict build-index`**: the one-time (or refresh-on-demand) setup command
described in the data pipeline section above.

## No-match / low-confidence handling

Because matching is similarity-based rather than exact, there is always some
ranking to show — there's no hard "invalid word" cutoff like the existing
bash script's API-error case. The top 10 are always shown; if the input is
gibberish or very obscure, the relative relevance indicator (see CLI section)
is simply visibly low for all candidates, which communicates weak confidence
without refusing to answer. The exact cutoff for "visibly low" is an empirical
threshold determined during implementation (by looking at real cross-encoder
scores for known-good vs. known-gibberish queries), not a number fixed in
advance.

## Components

```
revdict/
  data/
    wordnet_source.py      # load WordNet via `wn`, extract sense records
    wiktionary_source.py   # stream-parse + filter kaikki.org English JSONL
    sentiwordnet_source.py  # load SentiWordNet pos/neg/obj scores per synset
    nrc_emolex_source.py     # load NRC EmoLex word -> emotion/sentiment table
    corpus.py               # merge + dedupe WordNet & Wiktionary senses
    build_index.py          # orchestrates corpus build + embedding + save
  models/
    embedder.py              # bge-small-en-v1.5 bi-encoder wrapper
                              # (separate encode_query / encode_passage,
                              # query side gets the instruction prefix)
    reranker.py               # ms-marco-MiniLM-L-6-v2 cross-encoder wrapper
    emotion.py                 # lexicon-first (SentiWordNet + NRC EmoLex)
                                # lookup, classifier fallback
                                # (emotion-english-distilroberta-base) for
                                # words in neither lexicon
  search.py                # embed -> cosine top-75 -> rerank -> top-10,
                             # dedupe by headword, attach emotion tags
  dictionary.py             # exact-match lookup (all senses/POS/synonyms
                             # for a literal headword)
  picker.py                  # builds fzf input lines + preview callback,
                              # shells out to fzf, parses selection
  cli.py                      # argparse entry point: build-index /
                               # one-shot / interactive modes
pyproject.toml
.gitignore                    # excludes ~/.cache/rev_dictionary index
                                # artifacts and local model caches
```

## Error handling

- No index built yet → clear message pointing at `revdict build-index`.
- Empty/invalid input → friendly message, no crash (mirrors the existing bash
  script's input validation).
- `fzf` missing at runtime (e.g. after this was set up, fzf gets uninstalled)
  → falls back to a static `rich`-printed list so the tool still works.
- Model/data download failure during `build-index` (no internet, etc.) →
  clear, specific error naming which model or dataset failed to fetch, safe
  to re-run.
- Low-confidence results → always shown, never refused (see above).

## Testing

- Unit tests for all deterministic glue code, using small fake/injected data
  (no real model loads in tests): corpus merge/dedupe logic, the lexicon-first/
  classifier-fallback emotion lookup logic (including the "word in neither
  lexicon" fallback path), candidate dedup-by-headword and ranking logic,
  fzf input-line formatting and selection parsing.
- ML ranking/embedding quality itself is not meaningfully unit-testable; it is
  validated manually during implementation by running `build-index` once and
  then exercising real queries (a plain word, a near-synonym, a full
  descriptive phrase, an obscure/gibberish phrase) and reviewing output
  quality directly.

## Out of scope (for this spec)

- Phonetic/IPA transcription and stress-mark annotation.
- Replacing the existing bash+notify-send shortcut script — it can continue
  to be used independently for quick popup lookups.
- Non-English languages.
