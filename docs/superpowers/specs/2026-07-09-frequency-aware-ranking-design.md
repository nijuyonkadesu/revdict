# Frequency-Aware Candidate Ranking — Design

Date: 2026-07-09
Status: approved

## Problem

Real query investigation ("happy", "sad", "big", "beautiful", "angry")
showed obscure/dialectal words consistently outranking common, well-known
synonyms in the "related words" candidate list. Root-caused against real
raw cross-encoder scores: the pattern isn't really about word rarity per
se — it's that the reranker heavily rewards a candidate's definition
literally containing the query word. Querying "happy": "twinkly-eyed"
(gloss: *"happy, of a happy character"*) scores 7.28; "wealful" (gloss:
*"Happy; joyful; felicitous."*) scores 6.16 — both glosses repeat "happy"
verbatim. Meanwhile "joyful" (gloss: *"Feeling or causing joy"* — no
literal "happy") scores 0.08, and "cheerful" (*"being full of or
promoting cheer..."*) scores **-4.9**. A 7-12 point gap, not noise.

This correlates with rarity because obscure/dialectal Wiktionary entries
are often glossed as bare "= common word" one-liners (lexicographers
documenting an obscure term commonly just gloss it as its more common
synonym), while WordNet's carefully-written definitions for common words
avoid ever defining a word using itself — standard lexicographic practice.

## Design

### Data source

[`wordfreq`](https://github.com/rspeer/wordfreq) — an offline, MIT-licensed
word-frequency package (`zipf_frequency(word, "en")`, a 0-8 log-scale
familiarity score built from real corpora: subtitles, Wikipedia, news,
books, social media). Verified before choosing it:
- No network calls at runtime — data is bundled in the package (~58MB
  installed), consistent with revdict's offline-only architecture.
- Cheap: ~0.12s one-time load, then ~0.4µs per lookup — negligible against
  revdict's existing daemon-warmed model loads.
- Handles every real input shape revdict will pass it without raising:
  multi-word phrases (`"hot under the collar"` → 4.0, treated as a phrase),
  hyphenated words (`"good-humored"` → 2.16), and unknown/gibberish strings
  (→ 0.0). No defensive error handling needed — this is a pure, always-safe
  lookup, not an optional external tool like the stressmark integration.
- Added as a **required** dependency in `pyproject.toml` (not optional —
  this fixes a core ranking-quality issue, unlike stressmark's purely
  additive display feature).

### Blend mechanism and calibration

Immediately after reranking, for every candidate row:

```
blended_score = raw_reranker_score + FREQUENCY_WEIGHT * zipf_frequency(headword, "en")
```

`FREQUENCY_WEIGHT = 1.0`, chosen by testing several values (0.5, 1.0, 1.5,
2.0, 3.0) against the real "happy" candidate set gathered during
investigation:
- At **1.0**: "glad" (common *and* contains "happy" in its own gloss) moves
  to #1 — winning on both signals, as it should. "twinkly-eyed" drops from
  #1 to #2 but isn't eliminated (this is intentionally conservative — the
  user explicitly chose the conservative approach over directly penalizing
  literal query-term overlap). "cheerful" and "joyful" climb several
  positions from being buried near the bottom purely for not repeating the
  query word. Truly obscure words ("wealful", "vogie" — zipf 0.0) get no
  boost and stay ranked on semantic merit alone.
- Confirmed this doesn't reintroduce the "gibberish should read near-zero
  relevance" regression from the earlier daemon work: gibberish-query
  candidates are themselves obscure/unknown words (zipf ≈ 0 in every case
  checked but one, which moved from ~0% to ~3% — not a meaningful
  regression).

The blended score replaces the raw reranker score for everything
downstream in `search()` — deduping by headword, excluding the exact-match
word, truncating to `top_n`, and feeding into the already-tested
`absolute_relevance()` sigmoid for the displayed percentage. This is a
single insertion point, not a parallel scoring path — the displayed
percentage and the candidate order always agree with each other, and nothing
else in the pipeline needs to change or be re-validated.

Exact-match senses are unaffected — they're always pinned regardless of
score, so this only changes candidate ranking/display.

### Error handling

None needed beyond what's already there. `zipf_frequency` never raises for
any input this codebase would pass it (verified above), so the wrapper is
a plain function call, not a guarded one.

### Testing

- Unit tests for the blend formula itself using fake/injected raw scores —
  same style as the existing `absolute_relevance` tests.
- A data-driven regression test using the real "happy" candidates recorded
  during this investigation, asserting the concrete calibration finding:
  "glad" outranks "twinkly-eyed" after blending. This pins the actual
  finding, not just the formula in the abstract.
- Manual validation: re-run the same word list from the investigation
  ("happy", "sad", "big", "beautiful", "angry") against the real rebuilt
  index and confirm the improvement is visible; re-confirm the
  gibberish-query near-zero relevance property still holds.

## Out of scope

- Directly penalizing literal query-term overlap in glosses (the
  "aggressive" alternative — explicitly not chosen).
- Making `FREQUENCY_WEIGHT` runtime-configurable — a fixed, documented
  constant is enough for now; revisit only if 1.0 turns out wrong in
  practice.
- Any change to the exact-match display, or to `emphasis`/stressmark's
  own ranking (out of scope for this repo's ranking pipeline).
