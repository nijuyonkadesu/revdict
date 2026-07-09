# Score-Discount Ranking Fix — Design

Date: 2026-07-09
Status: approved (revised from an earlier pure-frequency-blend draft — see
"Rejected approach" below)

## Intent

revdict's stated use case: finding precise, evocative words for creative
writing — short stories, expressing a thought "with grand or comical
effect." The words that come back need to be words a novel, webnovel, or
translated manga would actually use — not words "buried during the
Shakespeare era," and not a word that's technically a correct synonym but
that no working writer would reach for.

## Problem

Real query investigation ("happy", "sad", "big", "beautiful", "angry")
showed obscure/dialectal words consistently outranking common, well-known
synonyms in the "related words" candidate list. Root-caused against real
raw cross-encoder scores: the reranker heavily rewards a candidate's
definition literally containing the query word. Querying "happy":
"twinkly-eyed" (gloss: *"happy, of a happy character"* — "happy" appears
twice) scores **7.28**; "wealful" (gloss: *"Happy; joyful; felicitous."*)
scores **6.16**. Meanwhile "joyful" (gloss: *"Feeling or causing joy"* — no
literal "happy") scores **0.08**, and "cheerful" (*"being full of or
promoting cheer..."*) scores **-4.9**. A 7-12 point gap, not noise.

## Rejected approach: blending in general word frequency

The first draft of this spec proposed blending in `wordfreq`'s general
English familiarity score as a positive boost. Cross-verified against the
actual use case before implementing anything, and rejected: `wordfreq`
averages together Wikipedia, subtitles, news, Google Books, web text,
Twitter, and Reddit into one number, which does **not** track "natural in
fiction." Real counter-example found during verification:

| Genuinely archaic | zipf | Fiction-staple verbs | zipf |
|---|---|---|---|
| wherefore | 2.79 | **murmured** | **2.54** |
| yonder | 2.97 | **scowled** | **2.02** |
| anon | 3.33 | **smirked** | **2.59** |

"Murmured" and "scowled" — words that appear on nearly every page of a
novel — score *lower* than Shakespeare-era archaisms, because they're rare
in tweets/news/casual subtitles despite being constant in narrative prose.
A general frequency blend would have failed at exactly the thing it was
supposed to fix.

## Design: two independent, strictly-subtractive discounts

Rather than trying to predict "is this word good for fiction" (unreliable,
per above), this targets only the two things that are reliably, narrowly
measurable: (1) a candidate whose score is inflated because its own
definition just restates the query word, and (2) a candidate that's
genuinely unattested in real-world text at all. **Both adjustments only
ever subtract from a score, never add.** This is a load-bearing property,
not an incidental one: it means the fix cannot, even in principle, inflate
any candidate's score — the existing "a low-confidence/gibberish query
reads as visibly low across the board" guarantee (`absolute_relevance`,
validated during the daemon work) stays intact by construction, not
because we re-tested every case.

Immediately after reranking, for every candidate row:

```
adjusted_score = raw_reranker_score + zero_frequency_penalty + overlap_discount
```

**1. Zero-frequency penalty** (`ZERO_FREQUENCY_PENALTY = -3.0`, applied when
`wordfreq.zipf_frequency(headword, "en") < 0.5`): catches words with
essentially no attestation in ANY of wordfreq's seven source corpora
combined (a word scoring exactly 0.0 there is a strong, unambiguous "this
is unused" signal — unlike using the raw zipf value as a general-purpose
familiarity ranking, which the rejected approach showed is unreliable, a
hard `< 0.5` cutoff only fires for words that are close to actually
unattested). Real examples: "wealful" and "vogie" both score 0.0.

**2. Overlap discount** (`OVERLAP_DISCOUNT_PER_OCCURRENCE = -3.5`, applied
once per literal occurrence of the query word in the candidate's own
definition, case-insensitive, whole-word match): directly targets the
actual inflation mechanism found in the investigation. "Twinkly-eyed"'s
gloss contains "happy" twice → `-7.0`; "glad"'s gloss contains "happy" once
→ `-3.5`. This is genre-agnostic by design — it doesn't touch a candidate's
score at all unless its own gloss happens to restate the query term, so it
never penalizes "murmured"-style words that are simply uncommon in casual
speech.

**Calibration**, verified against the real "happy" candidate set gathered
during investigation:

| Before (raw) | After (adjusted) | Word |
|---|---|---|
| 1st: twinkly-eyed (7.28) | 1st: happies (3.14) | |
| 2nd: vogie (7.00) | 2nd: **good-humored** (2.97) | |
| 3rd: happies (6.64) | 3rd: **glad** (1.92) | |
| 4th: good-humored (6.47) | 4th: vogie (0.50) | |
| 5th: wealful (6.16) | 5th: twinkly-eyed (0.28) | dropped from 1st |

"Twinkly-eyed" drops from 1st to 5th. "Good-humored" and "glad" — both
genuinely common, natural-sounding words — move into the top 3. "Wealful"
and "vogie" (zero real-world attestation) drop toward the bottom. This is
the concrete behavior this feature commits to; the implementation plan's
tests pin these exact before/after numbers, not just the formula in the
abstract.

### What this does *not* claim to do

This does not boost genuinely good synonyms that happen to be phrased
without repeating the query word (e.g. "cheerful"'s best WordNet sense,
*"being full of or promoting cheer..."*, stays near the bottom of its own
candidate set) — fixing that would require a positive signal for "sounds
natural in fiction," and the investigation found no reliable offline data
source for that. This fix removes a *known, measurable* distortion; it
does not add general literary-register judgment.

### Data source

`wordfreq` (`zipf_frequency(word, "en")`) is still used, but narrowly — only
as a `< 0.5` cutoff for the zero-frequency penalty, not as a general
ranking signal. Offline, no network calls at runtime, ~58MB installed,
~0.12s one-time load, ~0.4µs per lookup thereafter. Verified to handle
every real input shape (multi-word phrases, hyphenated words, unknown
strings) without raising. Added as a required `pyproject.toml` dependency.

### Error handling

Neither `zipf_frequency` nor the overlap regex match can raise for any
input this codebase would pass them (verified: multi-word phrases,
hyphens, empty strings, gibberish all handled cleanly). No defensive
wrapping needed beyond what a plain function call already provides.

### Testing

- Unit tests for both discount components independently (zero-frequency
  cutoff, overlap-occurrence counting) using fake/injected data.
- A data-driven regression test using the real "happy" candidates recorded
  above, pinning the exact before/after ranking change this feature
  commits to — not just testing the formula in the abstract.
- A test proving the "never inflates a score" invariant directly: for any
  input, `adjusted_score <= raw_reranker_score`.
- Manual validation: re-run the investigation's word list ("happy", "sad",
  "big", "beautiful", "angry") against the real rebuilt-index behavior and
  confirm the concrete improvement; re-confirm the gibberish-query
  near-zero relevance property still holds.

## Out of scope

- Any positive "sounds natural in fiction" boost — no reliable offline
  signal was found for this; explicitly not attempted.
- Making the discount constants runtime-configurable — fixed, documented
  constants for now.
- Any change to the exact-match display, or to `emphasis`/stressmark's own
  ranking.
