currently the CLI lists matching words unfiltered in an fzf window. I want to plan and implement the next set of features, based on OneLook's feature set (reference: onelook.com, scans 16,965,772 entries in 805 dictionaries) and my own notes. Below are four feature groups — **filters, categories, advanced filters, and searching capabilities** — please treat these as the required features to implement next. Read through everything, then produce a proper implementation plan (architecture, data/API needs, fzf integration approach, and phased rollout) before writing code.

**1. Searching capabilities** — one-line: support special query syntax so users can search by exact word, prefix/suffix patterns, letter-position wildcards, anagram/unscramble, and meaning-based lookup, not just plain reverse-dictionary queries.

- `bluebird` → get definitions of bluebird
- `blue*` → list words that start with blue
- `*bird` → ...that end with bird
- `bl????rd` → ...that start with bl, end with rd, with 4 letters in between
- `//fuljyo` → ...that have the letters "fuljyo"
- `?????,*y*` → ...that have 5 letters and contain a "y"
- `bl*:snow` → ...that start with bl and have a meaning related to snow
- `:snow` → list words related to snow
- `:winter sport` → ...related to the concept winter sport
- `**winter**` → phrases that contain the word winter
- `expand:nasa` → phrases that spell out n.a.s.a.
- Pattern symbols: `?` any letter, `*` any number of letters, `#` consonant, `@` vowel, `-abcd` disallow letters, `+abcd` restrict to letters, `//abcd//` unscramble, `pattern:meaning`

**2. Advanced filters** — one-line: let users narrow/refine reverse-dictionary results by phonetic, structural, and poetic properties.

- Starts with
- Letters count
- Sounds like
- Primary vowel
- Ends with
- Also related to
- Rhymes with
- Meters (for lyrics/poetry): `/`, `/x`, `x/`, `||`, `/xx`, `x/x` ...
- Syllable count

**3. Categories** — one-line: allow filtering results by part of speech / word class / register.

- All (default)
- Nouns
- Adjectives
- Verbs
- Adverbs
- Idioms/Slang
- Old

Note: fzf alone is limiting for this — we do not currently have support for these categories, so figure out how to properly implement category filtering (likely needs its own data/tagging layer, not just fzf filtering).

**4. Filters (result sorting/ranking)** — one-line: let users sort/rank the result list by similarity, recency, formality, tone, or word length.

- Most Similar
- A → Z / Z → A
- Most Modern / Oldest
- Most Common / Least Common
- Most formal (legal)
- Most funny-sounding
- Most lyrical
- Shortest / Longest
- (Stretch/uncertain) our own emotional buckets — currently inaccurate, more "art" than science, treat as experimental/low priority

Please plan how these four groups map onto our existing architecture, flag any that need external data we may not have, and propose an implementation order. We'll be needing all these features.
