# OneLook Feature Parity — Architecture & Rollout Roadmap

> **This is a roadmap/index document, not a task-execution plan.** It answers
> TODO.md's explicit ask ("architecture, data/API needs, fzf integration
> approach, and phased rollout") and resolves the fzf-vs-TUI question. Each
> phase below gets its own bite-sized implementation plan (per
> `superpowers:writing-plans`' Scope Check: a spec covering multiple
> independent subsystems should be split into separate plans, one per
> subsystem) written and executed in sequence, matching how this project's
> `docs/superpowers/{specs,plans}/` history has always worked — one
> design-then-build cycle at a time, not six at once. Phase 1's full plan is
> ready now: `docs/superpowers/plans/2026-07-19-query-syntax-implementation.md`.

**Goal:** Map TODO.md's four OneLook-inspired feature groups (query syntax,
advanced filters, categories, sort/ranking) onto revdict's existing
architecture, decide how fzf fits going forward, flag real data gaps, and
sequence the work into independently shippable phases.

**Architecture:** Everything in feature groups 1 and most of 2 and 4 is a new
query-parsing/matching/scoring layer that sits in front of the existing
`search.py` pipeline and needs zero UI changes (it rides through as query
text, exactly like today). Feature group 3 (categories) and the phonetic half
of group 2 each need one new, well-precedented data-tagging or cross-repo
extension. Only the *presentation* of categories/filters/sort as a live,
multi-facet panel needs new UI machinery — fzf's single-prompt/single-list
model doesn't fit that, so a new opt-in TUI is introduced for it, while fzf
keeps its current job for everything else. That TUI is written in **Go, not
Python** — it consumes `revdict`'s existing JSON query interface as a
subprocess, exactly the way `revdict.nvim` already does via `--jsonl-query`
(see `../revdict.nvim/lua/revdict/finder.lua`) — but it lives **inside this
same repository** as a self-contained Go module in its own subdirectory
(e.g. `tui/`), not a separate repo: Python (`pyproject.toml`/`src/`) and Go
(`tui/go.mod`) coexist fine side by side in one git tree, each with its own
independent build tooling, and a single `git clone` gets both. This keeps
`revdict` itself a pure query/data layer with no UI-framework dependency of
its own — the Go TUI never imports or links against the Python code, it
only shells out to the built `revdict` binary and parses its JSON output —
while avoiding a second repo to version, tag, and keep in sync.

**Tech Stack:** `revdict` itself: Python 3.11+, numpy, sentence-transformers
(embedder + cross-encoder reranker), rich, fzf (subprocess), NLTK
(WordNet/SentiWordNet), `nrclex`, the sibling `stressmark` package from
`../emphasis`. No new Python dependency is introduced by any phase in this
roadmap, including the TUI phase. Phase 6's TUI is a Go program living in
this repo under `tui/` (tech stack decided in that phase's own plan —
likely `bubbletea`/`lipgloss` from Charm, the standard modern choice for
JSON-driven terminal UIs in Go) that talks to `revdict` purely over stdout,
never linking against it.

## Global Constraints

- No change to `revdict.cli`'s existing flags/behavior for plain queries — every phase must be backward compatible with today's `revdict "query"` and live fzf session.
- No change to the daemon's JSON wire protocol shape (`{"query": ..., "top_n": ...}` request, `{"exact_match": ..., "candidates": [...]}` response) unless a phase explicitly says otherwise — `revdict.nvim`'s `--jsonl-query` and Telescope finder depend on this shape (see `../revdict.nvim/lua/revdict/finder.lua`).
- New pure-Python logic prefers zero new dependencies. `revdict` itself never gains a TUI-framework dependency — the advanced interactive UI (Phase 6) is a separate Go binary/repo consuming `revdict`'s JSON output as a subprocess, not code running inside this codebase.
- Any change to `../emphasis` (stressmark) is additive to its public API (new functions), never a breaking change to `resolve_word_by_pos`/`render_word`, since revdict already depends on today's shape via `src/revdict/models/stress.py`.
- Every new metadata field requires a full `revdict build-index` reindex to populate for existing users; each phase that adds one must say so explicitly (the raw Wiktionary/WordNet/Ngram downloads are cached and skip re-download on rebuild — see `download_raw_wiktextract`'s `if dest.exists(): return` — so a reindex is a reprocessing/re-embedding cost, not a re-download cost).

---

## 1. Current architecture recap

- `src/revdict/search.py:search()` is the single entrypoint: embed query (`Embedder.encode_query`) → cosine top-k retrieval over `embeddings.npy` (787,590 rows today) → cross-encoder rerank (`Reranker.score`) → literary-frequency score nudge (`combine_score`) → dedupe by headword → exclude the exact-match word → emotion-tag + stress-mark each candidate → return `{"exact_match": ..., "candidates": [...]}`.
- `src/revdict/dictionary.py` does the *exact*-match lookup via `word_index.json` (lowercased headword → list of row indices into `metadata.jsonl`).
- `src/revdict/daemon.py` keeps the embedder/reranker/embeddings warm behind a Unix socket so repeat queries don't pay model-load cost; `cli.py` falls back to a cold in-process call if the daemon is unreachable.
- `src/revdict/picker.py` is the fzf integration: a one-shot `run_picker` and a persistent live session `run_live_session` (`fzf --disabled` + `change:reload:revdict --query-only {q}`, debounced 0.1s). The query text is opaque to fzf — it just gets handed to `revdict.cli` on every keystroke.
- `src/revdict/models/stress.py` wraps the sibling `stressmark` package (`../emphasis`) for syllable/stress-marked rendering; `stressmark.engine.resolve_word_by_pos(word, pos)` already computes per-syllable stress and (via `pyphen`/`g2p_en`/`cmudict`) has direct access to phonemes — today only the rendered-to-ANSI-text form is exposed to revdict.
- Metadata record shape (`metadata.jsonl`, one row per sense): `headword, pos, definition, examples, source, sentiwordnet, emolex, synonyms`. `pos` today spans `noun/verb/adjective/adverb` (WordNet) plus Wiktionary's raw POS vocabulary — confirmed by direct sampling of the built index: `adjective, adverb, noun, verb, intj, name, character, num, symbol, prefix, prep, article, phrase, pron, det, particle, suffix, conj, proverb, prep_phrase, contraction, punct, postp`. Wiktionary senses also carry a `tags` field (e.g. `"archaic"`, `"slang"`, `"idiomatic"`, `"formal"`, `"legal"`) that is read today (`wiktionary_source.py:iter_filtered_entries`) only to filter out `form-of`/`alt-of` senses — the rest of `tags` is currently discarded, not stored.
- `literary_frequency.json` (built by `literary_frequency_source.py`) already gives a zipf-scale "how common in 2010-2019 published fiction" score per headword — today used only as an internal nudge inside `combine_score`, never exposed as a standalone sortable field.
- Master headword vocabulary: `word_index.json`'s keys are exactly the deduped, lowercased set of all headwords — **787,590 of them** in the current build (measured directly against the live index). A naive linear regex scan over all 787,590 keys for a structural pattern (e.g. `bl????rd`) measured **~57ms average** (20-run benchmark) — fast enough for interactive use with no new index structure required.
- `revdict.nvim` (sibling repo) drives its live Telescope picker by shelling out to `revdict --jsonl-query "$prompt"` per debounced keystroke (`../revdict.nvim/lua/revdict/finder.lua`) and decoding each JSONL row — it inherits any change to what `revdict --jsonl-query` emits for free, and is unaffected by anything that stays inside the query-text DSL.

## 2. Feature-group → architecture mapping

| TODO.md group | New data needed? | New code | Phase |
|---|---|---|---|
| 1. Searching capabilities (all 11 syntax forms) | None — pure structural matching over existing `word_index` keys | `query_syntax.py`, `pattern_matcher.py`, `structural_search.py`, small `search.py` dispatch | **Phase 1** |
| 4. Sort: A→Z / Z→A / Shortest / Longest | None | `sort.py` + `search()`/CLI param | **Phase 2** |
| 4. Sort: Most/Least Common | None — `literary_frequency.json` already computed, just needs exposing as a raw sort key instead of only an internal score nudge | same as above | **Phase 2** |
| 3. Categories: Nouns/Adjectives/Verbs/Adverbs | None — `pos` is already stored per sense | `category.py` (POS bucket) + `search()`/CLI param | **Phase 3** |
| 3. Categories: Idioms/Slang, Old | **Yes** — capture Wiktionary's discarded `tags` field into a new metadata field at index-build time (reindex required) | `wiktionary_source.py` change, `category.py` (register bucket), reindex | **Phase 3** |
| 2. Advanced filters: Starts with / Ends with / Letters count | None — literally the same primitives as Phase 1 | reuse `pattern_matcher.py` | **Phase 1/3** (exposed as discrete filter params in Phase 3) |
| 2. Advanced filters: Sounds like / Primary vowel / Rhymes with / Meters / Syllable count | None new to *revdict*, but needs `stressmark` (`../emphasis`) to expose phonemes/meter-pattern it already computes internally but doesn't return today | cross-repo: new `stressmark` functions + `models/stress.py` additions + `phonetics.py` | **Phase 4** |
| 2. Advanced filters: Also related to | None — reuses the existing embedding pipeline (combine two query vectors, or union two searches) | small `search.py` addition | **Phase 4** |
| 4. Sort: Most formal (legal) | Reuses Phase 3's new register tags | `sort.py` addition | **Phase 5** |
| 4. Sort: Most Modern / Oldest | Approximated — see data gap below | `sort.py` addition, reuses Phase 2's frequency field + Phase 3's `archaic`/`dated` tags | **Phase 5** |
| 4. Sort: Most lyrical | Approximated — reuses Phase 4's meter/syllable substrate | `sort.py` addition, flagged experimental | **Phase 5** |
| 4. Sort: Most funny-sounding | **No usable data source** | *(not built — see data gap below)* | deferred |
| 4. Sort: emotional buckets | User already flagged this "more art than science" / low priority | *(not built)* | deferred, see `[[revdict-backlog]]` memory |
| Categories/filters/sort as a live interactive panel | — | new Go TUI (`revdict-tui`, separate repo), consuming `revdict`'s JSON output | **Phase 6** |
| revdict.nvim surfacing of category/sort | — | Telescope picker options | **Phase 7** (optional follow-up) |

## 3. Data gaps flagged explicitly (per TODO.md's own request)

1. **Idioms/Slang/Old categories and "most formal" sort** need register/domain tags. We already download the raw data that has this (Wiktionary's per-sense `tags` array — `archaic`, `dated`, `obsolete`, `historical`, `slang`, `idiomatic`, `vulgar`, `colloquial`, `formal`, `legal`, `law`, etc.) but discard everything except the `form-of`/`alt-of` filter today. **No new external dependency — just capture what we already have and store it.** WordNet senses never carry these tags, so these categories will only ever surface Wiktionary-sourced senses — expected, not a bug.
2. **"Most Modern / Oldest" sort has no true etymological-date data source available offline.** `literary_frequency.json` measures *current* (2010-2019) usage frequency, not date-of-first-attestation — a word can be rare-but-ancient ("thou") or rare-but-brand-new ("rizz", if it's even in our corpus) and this signal can't tell them apart on its own. The plan is to approximate: rank by `literary_frequency` for "modern" (frequently used now) and separately surface Wiktionary's `archaic`/`dated`/`obsolete` tags (from data gap #1) as a categorical signal for "oldest," rather than pretending we have real historical dating. This should be labeled as an approximation in the UI, not presented as authoritative.
3. **"Most funny-sounding" sort has no data source at all.** No lexicon or model in this stack encodes phonetic humor. Recommend treating it the same way the user already treats emotional-bucket sorting: experimental/low-priority, not built in the initial phases. If it's ever wanted, the least-arbitrary starting point would be a phoneme-rarity heuristic (unusual consonant clusters, low letter-bigram frequency) — but this has no ground truth to validate against, unlike every other sort mode in this plan.
4. **"Most lyrical" sort** has no ground truth either, but it does have a *defensible mechanical proxy* once Phase 4 lands: stress-pattern regularity and vowel/consonant alternation are actual poetics concepts, and `stressmark` already computes the raw per-syllable stress data needed. Buildable, but should ship labeled as experimental/lower-confidence relative to the "hard data" sorts (alphabetical, length, frequency), not as equally authoritative.

## 4. The fzf-vs-TUI decision

fzf is a fuzzy-filter-over-a-flat-list tool with a single prompt line, single result list, and a preview pane. It has no concept of persistent multi-way tab state, simultaneous multi-facet toggles, or a visible "filters sidebar" — you *can* fake pieces of this with `--header`/`transform-header` and reload commands reading mutable state from temp files, but that's exactly the kind of fragile pile of shell-binding glue the brief's instinct to question fzf is correctly worried about.

The dividing line that falls out of the architecture mapping above is clean:

- **Everything in feature group 1 (query syntax) is just text.** `bl*:snow`, `//fuljyo`, `expand:nasa` are typed into the exact same prompt users already type `"feeling of intense annoyance"` into, whether that's the live fzf session or a one-shot `revdict "..."` call. fzf needs **zero changes** for this — the parsing happens entirely inside `search.py`, invisible to the UI layer. This resolves most of the brief's syntax and advanced-filter asks without touching fzf at all.
- **Single mutually-exclusive states fit fzf's binding model fine** — e.g. a sort-mode cycled by one hotkey (`ctrl-r` rotating through Most Similar → A-Z → Most Common → ...) is a small enum, not a multi-facet panel, and is a very fzf-native pattern (comparable to fzf's own built-in `toggle-sort`).
- **Categories-as-tabs and a simultaneous multi-facet filter panel (starts-with + ends-with + letter-count + sounds-like + rhymes + meter, all live and visibly toggled at once) do not fit fzf's model.** This is where a real TUI earns its keep.

**Decision:** keep fzf exactly as-is for the default/simple path (bare `revdict`, `revdict "query"`, and the live session) — it gets smarter for free as soon as `search()` understands the new query DSL, no picker changes needed. Add a **separate, opt-in TUI written in Go** for the full category-tabs + filter-panel + sort-selector experience, rather than embedding a Python TUI framework (e.g. Textual) inside `revdict` itself — but keep that Go code **in this same repository**, not a new one.

This went through two corrections during planning, both worth keeping on record: an earlier draft proposed a Textual-based TUI living inside `revdict`'s own Python code; that was rejected (Textual is heavy, and `revdict` should stay a UI-framework-free query layer). The next draft moved the TUI to its own separate `revdict-tui` repo; that was also rejected — one more repo to version, tag, and keep in sync isn't worth it when a monorepo subdirectory does the same job. The landed decision: `revdict` is meant to stay a pure query/data layer — the same role it already plays for `revdict.nvim`, which never imports any Python from this project and instead shells out to `revdict --jsonl-query "$prompt"` per keystroke and parses the resulting JSON lines (`../revdict.nvim/lua/revdict/finder.lua`/`entry_maker`) — but the Go TUI's *code* lives inside this repo, e.g. under `tui/` with its own `tui/go.mod`, alongside (not replacing) the existing `pyproject.toml`/`src/` Python layout. A Go TUI in `tui/` consumes `revdict` the exact same way `revdict.nvim` does: invoke `revdict --jsonl-query`-style output (extended, in Phases 2-5, with `--pos`/`--category`/`--sort`/phonetic-filter flags) as a subprocess and render it with a Go TUI toolkit (`bubbletea`/`lipgloss` is the natural pick). This means:

- `revdict` gains **zero** new Python dependencies for the TUI phase — it just needs its CLI to expose the category/sort/filter primitives Phases 2-5 already build as flags, which every consumer (fzf-typed queries, revdict.nvim, the Go TUI) can use.
- One `git clone` of this repo gets both the Python CLI and the Go TUI source — no second repo to track, tag, or keep versioned in lockstep with the first.
- The Go TUI still never imports or links against the Python code — it only shells out to the built `revdict` binary and parses JSON, so the two halves of the repo stay genuinely decoupled even while living in the same git tree (a `go build ./tui/...` doesn't need Python at all, and `uv sync` doesn't need Go at all).

This is Phase 6, deliberately last — pure presentation over backend primitives that Phases 1-5 will have already shipped and tested via CLI flags first, and its `tui/` subdirectory + plan are added once those flags exist to consume.

## 5. Phased rollout

| Phase | Delivers | Depends on | New deps | Plan doc |
|---|---|---|---|---|
| **1** | Full query-syntax DSL (feature group 1, all 11 forms) — exact/prefix/suffix/wildcard/length/contains-letters/exclude-letters/restrict-letters/anagram/acronym-expand/phrase-contains-word/meaning-combined | Nothing | None | `2026-07-19-query-syntax-implementation.md` (ready now) |
| **2** | Zero-new-data sort modes: A→Z, Z→A, Shortest, Longest, Most Common, Least Common, exposed as `search()` param + `--sort` CLI flag | Nothing (reuses existing `literary_frequency.json`) | None | to be written after Phase 1 ships |
| **3** | Register/category tagging foundation (capture Wiktionary `tags`, reindex) + full category filtering (All/Noun/Adjective/Verb/Adverb/Idioms-Slang/Old) as `search()` param + `--pos`/`--category` CLI flags | Nothing new (independent of Phase 1/2, could run in parallel) | None | to be written after Phase 2 ships |
| **4** | `stressmark` API extension (phonemes, meter-pattern string, syllable count as structured data, not just ANSI text) + phonetic advanced filters: sounds-like, primary vowel, rhymes-with, meters, syllable count, also-related-to | Cross-repo coordination with `../emphasis` | None (stdlib + existing `pyphen`/`g2p_en`/`cmudict` already in `stressmark`) | to be written after Phase 3 ships |
| **5** | Remaining sort modes needing Phase 3/4 substrate: most formal (legal), most modern/oldest (approximated), most lyrical (experimental) | Phase 3 (tags) + Phase 4 (phonetics) | None | to be written after Phase 4 ships |
| **6** | Go TUI under `tui/` in *this* repo: category tabs, multi-facet filter panel, sort selector, consuming `revdict`'s JSON query interface as a subprocess — same pattern as `revdict.nvim`, but same-repo, not a new one | Phases 1-5 (consumes their CLI flags/JSON output) | Go, `bubbletea`/`lipgloss` (scoped to `tui/go.mod` — `revdict`'s Python side gains nothing) | to be written after Phase 5 ships |
| **7** (optional) | `revdict.nvim` follow-up surfacing category/sort as Telescope picker options | Phase 6 | None | not required to consider the roadmap complete |
| **8** | Multi-repo bootstrapping/distribution (see section 7 below) — not a feature phase, a packaging/ops initiative that can happen in parallel with any of the above | None (independent) | TBD, see section 7 | not yet planned — captured as a backlog item, see `revdict_backlog` project memory |

Each phase after Phase 1 gets its own `superpowers:writing-plans` pass once the phase before it has shipped, mirroring this project's existing `docs/superpowers/{specs,plans}/` rhythm (six prior features were each speced and planned individually, not batched). Phase 1 is intentionally first because it's fully standalone (no data gaps, no reindex, no cross-repo coordination) and is the single largest chunk of what TODO.md asked for.

## 7. Multi-repo bootstrapping / distribution (flagged, not yet solved)

This family spans three repos — `revdict` (this repo, which now also holds
the planned Phase 6 Go TUI under `tui/` — deliberately kept in-repo rather
than becoming a fourth repo, precisely to avoid adding to this problem),
`revdict.nvim`, and `emphasis`/`stressmark` — with real runtime dependencies
between them that no package manager or plugin manager currently resolves
automatically. Concretely, as of this roadmap:

- All three existing repos **do** have real GitHub remotes today
  (`github.com/nijuyonkadesu/revdict`, `.../revdict.nvim`, `.../emphasis`) —
  this was confirmed directly against each repo's `git remote -v`, and
  supersedes an earlier point-in-time note that `revdict.nvim` had no
  remote. So "clone from GitHub" is at least possible for each repo
  individually; the gap is entirely in what happens *between* them.
- `revdict.nvim` → `revdict`: purely a README instruction ("have `revdict`
  installed and on PATH, with its index built — see that project's own
  README"), not automated. Its own `README.md`'s lazy.nvim install example
  still shows `dir = "~/redacted/revdict.nvim"` (a hardcoded local path),
  not a real GitHub source spec — even the documented example doesn't
  demonstrate installing from GitHub.
- `revdict` → `stressmark`: **not a declared dependency at all.** Verified
  directly: `grep -rn "stressmark" pyproject.toml uv.lock` in this repo
  returns nothing. The only place this dependency exists on disk is a
  `.venv/lib/.../__editable__.stressmark-0.1.0.pth` file pointing at
  `/home/shichika/redacted/emphasis/src` — created by a one-off manual `uv
  pip install -e /path/to/emphasis`, exactly as `revdict`'s own README
  documents it (`README.md`, "Optional: stress-marked pronunciation"
  section). This is intentional today (stressmark is meant to be optional,
  wrapped in `try/except ImportError` in `models/stress.py`) but it means
  the dependency is invisible to `uv sync`, `uv.lock`, and anyone who
  doesn't happen to read that specific README section.
- The Phase 6 Go TUI does **not** add a new instance of this problem: since
  it lives under `tui/` in this same repo, cloning `revdict` already gets
  it — no separate install/discovery step, no separate remote to find.
- `revdict`'s own index build (`revdict build-index`) is a genuine ~30
  minute, network-dependent, multi-GB operation (WordNet, a Wiktionary
  extract, Google Books Ngram data, embedding the full corpus) — no
  packaging fix eliminates this step, it's inherent to what the tool does.
  Bootstrapping work below is about the *code/dependency* graph, not this.

**Net effect today:** someone arriving at `revdict.nvim`'s GitHub page with
zero prior context cannot get to a fully working `:Revdict` (with stress
markup) by following only what's in front of them — they need to discover,
in order, that they also need `revdict` (a second repo, separately
installed and index-built) and, if they want stress markup, `emphasis` (a
third repo, manually `pip install -e`'d into `revdict`'s own venv, not
mentioned anywhere in `revdict.nvim`'s own README at all).

**Options worth evaluating when this is picked up (not decided here):**

1. **Turn the `stressmark` dependency into a real declared one.** Now that
   `emphasis` has a public GitHub remote, `revdict`'s `pyproject.toml` could
   depend on it via a git URL (`stressmark @
   git+https://github.com/nijuyonkadesu/emphasis.git`), either as a normal
   dependency or under `[project.optional-dependencies]` so `uv sync
   --extra stress` (or similar) becomes the one-line equivalent of today's
   manual `uv pip install -e`. Lowest effort of everything on this list, and
   wasn't possible before today (the manual-install README section predates
   `emphasis` having a public remote).
2. **A `build` hook in `revdict.nvim`'s plugin spec** (lazy.nvim supports
   `build = function() ... end`) that at least verifies `revdict` is on
   `PATH` and its index exists, printing actionable next steps if not,
   rather than silently failing on first `:Revdict`. A full auto-install
   (clone + `uv sync` + `build-index`) is possible but riskier to do
   silently given the 30-minute network-dependent build step — at minimum
   this should ask before running it, not block Neovim startup on it.
3. **A small top-level bootstrap script (or a lightweight fifth "meta"
   repo)** that clones the needed sibling repos and runs the right install
   commands in the right order — the closest to "one command, everything
   works," at the cost of one more thing to maintain and keep in sync as
   each project evolves independently.
4. **Free, immediate documentation fixes**, regardless of which of the
   above (if any) gets built: (a) fix `revdict.nvim`'s README lazy.nvim
   example to show a real GitHub source spec instead of a local `dir=`
   path; (b) add a one-paragraph "dependency graph" note to each of the
   three READMEs pointing at the other two, so the full chain is
   discoverable starting from any single repo, not just by tribal
   knowledge of how the author's own machine happens to be laid out.

This is captured as a `revdict_backlog` project memory so it surfaces in
future sessions across any of these repos, not just this one.

## 8. Interpretation calls made (ambiguous TODO.md syntax, resolved for the record)

TODO.md's examples were used to resolve a few points its own symbol legend left ambiguous, so future readers don't re-litigate these:

- **`//fuljyo` (no trailing `//`) vs. the legend's `//abcd//`:** confirmed as the *same* anagram/unscramble syntax with an optional closing `//` — verified against the example itself: `fuljyo` sorted is `fjlouy`, which is exactly `joyful` sorted. Phase 1's parser accepts both `//abcd` and `//abcd//` (strips `/` from both ends before treating the remainder as the letter set).
- **`?????,*y*` comma syntax:** comma is a top-level clause separator, ANDing multiple independent pattern clauses together (5-letter-count AND contains-y) — generalized as "split on comma, AND every clause" rather than special-cased to just length+contains.
- **`**winter**` vs. a plain `*word*` substring-contains:** these are different operations. `*word*` (falls out naturally from the general wildcard compiler) means "contains this letter sequence anywhere, including across word boundaries." `**word**` is explicitly phrase-level: the headword must be multi-word, and `word` must match one of its whole tokens exactly — not a substring match.
- **`?`/`#`/`@` colliding with natural-language punctuation in meaning queries:** a meaning query that happens to contain `*`, `?`, `#`, `@`, or a leading `+`/`-` will be misparsed as pattern syntax (e.g. `"a word for asking a question?"`). This is an accepted trade-off (OneLook's real syntax has the identical collision) — the `:meaning` explicit prefix is the documented escape hatch, not a gap to be closed with fragile natural-language-vs-pattern heuristics.
- **`y` for `#`/`@` (consonant/vowel wildcards):** treated as a consonant (not a vowel), since English orthography favors that reading (yellow, yes) and TODO.md's legend doesn't specify either way. Documented as an interpretation call in the code, not silently assumed.
- **A colon inside an otherwise-plain meaning query forces pattern mode (the inverse of the `?`/`*` collision above):** typing a natural-language query that happens to contain a colon (e.g. `"note: to make a written record"`) is parsed as combined mode with pattern clause `["note"]` ANDed against meaning text `"to make a written record"` — silently narrowing results to the literal headword "note" rather than running a plain meaning search over the whole phrase. This is the accepted cost of using `:` as the pattern/meaning separator (TODO.md's own syntax, not a code defect) — same "wrap the whole thing to force meaning mode" reasoning does not actually apply here since the colon itself is what triggers combined mode; there is currently no escape hatch for this specific case. Worth revisiting if it turns out to bite real usage, but not a blocker for Phase 1.
- **Acronym expansion (`expand:`) skips small function words** (`and, of, the, for, a, an, &`) when computing initials, rather than taking every token literally — confirmed necessary by hand-tracing the TODO.md-adjacent example "national aeronautics and space administration": naively including "and" produces `naasa`, not `nasa`.
