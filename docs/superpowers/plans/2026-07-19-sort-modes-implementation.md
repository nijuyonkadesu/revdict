# Sort Modes (OneLook Feature Group 4, zero-new-data subset) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add result sort/ranking modes (A→Z, Z→A, Shortest, Longest, Most Common, Least Common, plus the existing default relevance ordering) to `revdict`, exposed as a `search()` parameter, a daemon wire-protocol field, and a `--sort` CLI flag — reusing the already-computed `literary_frequency.json`, with zero new data pipeline and zero reindex required.

**Architecture:** A new pure module `src/revdict/sort.py` defines the seven valid sort-mode names and a single `apply_sort()` function that re-orders an already-built candidate list. `search()` calls it once at each of its two return points (the early structural/expand/phrase_contains return, and the final meaning/combined return), so every query mode gets sorting for free without any mode-specific logic. The daemon's JSON request gains one new optional `"sort"` field (`.get("sort")` on the server side keeps old clients working unchanged). The CLI gains a `--sort` flag threaded through to the same place. This is Phase 2 of the OneLook-feature-parity roadmap (`docs/superpowers/plans/2026-07-19-onelook-feature-parity-roadmap.md`) — the "zero-new-data" subset of TODO.md's sort/ranking feature group; "Most formal," "Most modern/oldest," and "Most lyrical" need data or tagging from later phases and are explicitly out of scope here.

**Tech Stack:** Pure Python stdlib (`sorted()`, no new dependencies).

## Global Constraints

- The default sort behavior (no `--sort` flag, `sort_mode=None`, or `sort_mode="relevance"`) must leave every existing candidate ordering **completely unchanged** — this is TODO.md's "Most Similar," already what `search()` produces today via reranking (meaning mode) or frequency-descending (structural/expand/phrase_contains modes). No task in this plan may alter that default path's output.
- `literary_frequency` lookups are keyed by **lowercased** headword (matches `combine_score`'s and `structural_search._score_and_sort`'s existing convention) — a candidate whose headword has no entry is treated as frequency `0.0`, consistent with how missing single-token entries are already treated elsewhere in this codebase.
- All string comparisons (`alpha`/`alpha_desc`) are case-insensitive; all sort modes that could otherwise produce ties (`shortest`/`longest`/`most_common`/`least_common`) break ties alphabetically for deterministic, reproducible output.
- The daemon wire-protocol change must be backward compatible: a request JSON without a `"sort"` key must behave exactly as it did before this plan (defaults to `None`/relevance), and `_handle_request`'s `search_fn` contract changes from 2 args (`query`, `top_n`) to 3 (`query`, `top_n`, `sort_mode`) — every existing test fake matching the old 2-arg shape must be updated, not left to silently break.
- No change to `revdict.picker`, the live fzf session's binding logic, or `revdict --query-only`/`--jsonl-query` (the live session and revdict.nvim's picker) — sort-mode selection inside the live/interactive UI is explicitly out of scope for this phase (flagged in the roadmap as a future fzf-hotkey enhancement, not required here). Those code paths keep defaulting to `sort_mode=None` implicitly.

---

### Task 1: `sort.py` — sort mode registry and `apply_sort()`

**Files:**
- Create: `src/revdict/sort.py`
- Test: `tests/test_sort.py`

**Interfaces:**
- Produces: `SORT_MODES: tuple[str, ...]` (the 7 valid mode names, in the order they should be presented in CLI help/README), `apply_sort(candidates: list[dict], sort_mode: str | None, literary_frequency: dict[str, float]) -> list[dict]`. Every candidate dict in the input list is guaranteed to have a `"headword"` key (the shape `build_candidate()` in `search.py` already produces, per the plan `2026-07-19-query-syntax-implementation.md`'s Global Constraints). Raises `ValueError` for any `sort_mode` string not in `SORT_MODES` (and not `None`).
- Consumes: nothing from other new modules — this task has no dependencies.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sort.py
import pytest

from revdict.sort import SORT_MODES, apply_sort


def _candidates(*headwords):
    return [{"headword": hw} for hw in headwords]


def test_sort_modes_contains_exactly_the_seven_documented_modes():
    assert SORT_MODES == (
        "relevance",
        "alpha",
        "alpha_desc",
        "shortest",
        "longest",
        "most_common",
        "least_common",
    )


def test_none_sort_mode_returns_candidates_in_their_original_order():
    candidates = _candidates("zebra", "apple", "mango")

    assert apply_sort(candidates, None, {}) == candidates


def test_relevance_sort_mode_returns_candidates_in_their_original_order():
    candidates = _candidates("zebra", "apple", "mango")

    assert apply_sort(candidates, "relevance", {}) == candidates


def test_alpha_sorts_case_insensitively_ascending():
    candidates = _candidates("Zebra", "apple", "Mango")

    result = apply_sort(candidates, "alpha", {})

    assert [c["headword"] for c in result] == ["apple", "Mango", "Zebra"]


def test_alpha_desc_sorts_case_insensitively_descending():
    candidates = _candidates("apple", "Zebra", "mango")

    result = apply_sort(candidates, "alpha_desc", {})

    assert [c["headword"] for c in result] == ["Zebra", "mango", "apple"]


def test_shortest_sorts_by_length_ascending_with_alphabetical_tiebreak():
    candidates = _candidates("bb", "aaa", "z", "aa")

    result = apply_sort(candidates, "shortest", {})

    assert [c["headword"] for c in result] == ["z", "aa", "bb", "aaa"]


def test_longest_sorts_by_length_descending_with_alphabetical_tiebreak():
    candidates = _candidates("bb", "aaa", "z", "aa")

    result = apply_sort(candidates, "longest", {})

    assert [c["headword"] for c in result] == ["aaa", "aa", "bb", "z"]


def test_most_common_sorts_by_literary_frequency_descending():
    candidates = _candidates("rare", "common", "medium")
    literary_frequency = {"common": 5.0, "medium": 2.0, "rare": 0.1}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert [c["headword"] for c in result] == ["common", "medium", "rare"]


def test_least_common_sorts_by_literary_frequency_ascending():
    candidates = _candidates("rare", "common", "medium")
    literary_frequency = {"common": 5.0, "medium": 2.0, "rare": 0.1}

    result = apply_sort(candidates, "least_common", literary_frequency)

    assert [c["headword"] for c in result] == ["rare", "medium", "common"]


def test_most_common_treats_a_missing_frequency_entry_as_zero():
    candidates = _candidates("known", "unknown")
    literary_frequency = {"known": 3.0}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert [c["headword"] for c in result] == ["known", "unknown"]


def test_frequency_lookup_is_case_insensitive():
    candidates = _candidates("Common")
    literary_frequency = {"common": 5.0}

    result = apply_sort(candidates, "most_common", literary_frequency)

    assert result == candidates


def test_unknown_sort_mode_raises_value_error():
    with pytest.raises(ValueError, match="nonsense"):
        apply_sort(_candidates("a"), "nonsense", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sort.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revdict.sort'`

- [ ] **Step 3: Write the implementation**

```python
# src/revdict/sort.py
SORT_MODES = (
    "relevance",
    "alpha",
    "alpha_desc",
    "shortest",
    "longest",
    "most_common",
    "least_common",
)


def apply_sort(
    candidates: list[dict], sort_mode: str | None, literary_frequency: dict[str, float]
) -> list[dict]:
    if not sort_mode or sort_mode == "relevance":
        return candidates
    if sort_mode == "alpha":
        return sorted(candidates, key=lambda c: c["headword"].lower())
    if sort_mode == "alpha_desc":
        return sorted(candidates, key=lambda c: c["headword"].lower(), reverse=True)
    if sort_mode == "shortest":
        return sorted(candidates, key=lambda c: (len(c["headword"]), c["headword"].lower()))
    if sort_mode == "longest":
        return sorted(candidates, key=lambda c: (-len(c["headword"]), c["headword"].lower()))
    if sort_mode == "most_common":
        return sorted(
            candidates,
            key=lambda c: (
                -literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    if sort_mode == "least_common":
        return sorted(
            candidates,
            key=lambda c: (
                literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    raise ValueError(f"Unknown sort mode: {sort_mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sort.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add src/revdict/sort.py tests/test_sort.py
git commit -m "Add sort mode registry and apply_sort() for candidate re-ordering"
```

---

### Task 2: Wire `sort_mode` into `search()`

**Files:**
- Modify: `src/revdict/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `sort.SORT_MODES`, `sort.apply_sort` (Task 1).
- Produces: `search(query: str, top_n: int = 10, sort_mode: str | None = None) -> dict` — the public signature gains one new optional parameter; existing callers that don't pass `sort_mode` are completely unaffected (default `None` reproduces today's behavior exactly).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search.py (append)


def test_search_sort_mode_defaults_to_none_and_preserves_relevance_order(monkeypatch):
    """Backward compatibility: calling search() exactly as before (no
    sort_mode argument at all) must produce the same order as today --
    proven here by NOT passing sort_mode and confirming the plain
    meaning-mode candidate order matches what the reranker/combine_score
    pipeline alone would produce (unsorted by anything sort.py adds)."""
    metadata = [
        {
            "headword": "aardvark", "pos": "noun", "definition": "def a",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "zebra", "pos": "noun", "definition": "def z",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"aardvark": [0], "zebra": [1]},
        "literary_frequency": {},
        "classifier": None,
    }

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            # "zebra"'s definition scores higher than "aardvark"'s, so the
            # un-sorted relevance order is [zebra, aardvark] -- the REVERSE
            # of alphabetical, so this test can't accidentally pass just
            # because alpha-sort happens to match the default order.
            return [1.0 if "def z" in d else 0.5 for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("some query")

    assert [c["headword"] for c in result["candidates"]] == ["zebra", "aardvark"]


def test_search_alpha_sort_mode_reorders_meaning_mode_candidates(monkeypatch):
    """Same fixture as the default-order test above, but with
    sort_mode="alpha" -- must flip the order to alphabetical, proving
    sort_mode is actually threaded through the meaning-mode return path."""
    metadata = [
        {
            "headword": "aardvark", "pos": "noun", "definition": "def a",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "zebra", "pos": "noun", "definition": "def z",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"aardvark": [0], "zebra": [1]},
        "literary_frequency": {},
        "classifier": None,
    }

    class FakeEmbedder:
        def encode_query(self, query):
            return np.array([1.0, 0.0], dtype="float32")

    class FakeReranker:
        def score(self, query, definitions):
            return [1.0 if "def z" in d else 0.5 for d in definitions]

    state["embedder"] = FakeEmbedder()
    state["reranker"] = FakeReranker()
    state["embeddings"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    state["embedding_norms"] = np.array([1.0, 1.0])
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    result = search_mod.search("some query", sort_mode="alpha")

    assert [c["headword"] for c in result["candidates"]] == ["aardvark", "zebra"]


def test_search_sort_mode_applies_to_structural_mode_results_too(monkeypatch):
    """Structural-mode results default to frequency-descending order
    (structural_search._score_and_sort) -- sort_mode="alpha" must override
    that default too, proving the dispatch branch's sort is wired, not
    just the meaning-mode branch's."""
    metadata = [
        {
            "headword": "bluebird", "pos": "noun", "definition": "a songbird",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
        {
            "headword": "blueprint", "pos": "noun", "definition": "a drawing",
            "examples": [], "source": "wordnet", "sentiwordnet": None,
            "emolex": ["joy"], "synonyms": None,
        },
    ]
    state = {
        "metadata": metadata,
        "word_index": {"bluebird": [0], "blueprint": [1]},
        "literary_frequency": {"bluebird": 1.0, "blueprint": 3.0},
        "classifier": None,
    }
    monkeypatch.setattr(search_mod, "_load_state", lambda: state)

    default_result = search_mod.search("blue*")
    alpha_result = search_mod.search("blue*", sort_mode="alpha")

    # Default: frequency-descending (blueprint=3.0 > bluebird=1.0).
    assert [c["headword"] for c in default_result["candidates"]] == ["blueprint", "bluebird"]
    # sort_mode="alpha" overrides that to alphabetical.
    assert [c["headword"] for c in alpha_result["candidates"]] == ["bluebird", "blueprint"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_search.py -v -k "sort_mode"`
Expected: FAIL — `search()` currently has no `sort_mode` parameter, so calls passing it raise `TypeError: search() got an unexpected keyword argument 'sort_mode'`.

- [ ] **Step 3: Wire `sort_mode` into `search()`**

Add the import at the top of `src/revdict/search.py` (alongside the existing `revdict` imports):

```python
from revdict import sort
```

Change `search()`'s signature (currently `def search(query: str, top_n: int = 10) -> dict:`) to:

```python
def search(query: str, top_n: int = 10, sort_mode: str | None = None) -> dict:
```

Change the early structural/expand/phrase_contains dispatch block (currently):

```python
    parsed = query_syntax.parse_query(query)
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        return structural_search.run_structural(parsed, state, top_n)
```

to:

```python
    parsed = query_syntax.parse_query(query)
    if parsed.mode in ("structural", "expand", "phrase_contains"):
        result = structural_search.run_structural(parsed, state, top_n)
        result["candidates"] = sort.apply_sort(
            result["candidates"], sort_mode, state["literary_frequency"]
        )
        return result
```

Change the final candidate-building/return section (currently):

```python
    candidates = [
        build_candidate(metadata[row_index], relevance, state)
        for (row_index, _), relevance in zip(deduped, relevances)
    ]

    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: get_classifier(state)
    )
    return {"exact_match": exact_match, "candidates": candidates}
```

to:

```python
    candidates = [
        build_candidate(metadata[row_index], relevance, state)
        for (row_index, _), relevance in zip(deduped, relevances)
    ]
    candidates = sort.apply_sort(candidates, sort_mode, literary_frequency)

    exact_match = tag_exact_match_senses(
        exact_match_raw, classifier_factory=lambda: get_classifier(state)
    )
    return {"exact_match": exact_match, "candidates": candidates}
```

No other line in `search()` changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: PASS (all existing tests plus the 3 new ones)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — every pre-existing test file still passes unmodified (nothing else calls `search()` with a fixed positional-arg-count assumption that `sort_mode` would break, since it's a new keyword-only-by-convention parameter with a default).

- [ ] **Step 6: Commit**

```bash
git add src/revdict/search.py tests/test_search.py
git commit -m "Wire sort_mode through search() for both meaning and structural modes"
```

---

### Task 3: Wire `sort_mode` through the daemon wire protocol

**Files:**
- Modify: `src/revdict/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: nothing from Task 1/2 directly (the daemon module doesn't import `sort.py` — it just forwards an opaque `sort_mode` string through to whatever `search_fn` it's given, per its existing pattern of never importing `revdict.search` at module scope).
- Produces: `send_query(query: str, top_n: int, sort_mode: str | None = None, timeout: float = 30.0) -> dict | None` (new `sort_mode` parameter inserted before the existing `timeout` parameter — every existing call site uses `timeout=` as a keyword, so this insertion doesn't break them). `_handle_request(request_text: str, search_fn) -> str` now calls `search_fn(request["query"], top_n=request["top_n"], sort_mode=request.get("sort"))` — the `search_fn` contract changes from 2 required args to 3.

- [ ] **Step 1: Update the two existing `_handle_request` test fakes to accept `sort_mode`**

In `tests/test_daemon.py`, the `_handle_request` contract is changing from `search_fn(query, top_n)` to `search_fn(query, top_n=..., sort_mode=...)`. Update these two existing tests' fake functions (their bodies and assertions stay otherwise the same — only the fake's signature and the `calls` dict gain the new field):

Replace:

```python
def test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result():
    calls = {}

    def fake_search(query, top_n):
        calls["query"] = query
        calls["top_n"] = top_n
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, fake_search)

    assert calls == {"query": "happy", "top_n": 10}
    assert json.loads(response_text) == {"exact_match": None, "candidates": []}
```

with:

```python
def test_handle_request_calls_search_fn_with_parsed_args_and_returns_json_result():
    calls = {}

    def fake_search(query, top_n, sort_mode):
        calls["query"] = query
        calls["top_n"] = top_n
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, fake_search)

    assert calls == {"query": "happy", "top_n": 10, "sort_mode": None}
    assert json.loads(response_text) == {"exact_match": None, "candidates": []}
```

Replace:

```python
def test_handle_request_returns_error_payload_when_search_fn_raises():
    def failing_search(query, top_n):
        raise RuntimeError("index not loaded")

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, failing_search)

    payload = json.loads(response_text)
    assert "index not loaded" in payload["error"]
```

with:

```python
def test_handle_request_returns_error_payload_when_search_fn_raises():
    def failing_search(query, top_n, sort_mode):
        raise RuntimeError("index not loaded")

    request_text = json.dumps({"query": "happy", "top_n": 10})

    response_text = daemon._handle_request(request_text, failing_search)

    payload = json.loads(response_text)
    assert "index not loaded" in payload["error"]
```

The third existing test, `test_handle_request_returns_error_payload_on_malformed_json`, is unaffected — its `lambda query, top_n: {}` is never actually called (the malformed JSON fails at `json.loads` before `search_fn` is invoked), so leave it exactly as-is.

- [ ] **Step 2: Run the updated tests to verify they still fail for the expected reason**

Run: `.venv/bin/pytest tests/test_daemon.py -v -k handle_request`
Expected: FAIL — the two updated tests now assert `sort_mode` fields that `_handle_request` doesn't produce yet (it still calls `search_fn(request["query"], top_n=request["top_n"])`, so the 3-arg fakes above would actually raise `TypeError: fake_search() missing 1 required positional argument: 'sort_mode'` at the point `_handle_request` invokes them).

- [ ] **Step 3: Write the new failing tests for `send_query`'s sort-mode wiring**

Add a shared capturing-server helper (reusing the existing `_run_echo_server` pattern already in this file) and two new tests:

```python
# tests/test_daemon.py (append)


def _run_capturing_server(socket_path, received, ready_event):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    ready_event.set()
    conn, _ = server.accept()
    with conn:
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        received["request"] = json.loads(b"".join(chunks).decode("utf-8"))
        conn.sendall(json.dumps({"exact_match": None, "candidates": []}).encode("utf-8"))
    server.close()


def test_send_query_includes_sort_mode_in_the_request_payload(tmp_path, monkeypatch):
    """The wire-protocol extension: a non-default sort_mode must actually
    reach the server in the request JSON, not get silently dropped."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, sort_mode="alpha", timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": "alpha"}


def test_send_query_defaults_sort_mode_to_none_when_omitted(tmp_path, monkeypatch):
    """Backward compatibility for the CLIENT side: an existing call site
    that doesn't pass sort_mode at all must still send a well-formed
    request (with "sort": null), matching what an updated server expects."""
    socket_path = tmp_path / "daemon.sock"
    monkeypatch.setattr(daemon, "DAEMON_SOCKET_PATH", socket_path)
    received = {}

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_run_capturing_server, args=(socket_path, received, ready_event)
    )
    server_thread.start()
    ready_event.wait(timeout=2)

    daemon.send_query("happy", 10, timeout=2.0)

    server_thread.join(timeout=2)
    assert received["request"] == {"query": "happy", "top_n": 10, "sort": None}


def test_handle_request_passes_sort_mode_through_to_search_fn():
    calls = {}

    def fake_search(query, top_n, sort_mode):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10, "sort": "alpha"})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"sort_mode": "alpha"}


def test_handle_request_defaults_sort_mode_to_none_for_requests_without_it():
    """Backward compatibility for the SERVER side: an OLD client's request
    (no "sort" key at all, not even null) must still work, with sort_mode
    defaulting to None."""
    calls = {}

    def fake_search(query, top_n, sort_mode):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    request_text = json.dumps({"query": "happy", "top_n": 10})

    daemon._handle_request(request_text, fake_search)

    assert calls == {"sort_mode": None}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_daemon.py -v -k "sort_mode"`
Expected: FAIL — `send_query` has no `sort_mode` parameter yet (`TypeError`), and `_handle_request` doesn't read `request.get("sort")` yet.

- [ ] **Step 5: Implement the wire-protocol changes**

In `src/revdict/daemon.py`, change `send_query`'s signature and request-building line (currently):

```python
def send_query(query: str, top_n: int, timeout: float = 30.0) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps({"query": query, "top_n": top_n})
```

to:

```python
def send_query(
    query: str, top_n: int, sort_mode: str | None = None, timeout: float = 30.0
) -> dict | None:
    if not DAEMON_SOCKET_PATH.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DAEMON_SOCKET_PATH))
            request = json.dumps({"query": query, "top_n": top_n, "sort": sort_mode})
```

No other line in `send_query` changes.

Change `_handle_request` (currently):

```python
def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(request["query"], top_n=request["top_n"])
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)
```

to:

```python
def _handle_request(request_text: str, search_fn) -> str:
    try:
        request = json.loads(request_text)
        result = search_fn(request["query"], top_n=request["top_n"], sort_mode=request.get("sort"))
    except Exception as error:
        return json.dumps({"error": str(error)})
    return json.dumps(result)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_daemon.py -v`
Expected: PASS (all existing tests plus the new sort-mode tests)

- [ ] **Step 7: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — `run_server`'s existing `_handle_request(request_text, search_mod.search)` call site needs no change, since Task 2 already gave `search()` a `sort_mode` parameter with a default, so it satisfies the new 3-arg `search_fn` contract automatically.

- [ ] **Step 8: Commit**

```bash
git add src/revdict/daemon.py tests/test_daemon.py
git commit -m "Thread sort_mode through the daemon wire protocol, backward compatible"
```

---

### Task 4: `--sort` CLI flag, end to end, and README docs

**Files:**
- Modify: `src/revdict/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `sort.SORT_MODES` (Task 1), `search.search`'s `sort_mode` parameter (Task 2, via the daemon/local-fallback path built in Task 3), `daemon.send_query`'s `sort_mode` parameter (Task 3).
- Produces: `_get_search_result(query: str, top_n: int, sort_mode: str | None = None) -> dict`, `_local_search_fallback(query: str, top_n: int, sort_mode: str | None = None) -> dict`, `_run_query(query: str, top_n: int, interactive: bool, sort_mode: str | None = None) -> int` — all three existing functions gain a new optional `sort_mode` parameter with a default of `None`, so every call site that doesn't pass it keeps working unchanged.

- [ ] **Step 1: Update the 12 existing `_get_search_result` mocks in `tests/test_cli.py` to accept the new keyword**

`_run_query` (modified in Step 3 below) will call `_get_search_result(query, top_n, sort_mode=sort_mode)` — an explicit keyword argument on every call, regardless of whether `sort_mode` is `None`. Every existing test that mocks `_get_search_result` with a lambda accepting only `(query, top_n)` will raise `TypeError: <lambda>() got an unexpected keyword argument 'sort_mode'` once that happens. Run this exact search-and-replace across `tests/test_cli.py` — every one of these 12 lines has the identical shape `lambda query, top_n: <expr>` and needs `sort_mode=None` added as a third parameter with a default (so the lambda still works if called the old way too, though nothing will call it that way after this task):

```
grep -n '"_get_search_result", lambda query, top_n:' tests/test_cli.py
```

This should list exactly these line numbers (verify against your checkout — line numbers may have shifted slightly since this plan was written, but the count and the lambda text should match): 233, 264, 290, 314, 340, 355, 372, 431, 467, 499, 538, 771.

For each, change `lambda query, top_n: <expr>` to `lambda query, top_n, sort_mode=None: <expr>` — the `<expr>` part (whatever result the test's fake returns, e.g. `fake_result` or `_FAKE_INTERACTIVE_RESULT` or a literal dict) is unchanged; only the lambda's parameter list gains `, sort_mode=None`.

- [ ] **Step 2: Run the full `test_cli.py` file to confirm this mechanical update alone doesn't break anything (before adding new sort-flag tests)**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS — no test's assertions change, only the mock signatures; `sort_mode` isn't threaded through the real code yet in this step, so nothing calls these lambdas with the new keyword yet either. This step is a pure safety checkpoint before the next step wires the real keyword-passing through.

- [ ] **Step 3: Write the failing tests for the `--sort` flag itself**

```python
# tests/test_cli.py (append)


def test_query_parser_accepts_all_seven_sort_modes():
    from revdict import cli

    parser = cli._query_parser()

    for mode in ("relevance", "alpha", "alpha_desc", "shortest", "longest", "most_common", "least_common"):
        args = parser.parse_args(["happy", "--sort", mode])
        assert args.sort == mode


def test_query_parser_rejects_an_invalid_sort_mode():
    import pytest

    from revdict import cli

    parser = cli._query_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["happy", "--sort", "nonsense"])


def test_query_parser_sort_defaults_to_none():
    from revdict import cli

    parser = cli._query_parser()

    args = parser.parse_args(["happy"])
    assert args.sort is None


def test_main_passes_sort_flag_through_to_get_search_result(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--sort", "alpha", "--no-interactive"])

    assert code == 0
    assert calls["sort_mode"] == "alpha"


def test_main_without_sort_flag_passes_none(monkeypatch):
    from revdict import cli

    monkeypatch.setattr(cli, "_index_exists", lambda: True)
    calls = {}

    def fake_get_search_result(query, top_n, sort_mode=None):
        calls["sort_mode"] = sort_mode
        return {"exact_match": None, "candidates": []}

    monkeypatch.setattr(cli, "_get_search_result", fake_get_search_result)

    code = cli.main(["happy", "--no-interactive"])

    assert code == 0
    assert calls["sort_mode"] is None
```

Note: this file does NOT import `pytest` at the top — the one existing test that needs it (`test_main_returns_1_with_clean_error_on_non_numeric_n_value`) imports it locally inside the test function. Match that convention: the new `pytest.raises` usage above already has its own local `import pytest`, matching the file's established style — don't add a top-level import.

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v -k "sort"`
Expected: FAIL — `_query_parser()` has no `--sort` argument yet, so `parse_args([..., "--sort", mode])` raises `_ArgumentError`/`SystemExit` for an unrecognized flag on every test; `main()` doesn't read or forward `args.sort` yet.

- [ ] **Step 5: Implement the CLI wiring**

Add the import at the top of `src/revdict/cli.py` (alongside the existing `revdict` imports):

```python
from revdict import sort
```

In `_query_parser()`, add the new argument (after the existing `--no-interactive` argument, before `return parser`):

```python
    parser.add_argument(
        "--sort",
        choices=list(sort.SORT_MODES),
        default=None,
        help='Sort order for results (default: relevance, i.e. "most similar").',
    )
```

Change `_local_search_fallback` (currently `def _local_search_fallback(query: str, top_n: int) -> dict:`) to:

```python
def _local_search_fallback(query: str, top_n: int, sort_mode: str | None = None) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(query, top_n=top_n, sort_mode=sort_mode)
```

Change `_get_search_result` (currently `def _get_search_result(query: str, top_n: int) -> dict:`) to:

```python
def _get_search_result(query: str, top_n: int, sort_mode: str | None = None) -> dict:
    result = daemon.send_query(query, top_n, sort_mode=sort_mode)
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(query, top_n, sort_mode=sort_mode)
        if result is not None:
            return result
    return _local_search_fallback(query, top_n, sort_mode=sort_mode)
```

Change `_run_query` (currently `def _run_query(query: str, top_n: int, interactive: bool) -> int:`) to:

```python
def _run_query(query: str, top_n: int, interactive: bool, sort_mode: str | None = None) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(query, top_n, sort_mode=sort_mode)
```

(the rest of `_run_query`'s body — the `if interactive:` branch, the picker call, everything after the `result = ...` line — is unchanged).

In `main()`, change the final dispatch line (currently `return _run_query(query, args.n, interactive)`) to:

```python
    return _run_query(query, args.n, interactive, sort_mode=args.sort)
```

Leave the earlier no-argv stdin-read call site (`return _run_query(query, top_n=30, interactive=False)`, inside the `if not argv:` block) **unchanged** — that path never parses `--sort` (there's no argv to parse it from), so it correctly keeps defaulting to `sort_mode=None`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (all existing tests, the Step 1 mock updates, and the new sort-flag tests)

- [ ] **Step 7: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — full regression check. Expect exactly 2 pre-existing, environment-only failures unrelated to this plan (`test_main_error_message_is_not_mangled_by_rich_markup`, `test_main_routes_daemon_status` — caused by a `FORCE_COLOR` shell variable forcing ANSI codes; reproduces with zero code changes when `FORCE_COLOR` is set, clean under `NO_COLOR=1`). Any other failure is a real regression from this task.

- [ ] **Step 8: Document the sort flag in README.md**

In `README.md`'s "Query syntax" section (added by the prior query-syntax plan), add a new subsection immediately after it:

```markdown
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
```

- [ ] **Step 9: Run the full test suite one final time**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — same result as Step 7 (README changes don't affect test outcomes), confirming Phase 2 is complete.

- [ ] **Step 10: Commit**

```bash
git add src/revdict/cli.py tests/test_cli.py README.md
git commit -m "Add --sort CLI flag, threaded through the daemon and local-fallback paths"
```

---

## Self-review notes (for the record)

- **Spec coverage:** all 6 new sort modes from TODO.md's zero-new-data subset (A→Z, Z→A, Shortest, Longest, Most Common, Least Common) are implemented in Task 1 and covered end-to-end (module → `search()` → daemon → CLI → README) by Tasks 2-4. "Most Similar" (the existing default) is preserved unchanged, not reimplemented. "Most formal," "Most modern/oldest," "Most lyrical," "Most funny-sounding," and the emotional-buckets stretch item are explicitly out of scope per the roadmap and are not touched by this plan.
- **No placeholders:** every step shows complete, real code; Task 4 Step 1's mechanical mock-signature update gives the exact transformation pattern and the exact expected line count/numbers rather than a vague "update the mocks" instruction.
- **Backward compatibility:** proven explicitly at every layer — `search()` (Task 2's default-order test), the daemon wire protocol (Task 3's two "defaults to None when omitted" tests, both client- and server-side), and the CLI (Task 4's "without sort flag passes None" test).
- **Type/name consistency:** `sort_mode: str | None = None` is the parameter name used identically in `search()` (Task 2), `daemon.send_query`/`_handle_request` (Task 3), and `cli._get_search_result`/`_local_search_fallback`/`_run_query` (Task 4). The wire-protocol JSON key is `"sort"` (not `"sort_mode"`) consistently in both `send_query`'s request-building and `_handle_request`'s `request.get("sort")` read. `sort.SORT_MODES`/`sort.apply_sort` (Task 1) are consumed with matching names in Task 2 (`search.py`) and Task 4 (`cli.py`'s `choices=list(sort.SORT_MODES)`).
