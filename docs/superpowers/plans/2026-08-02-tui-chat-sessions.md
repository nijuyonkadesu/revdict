# TUI Chat Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep one provider-neutral, in-memory chat conversation for each exact selected dictionary sense during a revdict TUI run.

**Architecture:** `ChatSession` owns provider history and a visible Markdown transcript for a `ChatSessionKey` composed of headword, part of speech, and definition. Its bootstrap context is prepended to the first real user turn; later turns replay that first turn and conversation history, so the definition is not injected repeatedly. Sessions are shared across providers and disappear when the TUI exits.

**Tech Stack:** Python 3.11+, prompt-toolkit, Rich, pytest.

## Global Constraints

- Do not persist chat sessions after revdict exits.
- Share one sense-specific session across providers.
- Preserve the single worker and 30 FPS streamed redraw cap.
- Do not alter untracked `check.txt`.

---

### Task 1: Seed lexical context once in history

**Files:**
- Modify: `src/revdict/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Produces: `writing_instruction() -> str` and `lexical_bootstrap(context: LexicalContext) -> str`.
- Changes: `ChatClient.stream(provider, history, message, on_chunk) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_stream_replays_bootstrap_history_without_definition_in_system_prompt():
    captured = []
    client = ChatClient(stream_transport=lambda _u, _h, payload: captured.append(payload) or ["data: [DONE]\\n"], environment={})
    history = [("user", lexical_bootstrap(LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")))]
    with pytest.raises(ChatRequestError):
        client.stream(ProviderSettings("ollama", "http://example.test", "model", None), history, "More examples?", lambda _chunk: None)
    assert "Definition:" not in captured[0]["messages"][0]["content"]
    assert "Definition: a fine cotton fabric" in captured[0]["messages"][1]["content"]
```

- [ ] **Step 2: Run it and verify it fails**

Run: `.venv/bin/pytest -q tests/test_chat.py::test_stream_replays_bootstrap_history_without_definition_in_system_prompt`

- [ ] **Step 3: Implement the minimal change**

```python
def writing_instruction() -> str:
    return "You are revdict's concise writing assistant. Explain writing and spoken use clearly."

def lexical_bootstrap(context: LexicalContext) -> str:
    return f"Search query: {context.query}\\nHighlighted word: {context.headword} ({context.part_of_speech})\\nDefinition: {context.definition}"
```

All four request builders serialize `history` verbatim after the static writing instruction.

- [ ] **Step 4: Run the chat tests and commit**

Run: `.venv/bin/pytest -q tests/test_chat.py`

Commit: `git add src/revdict/chat.py tests/test_chat.py && git commit -m "fix: seed chat context once per session"`

### Task 2: Restore one session for each selected sense

**Files:**
- Modify: `src/revdict/tui.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Produces: `ChatSessionKey.from_context(context)`, `ChatSession.bootstrap`, `ChatSession.history`, and `ChatSession.transcript`.
- Consumes: `chat_module.lexical_bootstrap(context)`.

- [ ] **Step 1: Write the failing test**

```python
def test_chat_sessions_restore_per_sense_and_survive_provider_changes():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    percale_context = LexicalContext("fabric", "percale", "a fine cotton fabric", "noun")
    reverence_context = LexicalContext("respect", "reverence", "deep respect", "noun")
    ui._activate_chat_session(percale_context)
    ui._append_chat_turn("You", "First question")
    ui._activate_chat_session(reverence_context)
    assert ui._chat_transcript_text == ""
    ui._activate_chat_session(percale_context)
    assert "First question" in ui._chat_transcript_text
```

- [ ] **Step 2: Run it and verify it fails**

Run: `.venv/bin/pytest -q tests/test_tui.py::test_chat_sessions_restore_per_sense_and_survive_provider_changes`

- [ ] **Step 3: Implement the minimal change**

```python
@dataclass(frozen=True)
class ChatSessionKey:
    headword: str
    part_of_speech: str
    definition: str

@dataclass
class ChatSession:
    history: list[tuple[str, str]]
    transcript: str = ""
```

On opening F4, activate the selected sense’s session, creating a bootstrap string only when absent. Prepend it to the first real user message, restore that session’s transcript, and use its history for later sends.

- [ ] **Step 4: Run the TUI tests and commit**

Run: `.venv/bin/pytest -q tests/test_tui.py`

Commit: `git add src/revdict/tui.py tests/test_tui.py && git commit -m "fix: retain TUI chats by selected sense"`

### Task 3: Scope streamed events to their initiating session

**Files:**
- Modify: `src/revdict/tui.py`, `src/revdict/chat.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: `ChatSessionKey` and `ChatSession`.
- Produces: session-keyed request, chunk, result, and error callbacks.

- [ ] **Step 1: Write the failing test**

```python
def test_streamed_answer_is_committed_to_its_requesting_session():
    ui = NativeTui(lambda _query, **_kwargs: {"exact_match": None, "candidates": []})
    key = ChatSessionKey("percale", "noun", "a fine cotton fabric")
    ui._activate_chat_session_key(key)
    ui._begin_chat_response(key)
    ui._queue_chat_chunk(key, "Answer")
    ui._flush_chat_chunks()
    assert "Answer" in ui._chat_sessions[key].transcript
```

- [ ] **Step 2: Run it and verify it fails**

Run: `.venv/bin/pytest -q tests/test_tui.py::test_streamed_answer_is_committed_to_its_requesting_session`

- [ ] **Step 3: Implement and verify**

Pass the key with each worker request and streamed callback. Store chunks and final answer in that key’s session; repaint only if it is active. Preserve the one-worker and 30 FPS batching behavior.

Run: `.venv/bin/pytest -q tests/test_tui.py tests/test_chat.py`

Commit: `git add src/revdict/chat.py src/revdict/tui.py tests/test_chat.py tests/test_tui.py && git commit -m "fix: scope streamed chat replies to sessions"`

### Task 4: Final verification and local merge

- [ ] Run: `.venv/bin/pytest -q && pgrep -af 'python.*revdict\\.cli daemon'`
- [ ] Confirm all tests pass and exactly one daemon is running.
- [ ] Run: `git switch main && git merge --ff-only fix/tui-chat-sessions`
