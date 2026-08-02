import json
import threading

import pytest

from revdict import chat


def test_openai_compatible_chat_includes_lexical_context_and_omits_empty_auth():
    calls = []

    def transport(url, headers, payload):
        calls.append((url, headers, payload))
        return {"choices": [{"message": {"content": "Use tailor for deliberate adaptation."}}]}

    client = chat.ChatClient(transport=transport, environment={})
    response = client.complete(
        chat.ProviderSettings("ollama", "http://example.test:11434", "Qwen3.6:35b-a3b", None),
        chat.LexicalContext("make clothing fit", "tailor", "a person who makes or alters garments", "noun"),
        [("user", "What register is it?")],
        "How can I use this word in writing and speech?",
    )

    assert response == "Use tailor for deliberate adaptation."
    url, headers, payload = calls[0]
    assert url == "http://example.test:11434/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["model"] == "Qwen3.6:35b-a3b"
    assert "Highlighted word: tailor (noun)" in payload["messages"][0]["content"]
    assert "Writing:" in payload["messages"][0]["content"]
    assert payload["messages"][-1] == {"role": "user", "content": "How can I use this word in writing and speech?"}


def test_gemini_chat_uses_generate_content_shape_and_parses_text():
    calls = []

    def transport(url, headers, payload):
        calls.append((url, headers, payload))
        return {"candidates": [{"content": {"parts": [{"text": "A concise answer."}]}}]}

    client = chat.ChatClient(transport=transport, environment={"REVDICT_GEMINI_API_KEY": "test-key"})
    response = client.complete(
        chat.ProviderSettings("gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini-3.6-flash", "REVDICT_GEMINI_API_KEY"),
        chat.LexicalContext("closely woven pillow fabric", "percale", "a fine cotton fabric", "noun"),
        [("assistant", "Earlier reply")],
        "Give me spoken examples.",
    )

    assert response == "A concise answer."
    url, headers, payload = calls[0]
    assert url.endswith("/models/gemini-3.6-flash:generateContent")
    assert headers["x-goog-api-key"] == "test-key"
    assert payload["contents"][0] == {"role": "model", "parts": [{"text": "Earlier reply"}]}
    assert "Search query: closely woven pillow fabric" in payload["systemInstruction"]["parts"][0]["text"]


def test_openai_compatible_chat_streams_each_sse_delta_with_full_context():
    calls = []
    chunks = []

    def stream_transport(url, headers, payload):
        calls.append((url, headers, payload))
        return [
            b'data: {"choices": [{"delta": {"content": "Use "}}]}\n',
            b'data: {"choices": [{"delta": {"content": "tailor."}}]}\n',
            b"data: [DONE]\n",
        ]

    client = chat.ChatClient(stream_transport=stream_transport, environment={})
    response = client.stream(
        chat.ProviderSettings("ollama", "http://example.test:11434", "Qwen3.6:35b-a3b", None),
        chat.LexicalContext("make clothing fit", "tailor", "a person who makes or alters garments", "noun"),
        [],
        "How should I use it?",
        chunks.append,
    )

    assert response == "Use tailor."
    assert chunks == ["Use ", "tailor."]
    url, _headers, payload = calls[0]
    assert url == "http://example.test:11434/v1/chat/completions"
    assert payload["stream"] is True
    assert "Definition: a person who makes or alters garments" in payload["messages"][0]["content"]


def test_gemini_chat_streams_server_sent_events():
    calls = []
    chunks = []

    def stream_transport(url, headers, payload):
        calls.append((url, headers, payload))
        return [
            'data: {"candidates": [{"content": {"parts": [{"text": "Natural "}]}}]}\n',
            'data: {"candidates": [{"content": {"parts": [{"text": "speech."}]}}]}\n',
        ]

    client = chat.ChatClient(stream_transport=stream_transport, environment={"REVDICT_GEMINI_API_KEY": "test-key"})
    response = client.stream(
        chat.ProviderSettings("gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini-3.6-flash", "REVDICT_GEMINI_API_KEY"),
        chat.LexicalContext("closely woven pillow fabric", "percale", "a fine cotton fabric", "noun"),
        [],
        "Give spoken examples.",
        chunks.append,
    )

    assert response == "Natural speech."
    assert chunks == ["Natural ", "speech."]
    url, headers, payload = calls[0]
    assert url.endswith("/models/gemini-3.6-flash:streamGenerateContent?alt=sse")
    assert headers["x-goog-api-key"] == "test-key"
    assert "Definition: a fine cotton fabric" in payload["systemInstruction"]["parts"][0]["text"]


def test_anthropic_chat_streams_content_block_deltas():
    chunks = []

    def stream_transport(_url, _headers, _payload):
        return [
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Use it naturally."}}\n',
        ]

    client = chat.ChatClient(stream_transport=stream_transport, environment={"ANTHROPIC_API_KEY": "test-key"})
    response = client.stream(
        chat.ProviderSettings("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
        chat.LexicalContext("make clothing fit", "tailor", "a person who makes garments", "noun"),
        [],
        "Give an example.",
        chunks.append,
    )

    assert response == "Use it naturally."
    assert chunks == ["Use it naturally."]


def test_required_key_is_reported_without_attempting_a_request():
    client = chat.ChatClient(transport=lambda *_args: pytest.fail("must not request"), environment={})

    with pytest.raises(chat.ChatConfigurationError, match="OPENAI_API_KEY"):
        client.complete(
            chat.ProviderSettings("openai", "https://api.openai.com/v1", "gpt-4.1-mini", "OPENAI_API_KEY"),
            chat.LexicalContext("happy", "joyful", "feeling great happiness", "adjective"), [], "Help me use this word."
        )


def test_discovered_gemini_text_models_are_persisted_without_a_key(tmp_path):
    path = tmp_path / "chat.json"

    settings = chat.ChatSettings.defaults()
    settings.gemini_models = chat.filter_gemini_models(
        {
            "models": [
                {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.6-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
            ]
        }
    )
    chat.save_settings(settings, path)

    saved = json.loads(path.read_text())
    assert saved["gemini_models"] == ["gemini-3.6-flash"]
    assert "test-key" not in path.read_text()
    assert chat.load_settings(path).gemini_models == ["gemini-3.6-flash"]


def test_gemini_model_discovery_uses_the_key_only_for_the_request():
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return {"models": [{"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]}]}

    assert chat.discover_gemini_models("test-key", fetch=fetch) == ["gemini-3.6-flash"]
    assert calls == [("https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": "test-key"})]


def test_chat_controller_keeps_only_one_provider_request_in_flight():
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []

    def execute(message):
        calls.append(message)
        started.set()
        assert release.wait(timeout=2)
        return "answer"

    controller = chat.ChatController(execute, lambda _answer: completed.set(), lambda error: pytest.fail(str(error)))
    try:
        assert controller.send(("first", "structured request")) is True
        assert started.wait(timeout=2)
        assert controller.send(("second", "structured request")) is False
        release.set()
        assert completed.wait(timeout=2)
        assert calls == [("first", "structured request")]
    finally:
        controller.close()
