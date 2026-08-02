"""Provider-neutral writing chat and non-secret local settings for revdict."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHAT_SETTINGS_PATH = Path.home() / ".config" / "revdict" / "chat.json"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")


class ChatConfigurationError(ValueError):
    """The selected provider is not configured well enough to send a chat."""


class ChatRequestError(RuntimeError):
    """A provider rejected or could not complete a chat request."""


@dataclass(frozen=True)
class LexicalContext:
    query: str
    headword: str
    definition: str
    part_of_speech: str


@dataclass
class ProviderSettings:
    provider: str
    base_url: str
    model: str
    api_key_env: str | None
    api_key: str | None = None


@dataclass
class ChatSettings:
    active_provider: str
    providers: dict[str, ProviderSettings]
    gemini_models: list[str] = field(default_factory=list)

    @classmethod
    def defaults(cls) -> ChatSettings:
        return cls(
            active_provider="ollama",
            providers={
                "openai": ProviderSettings("openai", "https://api.openai.com/v1", "gpt-4.1-mini", "OPENAI_API_KEY"),
                "anthropic": ProviderSettings("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
                "gemini": ProviderSettings("gemini", GEMINI_API_BASE, "gemini-3.6-flash", "REVDICT_GEMINI_API_KEY"),
                "ollama": ProviderSettings("ollama", "http://localhost:11434", "", "REVDICT_OLLAMA_API_KEY"),
            },
        )


def load_settings(path: Path | None = None) -> ChatSettings:
    path = CHAT_SETTINGS_PATH if path is None else path
    if not path.exists():
        return ChatSettings.defaults()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        providers = {
            name: ProviderSettings(name, values["base_url"], values["model"], values.get("api_key_env"), values.get("api_key"))
            for name, values in raw["providers"].items()
            if name in SUPPORTED_PROVIDERS
        }
        defaults = ChatSettings.defaults()
        providers = {**defaults.providers, **providers}
        active_provider = raw.get("active_provider", defaults.active_provider)
        if active_provider not in providers:
            active_provider = defaults.active_provider
        return ChatSettings(active_provider, providers, list(raw.get("gemini_models", [])))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise ChatConfigurationError(f"Could not read chat settings at {path}: {error}") from error


def save_settings(settings: ChatSettings, path: Path | None = None) -> None:
    path = CHAT_SETTINGS_PATH if path is None else path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "active_provider": settings.active_provider,
        "providers": {name: asdict(config) | {"provider": None} for name, config in settings.providers.items()},
        "gemini_models": settings.gemini_models,
    }
    for config in payload["providers"].values():
        config.pop("provider")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def filter_gemini_models(payload: Mapping) -> list[str]:
    """Keep text-chat Gemini/Gemma models; exclude audio, image, and tool-only APIs."""
    excluded = ("tts", "image", "computer-use", "robotics", "lyria", "deep-research", "antigravity", "customtools", "omni")
    models = []
    for model in payload.get("models", []):
        name = str(model.get("name", "")).removeprefix("models/")
        methods = model.get("supportedGenerationMethods", [])
        if name and "generateContent" in methods and not any(token in name for token in excluded):
            models.append(name)
    return sorted(set(models))


def _get_json(url: str, headers: dict[str, str]) -> dict:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        raise ChatRequestError(f"Provider returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ChatRequestError(f"Could not reach provider: {error.reason if isinstance(error, URLError) else error}") from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ChatRequestError("Provider returned an invalid model list.") from error


def discover_gemini_models(api_key: str, *, fetch: Callable[[str, dict[str, str]], dict] = _get_json) -> list[str]:
    """Fetch and retain only general-purpose text models from Gemini's model API."""
    if not api_key:
        raise ChatConfigurationError("Set REVDICT_GEMINI_API_KEY before refreshing Gemini models.")
    return filter_gemini_models(fetch(GEMINI_API_BASE + "/models", {"x-goog-api-key": api_key}))


def test_provider(
    provider: ProviderSettings,
    *,
    environment: Mapping[str, str] | None = None,
    fetch: Callable[[str, dict[str, str]], dict] = _get_json,
) -> list[str]:
    """Make one cheap models request; this intentionally never generates text."""
    environment = os.environ if environment is None else environment
    api_key = provider.api_key or (environment.get(provider.api_key_env) if provider.api_key_env else None)
    if provider.provider != "ollama" and not api_key:
        raise ChatConfigurationError(f"Set or configure {provider.api_key_env} before testing {provider.provider}.")
    if provider.provider in {"openai", "ollama"}:
        base_url = provider.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = fetch(base_url + "/models", headers)
        return sorted(item["id"] for item in response.get("data", []) if item.get("id"))
    if provider.provider == "anthropic":
        response = fetch(provider.base_url.rstrip("/") + "/models", {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"})
        return sorted(item["id"] for item in response.get("data", []) if item.get("id"))
    if provider.provider == "gemini":
        return discover_gemini_models(api_key or "", fetch=fetch)
    raise ChatConfigurationError(f"Unsupported chat provider: {provider.provider}")


def lexical_instruction(context: LexicalContext) -> str:
    return (
        "You are revdict's concise writing assistant. Use the current lexical context accurately. "
        "When explaining use, distinguish formal writing from natural spoken English; include register, collocations, "
        "two short writing examples, two short spoken examples, and a likely misuse when useful.\n\n"
        f"Search query: {context.query}\n"
        f"Highlighted word: {context.headword} ({context.part_of_speech})\n"
        f"Definition: {context.definition}\n\n"
        "Writing: use the word deliberately and match its register.\n"
        "Speech: prefer natural phrasing and say when a simpler alternative sounds more conversational."
    )


def default_writing_prompt(context: LexicalContext) -> str:
    return (
        f"How can I use “{context.headword}” naturally in writing and in spoken conversation? "
        "Give register guidance, collocations, two writing examples, two spoken examples, and common mistakes."
    )


Transport = Callable[[str, dict[str, str], dict], dict]
StreamTransport = Callable[[str, dict[str, str], dict], Iterable[str | bytes]]


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        raise ChatRequestError(f"Provider returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ChatRequestError(f"Could not reach provider: {error.reason if isinstance(error, URLError) else error}") from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ChatRequestError("Provider returned an invalid response.") from error


def _post_sse(url: str, headers: dict[str, str], payload: dict) -> Iterable[str | bytes]:
    """Yield the lines of a provider's Server-Sent Events response."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            yield from response
    except HTTPError as error:
        raise ChatRequestError(f"Provider returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ChatRequestError(f"Could not reach provider: {error.reason if isinstance(error, URLError) else error}") from error


def _sse_events(lines: Iterable[str | bytes]) -> Iterable[dict]:
    """Decode single-line JSON SSE events emitted by supported chat APIs."""
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if not data:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ChatRequestError("Provider returned an invalid streaming response.") from error
        if isinstance(payload, dict):
            yield payload


class ChatClient:
    """Turns one provider-neutral conversation turn into the provider's HTTP API."""

    def __init__(
        self,
        *,
        transport: Transport = _post_json,
        stream_transport: StreamTransport = _post_sse,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._transport = transport
        self._stream_transport = stream_transport
        self._environment = os.environ if environment is None else environment

    def complete(
        self,
        provider: ProviderSettings,
        context: LexicalContext,
        history: list[tuple[str, str]],
        message: str,
    ) -> str:
        if not provider.model.strip():
            raise ChatConfigurationError(f"Choose a model for {provider.provider} before sending a message.")
        api_key = self._api_key(provider)
        if provider.provider in {"openai", "ollama"}:
            return self._openai_compatible(provider, context, history, message, api_key)
        if provider.provider == "anthropic":
            return self._anthropic(provider, context, history, message, api_key)
        if provider.provider == "gemini":
            return self._gemini(provider, context, history, message, api_key)
        raise ChatConfigurationError(f"Unsupported chat provider: {provider.provider}")

    def stream(
        self,
        provider: ProviderSettings,
        context: LexicalContext,
        history: list[tuple[str, str]],
        message: str,
        on_chunk: Callable[[str], None],
    ) -> str:
        """Stream a response and return the exact assembled assistant reply."""
        if not provider.model.strip():
            raise ChatConfigurationError(f"Choose a model for {provider.provider} before sending a message.")
        api_key = self._api_key(provider)
        if provider.provider in {"openai", "ollama"}:
            return self._stream_openai_compatible(provider, context, history, message, api_key, on_chunk)
        if provider.provider == "anthropic":
            return self._stream_anthropic(provider, context, history, message, api_key, on_chunk)
        if provider.provider == "gemini":
            return self._stream_gemini(provider, context, history, message, api_key, on_chunk)
        raise ChatConfigurationError(f"Unsupported chat provider: {provider.provider}")

    def _api_key(self, provider: ProviderSettings) -> str | None:
        if not provider.api_key_env:
            return None
        api_key = provider.api_key or self._environment.get(provider.api_key_env)
        if provider.provider == "ollama":
            return api_key
        if not api_key:
            raise ChatConfigurationError(f"Set {provider.api_key_env} before using {provider.provider}.")
        return api_key

    def _stream_response(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        extract_text: Callable[[dict], str],
        on_chunk: Callable[[str], None],
        *,
        cumulative: bool = False,
    ) -> str:
        chunks: list[str] = []
        for event in _sse_events(self._stream_transport(url, headers, payload)):
            piece = extract_text(event)
            if cumulative and chunks:
                assembled = "".join(chunks)
                if piece.startswith(assembled):
                    piece = piece[len(assembled):]
            if piece:
                chunks.append(piece)
                on_chunk(piece)
        answer = "".join(chunks).strip()
        if not answer:
            raise ChatRequestError("Provider did not return a chat response.")
        return answer

    def _stream_openai_compatible(
        self,
        provider: ProviderSettings,
        context: LexicalContext,
        history: list[tuple[str, str]],
        message: str,
        api_key: str | None,
        on_chunk: Callable[[str], None],
    ) -> str:
        base_url = provider.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        payload = {
            "model": provider.model,
            "stream": True,
            "messages": [{"role": "system", "content": lexical_instruction(context)}]
            + [{"role": role, "content": text} for role, text in history]
            + [{"role": "user", "content": message}],
        }

        def extract_text(event: dict) -> str:
            try:
                return event["choices"][0]["delta"].get("content") or ""
            except (IndexError, KeyError, TypeError, AttributeError):
                return ""

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return self._stream_response(base_url + "/chat/completions", headers, payload, extract_text, on_chunk)

    def _stream_anthropic(
        self,
        provider: ProviderSettings,
        context: LexicalContext,
        history: list[tuple[str, str]],
        message: str,
        api_key: str | None,
        on_chunk: Callable[[str], None],
    ) -> str:
        assert api_key is not None
        payload = {
            "model": provider.model,
            "max_tokens": 900,
            "stream": True,
            "system": lexical_instruction(context),
            "messages": [{"role": role, "content": text} for role, text in history] + [{"role": "user", "content": message}],
        }

        def extract_text(event: dict) -> str:
            if event.get("type") != "content_block_delta":
                return ""
            delta = event.get("delta", {})
            return delta.get("text", "") if isinstance(delta, dict) else ""

        return self._stream_response(
            provider.base_url.rstrip("/") + "/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            payload,
            extract_text,
            on_chunk,
        )

    def _stream_gemini(
        self,
        provider: ProviderSettings,
        context: LexicalContext,
        history: list[tuple[str, str]],
        message: str,
        api_key: str | None,
        on_chunk: Callable[[str], None],
    ) -> str:
        assert api_key is not None
        contents = [
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
            for role, text in history
        ]
        contents.append({"role": "user", "parts": [{"text": message}]})
        payload = {"systemInstruction": {"parts": [{"text": lexical_instruction(context)}]}, "contents": contents}

        def extract_text(event: dict) -> str:
            try:
                return "".join(part["text"] for part in event["candidates"][0]["content"]["parts"] if "text" in part)
            except (IndexError, KeyError, TypeError, AttributeError):
                return ""

        url = provider.base_url.rstrip("/") + f"/models/{provider.model}:streamGenerateContent?alt=sse"
        return self._stream_response(url, {"x-goog-api-key": api_key}, payload, extract_text, on_chunk, cumulative=True)

    def _openai_compatible(self, provider: ProviderSettings, context: LexicalContext, history: list[tuple[str, str]], message: str, api_key: str | None) -> str:
        base_url = provider.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "model": provider.model,
            "stream": False,
            "messages": [{"role": "system", "content": lexical_instruction(context)}]
            + [{"role": role, "content": text} for role, text in history]
            + [{"role": "user", "content": message}],
        }
        response = self._transport(base_url + "/chat/completions", headers, payload)
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (IndexError, KeyError, TypeError, AttributeError) as error:
            raise ChatRequestError("Provider did not return a chat response.") from error

    def _anthropic(self, provider: ProviderSettings, context: LexicalContext, history: list[tuple[str, str]], message: str, api_key: str | None) -> str:
        assert api_key is not None
        payload = {
            "model": provider.model,
            "max_tokens": 900,
            "system": lexical_instruction(context),
            "messages": [{"role": role, "content": text} for role, text in history] + [{"role": "user", "content": message}],
        }
        response = self._transport(
            provider.base_url.rstrip("/") + "/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            payload,
        )
        try:
            return response["content"][0]["text"].strip()
        except (IndexError, KeyError, TypeError, AttributeError) as error:
            raise ChatRequestError("Provider did not return a chat response.") from error

    def _gemini(self, provider: ProviderSettings, context: LexicalContext, history: list[tuple[str, str]], message: str, api_key: str | None) -> str:
        assert api_key is not None
        contents = [
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
            for role, text in history
        ]
        contents.append({"role": "user", "parts": [{"text": message}]})
        payload = {"systemInstruction": {"parts": [{"text": lexical_instruction(context)}]}, "contents": contents}
        response = self._transport(
            provider.base_url.rstrip("/") + f"/models/{provider.model}:generateContent",
            {"x-goog-api-key": api_key},
            payload,
        )
        try:
            return "".join(part["text"] for part in response["candidates"][0]["content"]["parts"]).strip()
        except (IndexError, KeyError, TypeError, AttributeError) as error:
            raise ChatRequestError("Provider did not return a chat response.") from error


class ChatController:
    """A single persistent worker; a slow provider can never be spammed in parallel."""

    def __init__(self, execute: Callable[[object], str], on_result: Callable[[str], None], on_error: Callable[[Exception], None]) -> None:
        self._execute, self._on_result, self._on_error = execute, on_result, on_error
        self._condition = threading.Condition()
        self._pending: object | None = None
        self._busy = False
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="revdict-chat", daemon=True)
        self._worker.start()

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._busy

    def send(self, message: object) -> bool:
        with self._condition:
            if self._closed or self._busy or message is None:
                return False
            self._busy = True
            self._pending = message
            self._condition.notify()
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._worker.join(timeout=0.2)

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending is None:
                    self._condition.wait()
                if self._closed:
                    return
                message = self._pending
                self._pending = None
            try:
                answer = self._execute(message)
            except Exception as error:
                self._on_error(error)
            else:
                self._on_result(answer)
            finally:
                with self._condition:
                    self._busy = False
