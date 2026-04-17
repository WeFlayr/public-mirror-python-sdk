"""Weflayr instrumented wrapper for the OpenAI SDK.

Connector: **OpenAI**
Provider:  https://openai.com

Available connectors
--------------------
- :class:`OpenAI`      — synchronous client (drop-in for ``openai.OpenAI``)
- :class:`AsyncOpenAI` — async client (drop-in for ``openai.AsyncOpenAI``)

Instrumented methods
--------------------
All token-consuming OpenAI endpoints are covered:

**Chat completions** (``client.chat.completions``):

- :meth:`Completions.create`       — synchronous
- :meth:`AsyncCompletions.create`  — async

**Legacy text completions** (``client.completions``):

- :meth:`TextCompletions.create`       — synchronous
- :meth:`AsyncTextCompletions.create`  — async

**Embeddings** (``client.embeddings``):

- :meth:`Embeddings.create`       — synchronous
- :meth:`AsyncEmbeddings.create`  — async

**Responses API** (``client.responses``):

- :meth:`Responses.create`       — synchronous
- :meth:`AsyncResponses.create`  — async

**Text-to-Speech** (``client.audio.speech``):

- :meth:`Speech.create`       — synchronous  (billed by character count)
- :meth:`AsyncSpeech.create`  — async        (billed by character count)

**Transcriptions / Speech-to-Text** (``client.audio.transcriptions``):

- :meth:`Transcriptions.create`       — synchronous  (billed by tokens or audio seconds)
- :meth:`AsyncTranscriptions.create`  — async        (billed by tokens or audio seconds)

**Translations** (``client.audio.translations``):

- :meth:`Translations.create`       — synchronous  (billed by audio seconds, whisper-1 only)
- :meth:`AsyncTranslations.create`  — async        (billed by audio seconds, whisper-1 only)

Example::

    from weflayr.sdk.openai.client import OpenAI, AsyncOpenAI

    # Sync
    client = OpenAI(api_key="sk-...")
    # chat
    client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    # legacy text completions
    client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Hello")
    # embeddings
    client.embeddings.create(model="text-embedding-3-small", input="Hello")
    # responses API
    client.responses.create(model="gpt-4o", input="Hello")
    client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
    client.audio.transcriptions.create(model="whisper-1", file=open("audio.mp3", "rb"))
    client.audio.translations.create(model="whisper-1", file=open("audio.mp3", "rb"))

    # Async
    async_client = AsyncOpenAI(api_key="sk-...")
    await async_client.chat.completions.create(...)
    await async_client.completions.create(...)
    await async_client.embeddings.create(...)
    await async_client.responses.create(...)
    await async_client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
    await async_client.audio.transcriptions.create(model="whisper-1", file=open("audio.mp3", "rb"))
    await async_client.audio.translations.create(model="whisper-1", file=open("audio.mp3", "rb"))
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from openai import AsyncOpenAI as _AsyncOpenAI
from openai import OpenAI as _OpenAI
from openai.resources.chat.completions import AsyncCompletions as _AsyncCompletions
from openai.resources.chat.completions import Completions as _Completions
from openai.resources.completions import AsyncCompletions as _AsyncTextCompletions
from openai.resources.completions import Completions as _TextCompletions
from openai.resources.embeddings import AsyncEmbeddings as _AsyncEmbeddings
from openai.resources.embeddings import Embeddings as _Embeddings
from openai.resources.audio.speech import AsyncSpeech as _AsyncSpeech
from openai.resources.audio.speech import Speech as _Speech
from openai.resources.audio.transcriptions import AsyncTranscriptions as _AsyncTranscriptions
from openai.resources.audio.transcriptions import Transcriptions as _Transcriptions
from openai.resources.audio.translations import AsyncTranslations as _AsyncTranslations
from openai.resources.audio.translations import Translations as _Translations
from openai.types.audio import Translation as _Translation  # type: ignore[import]
from openai.types.audio import TranslationVerbose as _TranslationVerbose  # type: ignore[import]
from openai.resources.responses.responses import AsyncResponses as _AsyncResponses
from openai.resources.responses.responses import Responses as _Responses

_PROVIDER = "openai"

from weflayr.sdk.helpers import (
    CLIENT_ID,
    CLIENT_SECRET,
    INTAKE_URL,
    _auth_headers,
    _build_url,
    _error_payload,
    post_async,
    post_sync,
    track_async,
    track_sync,
)

# ---------------------------------------------------------------------------
# Usage extractors
# ---------------------------------------------------------------------------


def _usage(response) -> dict:
    """Extract token usage from an OpenAI chat completion response.

    Args:
        response: A ``ChatCompletion`` returned by the OpenAI SDK.

    Returns:
        A dict with ``prompt_tokens`` and ``completion_tokens`` (both ``int | None``).
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _text_completion_usage(response) -> dict:
    """Extract token usage from a legacy text completion response.

    Args:
        response: A ``Completion`` returned by the OpenAI SDK.

    Returns:
        A dict with ``prompt_tokens`` and ``completion_tokens`` (both ``int | None``).
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _embedding_usage(response) -> dict:
    """Extract token usage from an embeddings response.

    Args:
        response: A ``CreateEmbeddingResponse`` returned by the OpenAI SDK.

    Returns:
        A dict with ``prompt_tokens`` and ``total_tokens`` (both ``int | None``).
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _response_usage(response) -> dict:
    """Extract token usage from a Responses API response.

    Args:
        response: A ``Response`` returned by the OpenAI Responses API.

    Returns:
        A dict with ``input_tokens``, ``output_tokens``, and ``cached_tokens``
        (all ``int | None``).
    """
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cached_tokens": getattr(input_details, "cached_tokens", None),
    }


# ---------------------------------------------------------------------------
# Shared init helper
# ---------------------------------------------------------------------------

def _weflayr_kwargs(intake_url: str, client_id: str, bearer_token: str) -> dict:
    return {"intake_url": intake_url, "client_id": client_id, "bearer_token": bearer_token}


# ---------------------------------------------------------------------------
# Streaming proxy wrappers
# ---------------------------------------------------------------------------


class _TrackedSyncStream:
    """Transparent proxy around an OpenAI ``Stream[ChatCompletionChunk]``.

    Fires a ``chat.completions.create.after`` telemetry event once the stream
    is fully consumed or the context manager exits. The ``_fired`` flag prevents
    the event from being sent twice when both paths are used together.

    Injects ``stream_options={"include_usage": True}`` upstream so the final
    SSE chunk carries token counts.
    """

    def __init__(
        self,
        stream,
        *,
        start: float,
        request_id: str,
        before: dict,
        endpoint: str,
        headers: dict,
    ) -> None:
        self._stream = stream
        self._start = start
        self._request_id = request_id
        self._before = before
        self._endpoint = endpoint
        self._headers = headers
        self._usage = None
        self._fired = False

    def _fire_after(self) -> None:
        if self._fired:
            return
        self._fired = True
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        post_sync(
            self._endpoint,
            {
                "event_id": self._request_id,
                "event_type": "chat.completions.create.after",
                **self._before,
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": getattr(self._usage, "prompt_tokens", None),
                "completion_tokens": getattr(self._usage, "completion_tokens", None),
            },
            self._headers,
        )

    def __iter__(self):
        try:
            for chunk in self._stream:
                if getattr(chunk, "usage", None) is not None:
                    self._usage = chunk.usage
                yield chunk
        finally:
            self._fire_after()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        result = self._stream.__exit__(*args)
        self._fire_after()
        return result

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


class _TrackedAsyncStream:
    """Transparent proxy around an OpenAI ``AsyncStream[ChatCompletionChunk]``.

    Async equivalent of :class:`_TrackedSyncStream`.
    """

    def __init__(
        self,
        stream,
        *,
        start: float,
        request_id: str,
        before: dict,
        endpoint: str,
        headers: dict,
    ) -> None:
        self._stream = stream
        self._start = start
        self._request_id = request_id
        self._before = before
        self._endpoint = endpoint
        self._headers = headers
        self._usage = None
        self._fired = False

    async def _fire_after(self) -> None:
        if self._fired:
            return
        self._fired = True
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        await post_async(
            self._endpoint,
            {
                "event_id": self._request_id,
                "event_type": "chat.completions.create.after",
                **self._before,
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": getattr(self._usage, "prompt_tokens", None),
                "completion_tokens": getattr(self._usage, "completion_tokens", None),
            },
            self._headers,
        )

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                if getattr(chunk, "usage", None) is not None:
                    self._usage = chunk.usage
                yield chunk
        finally:
            await self._fire_after()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        result = await self._stream.__aexit__(*args)
        await self._fire_after()
        return result

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Chat completions (existing)
# ---------------------------------------------------------------------------


class Completions(_Completions):
    """Instrumented synchronous chat completions client.

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous chat completion request with telemetry.

        Supports both regular and streaming responses (``stream=True``).
        When streaming, ``stream_options={"include_usage": True}`` is injected
        automatically so token counts are available on the final chunk.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
            messages (list[dict]): Conversation history in OpenAI message format.
            stream (bool, optional): If ``True``, returns a :class:`_TrackedSyncStream`
                that yields ``ChatCompletionChunk`` objects and fires telemetry on completion.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``chat.completions.create()``.

        Returns:
            ``ChatCompletion`` when ``stream=False`` (default), or a
            :class:`_TrackedSyncStream` when ``stream=True``.
        """
        tags = kwargs.pop("tags", {})
        before = {"provider": _PROVIDER, "model": kwargs.get("model"), "message_count": len(kwargs.get("messages", [])), "tags": tags}

        if not kwargs.get("stream"):
            return track_sync(
                url=self._intake_url,
                call="chat.completions.create",
                before=before,
                fn=lambda: super(Completions, self).create(**kwargs),
                after_extra=_usage,
                client_id=self._client_id,
                bearer_token=self._bearer_token,
            )

        # Streaming path — inject include_usage so the final chunk carries token counts.
        kwargs.setdefault("stream_options", {})
        kwargs["stream_options"]["include_usage"] = True

        endpoint = _build_url(self._intake_url, self._client_id)
        headers = _auth_headers(self._bearer_token)
        request_id = str(uuid.uuid4())

        post_sync(endpoint, {"event_id": request_id, "event_type": "chat.completions.create.before", **before}, headers)

        start = time.perf_counter()
        try:
            raw_stream = super().create(**kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            post_sync(endpoint, {"event_id": request_id, "event_type": "chat.completions.create.error",
                                 **before, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, headers)
            raise

        return _TrackedSyncStream(
            raw_stream,
            start=start,
            request_id=request_id,
            before=before,
            endpoint=endpoint,
            headers=headers,
        )


class AsyncCompletions(_AsyncCompletions):
    """Instrumented async chat completions client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async chat completion request with telemetry.

        Supports both regular and streaming responses (``stream=True``).
        When streaming, ``stream_options={"include_usage": True}`` is injected
        automatically so token counts are available on the final chunk.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
            messages (list[dict]): Conversation history in OpenAI message format.
            stream (bool, optional): If ``True``, returns a :class:`_TrackedAsyncStream`
                that yields ``ChatCompletionChunk`` objects and fires telemetry on completion.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``chat.completions.create()``.

        Returns:
            ``ChatCompletion`` when ``stream=False`` (default), or a
            :class:`_TrackedAsyncStream` when ``stream=True``.
        """
        tags = kwargs.pop("tags", {})
        before = {"provider": _PROVIDER, "model": kwargs.get("model"), "message_count": len(kwargs.get("messages", [])), "tags": tags}

        if not kwargs.get("stream"):
            return await track_async(
                url=self._intake_url,
                call="chat.completions.create_async",
                before=before,
                fn=lambda: super(AsyncCompletions, self).create(**kwargs),
                after_extra=_usage,
                client_id=self._client_id,
                bearer_token=self._bearer_token,
            )

        # Streaming path — inject include_usage so the final chunk carries token counts.
        kwargs.setdefault("stream_options", {})
        kwargs["stream_options"]["include_usage"] = True

        endpoint = _build_url(self._intake_url, self._client_id)
        headers = _auth_headers(self._bearer_token)
        request_id = str(uuid.uuid4())

        await post_async(endpoint, {"event_id": request_id, "event_type": "chat.completions.create.before", **before}, headers)

        start = time.perf_counter()
        try:
            raw_stream = await super().create(**kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            await post_async(endpoint, {"event_id": request_id, "event_type": "chat.completions.create.error",
                                        **before, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, headers)
            raise

        return _TrackedAsyncStream(
            raw_stream,
            start=start,
            request_id=request_id,
            before=before,
            endpoint=endpoint,
            headers=headers,
        )


# ---------------------------------------------------------------------------
# Legacy text completions  (client.completions)
# ---------------------------------------------------------------------------


class TextCompletions(_TextCompletions):
    """Instrumented synchronous legacy text completions client.

    Wraps ``openai.resources.completions.Completions`` (the ``/v1/completions``
    endpoint, e.g. ``gpt-3.5-turbo-instruct``).

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous legacy text completion request with telemetry.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-3.5-turbo-instruct"``).
            prompt (str | list, optional): The prompt(s) to complete.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``completions.create()``.

        Returns:
            ``Completion``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        prompt = kwargs.get("prompt", "")
        prompt_length = len(prompt) if isinstance(prompt, (str, list)) else 0
        return track_sync(
            url=self._intake_url,
            call="completions.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "prompt_length": prompt_length, "tags": tags},
            fn=lambda: super(TextCompletions, self).create(**kwargs),
            after_extra=_text_completion_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncTextCompletions(_AsyncTextCompletions):
    """Instrumented async legacy text completions client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async legacy text completion request with telemetry.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-3.5-turbo-instruct"``).
            prompt (str | list, optional): The prompt(s) to complete.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``completions.create()``.

        Returns:
            ``Completion``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        prompt = kwargs.get("prompt", "")
        prompt_length = len(prompt) if isinstance(prompt, (str, list)) else 0
        return await track_async(
            url=self._intake_url,
            call="completions.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "prompt_length": prompt_length, "tags": tags},
            fn=lambda: super(AsyncTextCompletions, self).create(**kwargs),
            after_extra=_text_completion_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Embeddings  (client.embeddings)
# ---------------------------------------------------------------------------


class Embeddings(_Embeddings):
    """Instrumented synchronous embeddings client.

    Wraps ``openai.resources.embeddings.Embeddings`` (``/v1/embeddings``).

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous embeddings request with telemetry.

        Args:
            model (str): Embedding model identifier (e.g. ``"text-embedding-3-small"``).
            input (str | list[str]): Text(s) to embed.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``embeddings.create()``.

        Returns:
            ``CreateEmbeddingResponse``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        raw_input = kwargs.get("input", [])
        input_count = len(raw_input) if isinstance(raw_input, list) else 1
        return track_sync(
            url=self._intake_url,
            call="embeddings.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "input_count": input_count, "tags": tags},
            fn=lambda: super(Embeddings, self).create(**kwargs),
            after_extra=_embedding_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncEmbeddings(_AsyncEmbeddings):
    """Instrumented async embeddings client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async embeddings request with telemetry.

        Args:
            model (str): Embedding model identifier (e.g. ``"text-embedding-3-small"``).
            input (str | list[str]): Text(s) to embed.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``embeddings.create()``.

        Returns:
            ``CreateEmbeddingResponse``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        raw_input = kwargs.get("input", [])
        input_count = len(raw_input) if isinstance(raw_input, list) else 1
        return await track_async(
            url=self._intake_url,
            call="embeddings.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "input_count": input_count, "tags": tags},
            fn=lambda: super(AsyncEmbeddings, self).create(**kwargs),
            after_extra=_embedding_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Responses API  (client.responses)
# ---------------------------------------------------------------------------


class Responses(_Responses):
    """Instrumented synchronous Responses API client.

    Wraps ``openai.resources.responses.responses.Responses`` (``/v1/responses``).
    The Responses API is the newer stateful alternative to chat completions,
    supporting tool calls, built-in tools, and multi-turn conversations.

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous Responses API request with telemetry.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-4o"``).
            input (str | list): The input message(s) or conversation.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``responses.create()``.

        Returns:
            ``Response``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        raw_input = kwargs.get("input", [])
        input_count = len(raw_input) if isinstance(raw_input, list) else 1
        return track_sync(
            url=self._intake_url,
            call="responses.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "input_count": input_count, "tags": tags},
            fn=lambda: super(Responses, self).create(**kwargs),
            after_extra=_response_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncResponses(_AsyncResponses):
    """Instrumented async Responses API client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async Responses API request with telemetry.

        Args:
            model (str): OpenAI model identifier (e.g. ``"gpt-4o"``).
            input (str | list): The input message(s) or conversation.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``responses.create()``.

        Returns:
            ``Response``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        raw_input = kwargs.get("input", [])
        input_count = len(raw_input) if isinstance(raw_input, list) else 1
        return await track_async(
            url=self._intake_url,
            call="responses.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "input_count": input_count, "tags": tags},
            fn=lambda: super(AsyncResponses, self).create(**kwargs),
            after_extra=_response_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Text-to-Speech  (client.audio.speech)
# ---------------------------------------------------------------------------


def _tts_usage(_response) -> dict:
    """Extract character count from a TTS response.

    TTS is billed by character count, not tokens. The response is raw audio
    bytes — there is no usage object — so we return an empty dict here.
    Character count is captured in the *before* payload from the request input.

    Args:
        response: An ``HttpxBinaryResponseContent`` returned by the OpenAI SDK.

    Returns:
        An empty dict (no server-side usage metadata available).
    """
    return {}


class Speech(_Speech):
    """Instrumented synchronous Text-to-Speech client.

    Wraps ``openai.resources.audio.speech.Speech`` (``/v1/audio/speech``).
    Billing is by **character count** in the input text, not tokens.

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous TTS request with telemetry.

        Args:
            model (str): TTS model (e.g. ``"tts-1"``, ``"tts-1-hd"``, ``"gpt-4o-mini-tts"``).
            voice (str): Voice identifier (e.g. ``"alloy"``, ``"nova"``).
            input (str): The text to synthesise. Max 4096 characters.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.speech.create()``.

        Returns:
            ``HttpxBinaryResponseContent``: The raw audio response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return track_sync(
            url=self._intake_url,
            call="audio.speech.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "voice": kwargs.get("voice"), "tags": tags},
            fn=lambda: super(Speech, self).create(**kwargs),
            after_extra=_tts_usage,
            input_text=kwargs.get("input"),
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncSpeech(_AsyncSpeech):
    """Instrumented async Text-to-Speech client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async TTS request with telemetry.

        Args:
            model (str): TTS model (e.g. ``"tts-1"``, ``"tts-1-hd"``, ``"gpt-4o-mini-tts"``).
            voice (str): Voice identifier (e.g. ``"alloy"``, ``"nova"``).
            input (str): The text to synthesise. Max 4096 characters.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.speech.create()``.

        Returns:
            ``HttpxBinaryResponseContent``: The raw audio response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return await track_async(
            url=self._intake_url,
            call="audio.speech.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "voice": kwargs.get("voice"), "tags": tags},
            fn=lambda: super(AsyncSpeech, self).create(**kwargs),
            after_extra=_tts_usage,
            input_text=kwargs.get("input"),
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Transcriptions / Speech-to-Text  (client.audio.transcriptions)
# ---------------------------------------------------------------------------


def _stt_usage(response) -> dict:
    """Extract usage from a transcription or translation response.

    Newer models (``gpt-4o-transcribe``, ``gpt-4o-mini-transcribe``) return
    token-based usage (``type="tokens"``).  ``whisper-1`` returns duration-based
    usage (``type="duration"``, measured in seconds).  Both variants are
    normalised here so the telemetry payload is consistent.

    Args:
        response: A ``Transcription`` or ``Translation`` returned by the OpenAI SDK.

    Returns:
        A dict with ``usage_type``, and either ``input_tokens`` + ``audio_tokens``
        (tokens variant) or ``duration_seconds`` (duration variant).
        All fields are ``None`` when usage is absent.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"usage_type": None, "input_tokens": None, "audio_tokens": None, "duration_seconds": None}

    usage_type = getattr(usage, "type", None)
    if usage_type == "tokens":
        details = getattr(usage, "input_token_details", None)
        return {
            "usage_type": "tokens",
            "input_tokens": getattr(usage, "input_tokens", None),
            "audio_tokens": getattr(details, "audio_tokens", None),
            "duration_seconds": None,
        }
    if usage_type == "duration":
        return {
            "usage_type": "duration",
            "input_tokens": None,
            "audio_tokens": None,
            "duration_seconds": getattr(usage, "seconds", None),
        }
    return {"usage_type": None, "input_tokens": None, "audio_tokens": None, "duration_seconds": None}


class Transcriptions(_Transcriptions):
    """Instrumented synchronous Speech-to-Text client.

    Wraps ``openai.resources.audio.transcriptions.Transcriptions``
    (``/v1/audio/transcriptions``).

    Billing depends on the model:

    - ``whisper-1`` — billed by audio duration (seconds)
    - ``gpt-4o-transcribe`` / ``gpt-4o-mini-transcribe`` — billed by tokens

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous transcription request with telemetry.

        Args:
            model (str): Model to use (e.g. ``"whisper-1"``, ``"gpt-4o-mini-transcribe"``).
            file: Audio file object to transcribe.
            language (str, optional): ISO-639-1 language code (e.g. ``"en"``).
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.transcriptions.create()``.

        Returns:
            ``Transcription``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return track_sync(
            url=self._intake_url,
            call="audio.transcriptions.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "language": kwargs.get("language"), "tags": tags},
            fn=lambda: super(Transcriptions, self).create(**kwargs),
            after_extra=_stt_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncTranscriptions(_AsyncTranscriptions):
    """Instrumented async Speech-to-Text client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async transcription request with telemetry.

        Args:
            model (str): Model to use (e.g. ``"whisper-1"``, ``"gpt-4o-mini-transcribe"``).
            file: Audio file object to transcribe.
            language (str, optional): ISO-639-1 language code (e.g. ``"en"``).
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.transcriptions.create()``.

        Returns:
            ``Transcription``: The upstream OpenAI response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return await track_async(
            url=self._intake_url,
            call="audio.transcriptions.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "language": kwargs.get("language"), "tags": tags},
            fn=lambda: super(AsyncTranscriptions, self).create(**kwargs),
            after_extra=_stt_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Translations  (client.audio.translations)
# ---------------------------------------------------------------------------


def _translation_verbose_usage(verbose_response: _TranslationVerbose) -> dict:
    """Extract duration and token count from a verbose translation response.

    Args:
        verbose_response: A ``TranslationVerbose`` returned when
            ``response_format="verbose_json"`` is used.

    Returns:
        A dict with ``duration_seconds`` (audio length) and
        ``completion_tokens`` (sum of token counts across all segments).
    """
    segments = getattr(verbose_response, "segments", None) or []
    completion_tokens = sum(len(getattr(seg, "tokens", [])) for seg in segments)
    return {
        "duration_seconds": getattr(verbose_response, "duration", None),
        "completion_tokens": completion_tokens,
    }


def _convert_translation_response(verbose_response: _TranslationVerbose, original_format: str | None) -> Any:
    """Convert a verbose translation response back to the originally requested format.

    Args:
        verbose_response: The ``TranslationVerbose`` fetched internally.
        original_format: The ``response_format`` value the caller originally passed
            (``None`` / ``"json"`` / ``"verbose_json"`` / ``"text"``).

    Returns:
        The response in the format the caller expected.
    """
    if original_format in (None, "json"):
        return _Translation(text=verbose_response.text)
    if original_format == "verbose_json":
        return verbose_response
    if original_format == "text":
        return verbose_response.text
    # TODO: reconstruct "srt" and "vtt" from verbose_response.segments timestamps
    raise NotImplementedError(f"response_format={original_format!r} is not yet supported with Weflayr telemetry")


class Translations(_Translations):
    """Instrumented synchronous audio translation client.

    Wraps ``openai.resources.audio.translations.Translations``
    (``/v1/audio/translations``).  Translates audio into English.
    Only ``whisper-1`` is supported — billed by audio duration (seconds).

    Args:
        client: The parent ``openai.OpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _OpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def create(self, **kwargs: Any):
        """Send a synchronous translation request with telemetry.

        Internally forces ``response_format="verbose_json"`` to capture
        ``duration_seconds`` and ``completion_tokens`` for telemetry, then
        converts the response back to the format originally requested by the
        caller so the return value is unchanged from the upstream API.

        Args:
            model (str): Must be ``"whisper-1"``.
            file: Audio file object in a non-English language.
            response_format (str, optional): ``"json"`` (default), ``"text"``,
                or ``"verbose_json"``. ``"srt"`` and ``"vtt"`` are not yet supported.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.translations.create()``.

        Returns:
            The response in the originally requested format.
        """
        tags = kwargs.pop("tags", {})
        original_format = kwargs.get("response_format")
        kwargs["response_format"] = "verbose_json"
        return track_sync(
            url=self._intake_url,
            call="audio.translations.create",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "tags": tags},
            fn=lambda: super(Translations, self).create(**kwargs),
            after_extra=_translation_verbose_usage,
            response_transform=lambda r: _convert_translation_response(r, original_format),
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class AsyncTranslations(_AsyncTranslations):
    """Instrumented async audio translation client.

    Args:
        client: The parent ``openai.AsyncOpenAI`` instance.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
    """

    def __init__(
        self,
        client: _AsyncOpenAI,
        *,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
    ) -> None:
        super().__init__(client)
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    async def create(self, **kwargs: Any):
        """Send an async translation request with telemetry.

        Internally forces ``response_format="verbose_json"`` to capture
        ``duration_seconds`` and ``completion_tokens`` for telemetry, then
        converts the response back to the format originally requested by the
        caller so the return value is unchanged from the upstream API.

        Args:
            model (str): Must be ``"whisper-1"``.
            file: Audio file object in a non-English language.
            response_format (str, optional): ``"json"`` (default), ``"text"``,
                or ``"verbose_json"``. ``"srt"`` and ``"vtt"`` are not yet supported.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``openai`` ``audio.translations.create()``.

        Returns:
            The response in the originally requested format.
        """
        tags = kwargs.pop("tags", {})
        original_format = kwargs.get("response_format")
        kwargs["response_format"] = "verbose_json"
        return await track_async(
            url=self._intake_url,
            call="audio.translations.create_async",
            before={"provider": _PROVIDER, "model": kwargs.get("model"), "tags": tags},
            fn=lambda: super(AsyncTranslations, self).create(**kwargs),
            after_extra=_translation_verbose_usage,
            response_transform=lambda r: _convert_translation_response(r, original_format),
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


# ---------------------------------------------------------------------------
# Top-level client wrappers
# ---------------------------------------------------------------------------


class OpenAI(_OpenAI):
    """Drop-in replacement for ``openai.OpenAI`` with Weflayr telemetry.

    Instruments all token-consuming endpoints:
    - ``client.chat.completions.create``
    - ``client.completions.create``  (legacy text completions)
    - ``client.embeddings.create``
    - ``client.responses.create``    (Responses API)

    Args:
        api_key (str): Your OpenAI API key.
        intake_url (str, optional): Weflayr intake API base URL.
        client_id (str, optional): Client identifier sent in the endpoint path.
        bearer_token (str, optional): Bearer token for the Authorization header.
        **kwargs: Forwarded unchanged to ``openai.OpenAI``.

    Example::

        client = OpenAI(api_key="sk-...")
        client.chat.completions.create(model="gpt-4o-mini", messages=[...])
        client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Hello")
        client.embeddings.create(model="text-embedding-3-small", input="Hello")
        client.responses.create(model="gpt-4o", input="Hello")
    """

    def __init__(
        self,
        *args: Any,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        kw = _weflayr_kwargs(intake_url, client_id, bearer_token)
        self.chat.completions = Completions(self, **kw)
        self.completions = TextCompletions(self, **kw)
        self.embeddings = Embeddings(self, **kw)
        self.responses = Responses(self, **kw)
        self.audio.speech = Speech(self, **kw)
        self.audio.transcriptions = Transcriptions(self, **kw)
        self.audio.translations = Translations(self, **kw)


class AsyncOpenAI(_AsyncOpenAI):
    """Drop-in replacement for ``openai.AsyncOpenAI`` with Weflayr telemetry.

    Instruments all token-consuming endpoints:
    - ``client.chat.completions.create``
    - ``client.completions.create``  (legacy text completions)
    - ``client.embeddings.create``
    - ``client.responses.create``    (Responses API)

    Args:
        api_key (str): Your OpenAI API key.
        intake_url (str, optional): Weflayr intake API base URL.
        client_id (str, optional): Client identifier sent in the endpoint path.
        bearer_token (str, optional): Bearer token for the Authorization header.
        **kwargs: Forwarded unchanged to ``openai.AsyncOpenAI``.

    Example::

        client = AsyncOpenAI(api_key="sk-...")
        await client.chat.completions.create(model="gpt-4o-mini", messages=[...])
        await client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Hello")
        await client.embeddings.create(model="text-embedding-3-small", input="Hello")
        await client.responses.create(model="gpt-4o", input="Hello")
    """

    def __init__(
        self,
        *args: Any,
        intake_url: str = INTAKE_URL,
        client_id: str = CLIENT_ID,
        bearer_token: str = CLIENT_SECRET,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        kw = _weflayr_kwargs(intake_url, client_id, bearer_token)
        self.chat.completions = AsyncCompletions(self, **kw)
        self.completions = AsyncTextCompletions(self, **kw)
        self.embeddings = AsyncEmbeddings(self, **kw)
        self.responses = AsyncResponses(self, **kw)
        self.audio.speech = AsyncSpeech(self, **kw)
        self.audio.transcriptions = AsyncTranscriptions(self, **kw)
        self.audio.translations = AsyncTranslations(self, **kw)
