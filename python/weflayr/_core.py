"""Weflayr SDK — full implementation.

Public surface (re-exported from the top-level ``weflayr`` package)
-------------------------------------------------------------------
- :func:`weflayr_setup`      — configure credentials and options
- :func:`weflayr_instrument` — wrap any LLM client object or bare function
- :func:`send_event`         — fire an arbitrary event to the intake API

Usage::

    from weflayr import weflayr_setup, weflayr_instrument

    weflayr_setup({
        "intake_url": "https://api.weflayr.com",
        "client_id": "your-client-id",
        "client_secret": "your-client-secret",
        "event_mode": "default",
        "allow_fields": lambda data: {
            "model": data.get("model"),
            "usage": data.get("usage"),
        },
        "methods": [
            {"call": "chat.completions.create"},
        ],
    })

    import openai
    client = weflayr_instrument(openai.OpenAI(api_key="sk-..."))
    client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        __weflayr_tags={"feature": "chat"},
    )
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
import warnings
from typing import Any

import httpx

logger = logging.getLogger("weflayr")

TAGS_KEY = "__weflayr_tags"

# Active settings dict; None means instrumentation is disabled.
_state: dict | None = None


# ---------------------------------------------------------------------------
# HTTP primitives (fire-and-forget)
# ---------------------------------------------------------------------------


def _build_url(base_url: str, client_id: str) -> str:
    return f"{base_url.rstrip('/')}/{client_id}/"


def _auth_headers(bearer_token: str) -> dict:
    return {"Authorization": f"Bearer {bearer_token}"}


def _post_sync(url: str, payload: dict, headers: dict | None = None) -> None:
    """POST *payload* to *url* in a background daemon thread; never blocks."""
    def _send() -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(url, json=payload, headers=headers or {})
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


async def _post_async(url: str, payload: dict, headers: dict | None = None) -> None:
    """POST *payload* to *url* asynchronously; swallows all exceptions."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload, headers=headers or {})
    except Exception:
        pass


def _error_payload(exc: Exception) -> dict:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "status_code": getattr(exc, "status_code", None),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def weflayr_setup(settings: dict) -> None:
    """Configure Weflayr credentials and instrumentation options.

    Must be called before :func:`weflayr_instrument` or :func:`send_event`.
    No-ops (resets to disabled) when ``settings["enabled"]`` is ``False``.

    Args:
        settings: Plain dict with the following keys:

            - ``intake_url`` *(str, required)* — base URL of the intake API
            - ``client_id`` *(str, required)* — UUID of the flayr credential pair
            - ``client_secret`` *(str, required)* — bearer token
            - ``enabled`` *(bool, default True)* — set ``False`` to disable
            - ``event_mode`` *(str, default ``"default"``)* — ``"default"``
              emits ``before`` + ``after``; ``"light"`` skips ``before``
            - ``ignore_fields`` *(callable)* — ``fn(data) -> data``; mutually
              exclusive with ``allow_fields``
            - ``allow_fields`` *(callable)* — ``fn(data) -> data``; mutually
              exclusive with ``ignore_fields``
            - ``methods`` *(list[dict])* — methods to instrument; each dict has:

              - ``call`` *(str, required)* — dot-path (e.g. ``"chat.completions.create"``)
                or bare function name
              - ``middleware`` *(callable)* — ``fn(args, response) -> dict``
              - ``stream_middleware`` *(callable)* — factory returning a
                ``StreamAccumulator`` with ``on_chunk`` and ``finalize``
              - ``stream_path`` *(str)* — attribute on the response holding the stream

    Raises:
        ValueError: If ``intake_url``, ``client_id``, or ``client_secret`` are missing.

    Example::

        weflayr_setup({
            "intake_url": "https://api.weflayr.com",
            "client_id": "...",
            "client_secret": "...",
            "allow_fields": lambda data: {
                "model": data.get("model"),
                "usage": data.get("usage"),
            },
            "methods": [{"call": "chat.completions.create"}],
        })
    """
    global _state
    if not settings.get("intake_url") or not settings.get("client_id") or not settings.get("client_secret"):
        raise ValueError("weflayr_setup: intake_url, client_id, and client_secret are required")
    if not settings.get("enabled", True):
        _state = None
        return
    if settings.get("ignore_fields") and settings.get("allow_fields"):
        warnings.warn(
            "weflayr: ignore_fields and allow_fields are mutually exclusive — no events will be sent",
            stacklevel=2,
        )
    _state = settings


def weflayr_instrument(target: Any) -> Any:
    """Wrap an LLM client object or bare function with Weflayr telemetry.

    Returns *target* unchanged when disabled or no method in ``settings["methods"]``
    matches.  For objects uses a transparent proxy; for bare functions wraps by
    ``__name__``.
    """
    if _state is None:
        return target
    if inspect.isfunction(target) or inspect.ismethod(target):
        return _wrap_bare_function(target, _state)
    return _ObjectProxy(target, "", _state)


def send_event(event_type: str, data: dict) -> None:
    """Fire an arbitrary event to the intake API.

    No-ops when :func:`weflayr_setup` has not been called.  Fire-and-forget —
    failures are swallowed.
    """
    if _state is None:
        return
    _post_sync(
        _build_url(_state["intake_url"], _state["client_id"]),
        {"event_id": str(uuid.uuid4()), "event_type": event_type, **data},
        _auth_headers(_state["client_secret"]),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """Convert an SDK response object to a JSON-serializable dict."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _sanitize(value: Any) -> Any:
    """Recursively replace binary payloads (bytes/bytearray/memoryview) with None."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _apply_policy(data: dict, settings: dict) -> dict | None:
    """Apply content policy; returns None to block the event entirely."""
    if settings.get("ignore_fields") and settings.get("allow_fields"):
        return None
    if settings.get("ignore_fields"):
        return settings["ignore_fields"](data)
    if settings.get("allow_fields"):
        return settings["allow_fields"](data)
    return data


def _fire_sync(endpoint: str, headers: dict, request_id: str, event_type: str,
               data: dict, settings: dict) -> None:
    filtered = _apply_policy(_sanitize(data), settings)
    if filtered is not None:
        _post_sync(endpoint, {"event_id": request_id, "event_type": event_type, **filtered}, headers)


async def _fire_async(endpoint: str, headers: dict, request_id: str, event_type: str,
                      data: dict, settings: dict) -> None:
    filtered = _apply_policy(_sanitize(data), settings)
    if filtered is not None:
        await _post_async(endpoint, {"event_id": request_id, "event_type": event_type, **filtered}, headers)


def _get_methods(settings: dict) -> list[dict]:
    return settings.get("methods") or []


# ---------------------------------------------------------------------------
# Object proxy
# ---------------------------------------------------------------------------


class _ObjectProxy:
    """Transparent proxy that intercepts method calls matching ``settings["methods"]``."""

    def __init__(self, target: Any, prefix: str, settings: dict) -> None:
        object.__setattr__(self, "_pt", target)
        object.__setattr__(self, "_pp", prefix)
        object.__setattr__(self, "_ps", settings)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_pt")
        prefix = object.__getattribute__(self, "_pp")
        settings = object.__getattribute__(self, "_ps")
        attr = getattr(target, name)
        path = f"{prefix}.{name}" if prefix else name

        config = next((m for m in _get_methods(settings) if m["call"] == path), None)
        if config is not None and callable(attr):
            return _make_wrapper(attr, config, path, settings)

        if any(m["call"].startswith(path + ".") for m in _get_methods(settings)):
            return _ObjectProxy(attr, path, settings)

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_pt"), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_pt")(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_pt"))


# ---------------------------------------------------------------------------
# Wrapper factory
# ---------------------------------------------------------------------------


def _make_wrapper(fn: Any, config: dict, call_name: str, settings: dict) -> Any:
    if inspect.iscoroutinefunction(fn):
        return _make_async_wrapper(fn, config, call_name, settings)
    return _make_sync_wrapper(fn, config, call_name, settings)


def _make_sync_wrapper(fn: Any, config: dict, call_name: str, settings: dict):
    middleware = config.get("middleware")
    stream_middleware = config.get("stream_middleware")
    stream_path = config.get("stream_path")

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tags = kwargs.pop(TAGS_KEY, {})
        endpoint = _build_url(settings["intake_url"], settings["client_id"])
        headers = _auth_headers(settings["client_secret"])
        request_id = str(uuid.uuid4())

        before_data: dict = {"tags": tags, "method": call_name, **kwargs}
        if middleware:
            before_data.update(middleware(kwargs, None) or {})

        if settings.get("event_mode") != "light":
            _fire_sync(endpoint, headers, request_id, "before", before_data, settings)

        start = time.perf_counter()
        try:
            response = fn(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            _fire_sync(endpoint, headers, request_id, "after",
                       {**before_data, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, settings)
            raise

        if stream_middleware is not None:
            stream = getattr(response, stream_path) if stream_path else response
            return _SyncStreamProxy(response, stream, config, call_name, start,
                                    request_id, before_data, endpoint, headers, settings)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        after_data: dict = {**before_data, "elapsed_ms": elapsed_ms, "response": _to_plain(response)}
        if middleware:
            after_data.update(middleware(kwargs, response) or {})
        _fire_sync(endpoint, headers, request_id, "after", after_data, settings)
        return response

    return wrapper


def _make_async_wrapper(fn: Any, config: dict, call_name: str, settings: dict):
    middleware = config.get("middleware")
    stream_middleware = config.get("stream_middleware")
    stream_path = config.get("stream_path")

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tags = kwargs.pop(TAGS_KEY, {})
        endpoint = _build_url(settings["intake_url"], settings["client_id"])
        headers = _auth_headers(settings["client_secret"])
        request_id = str(uuid.uuid4())

        before_data: dict = {"tags": tags, "method": call_name, **kwargs}
        if middleware:
            result = middleware(kwargs, None)
            if asyncio.iscoroutine(result):
                result = await result
            before_data.update(result or {})

        if settings.get("event_mode") != "light":
            await _fire_async(endpoint, headers, request_id, "before", before_data, settings)

        start = time.perf_counter()
        try:
            response = await fn(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            await _fire_async(endpoint, headers, request_id, "after",
                              {**before_data, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, settings)
            raise

        if stream_middleware is not None:
            stream = getattr(response, stream_path) if stream_path else response
            return _AsyncStreamProxy(response, stream, config, call_name, start,
                                     request_id, before_data, endpoint, headers, settings)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        after_data: dict = {**before_data, "elapsed_ms": elapsed_ms, "response": _to_plain(response)}
        if middleware:
            result = middleware(kwargs, response)
            if asyncio.iscoroutine(result):
                result = await result
            after_data.update(result or {})
        await _fire_async(endpoint, headers, request_id, "after", after_data, settings)
        return response

    return wrapper


# ---------------------------------------------------------------------------
# Streaming proxies
# ---------------------------------------------------------------------------


class _SyncStreamProxy:
    """Transparent proxy around a synchronous stream with Weflayr telemetry."""

    def __init__(self, response: Any, stream: Any, config: dict,
                 call_name: str, start: float, request_id: str, before_data: dict,
                 endpoint: str, headers: dict, settings: dict) -> None:
        self._response = response
        self._stream = stream
        self._config = config
        self._call_name = call_name
        self._start = start
        self._request_id = request_id
        self._before_data = before_data
        self._endpoint = endpoint
        self._headers = headers
        self._settings = settings
        self._fired = False
        self._accumulator: Any = None

    def _fire_after(self) -> None:
        if self._fired:
            return
        self._fired = True
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        extra = self._accumulator.finalize() if self._accumulator is not None else {}
        _fire_sync(self._endpoint, self._headers, self._request_id, "after",
                   {**self._before_data, "elapsed_ms": elapsed_ms, **extra}, self._settings)

    def __iter__(self):
        self._accumulator = self._config["stream_middleware"]()
        _fire_sync(self._endpoint, self._headers, self._request_id, "stream_start",
                   self._before_data, self._settings)
        try:
            for chunk in self._stream:
                if self._accumulator.on_chunk(chunk):
                    _fire_sync(self._endpoint, self._headers, self._request_id,
                               "stream_pending", self._before_data, self._settings)
                yield chunk
        finally:
            self._fire_after()

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any):
        result = None
        if hasattr(self._stream, "__exit__"):
            result = self._stream.__exit__(*args)
        self._fire_after()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class _AsyncStreamProxy:
    """Transparent proxy around an async stream with Weflayr telemetry."""

    def __init__(self, response: Any, stream: Any, config: dict,
                 call_name: str, start: float, request_id: str, before_data: dict,
                 endpoint: str, headers: dict, settings: dict) -> None:
        self._response = response
        self._stream = stream
        self._config = config
        self._call_name = call_name
        self._start = start
        self._request_id = request_id
        self._before_data = before_data
        self._endpoint = endpoint
        self._headers = headers
        self._settings = settings
        self._fired = False
        self._accumulator: Any = None

    async def _fire_after(self) -> None:
        if self._fired:
            return
        self._fired = True
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        extra = self._accumulator.finalize() if self._accumulator is not None else {}
        await _fire_async(self._endpoint, self._headers, self._request_id, "after",
                          {**self._before_data, "elapsed_ms": elapsed_ms, **extra}, self._settings)

    async def __aiter__(self):
        self._accumulator = self._config["stream_middleware"]()
        await _fire_async(self._endpoint, self._headers, self._request_id, "stream_start",
                          self._before_data, self._settings)
        try:
            async for chunk in self._stream:
                if self._accumulator.on_chunk(chunk):
                    await _fire_async(self._endpoint, self._headers, self._request_id,
                                      "stream_pending", self._before_data, self._settings)
                yield chunk
        finally:
            await self._fire_after()

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any):
        result = None
        if hasattr(self._stream, "__aexit__"):
            result = await self._stream.__aexit__(*args)
        await self._fire_after()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


# ---------------------------------------------------------------------------
# Bare function wrapping
# ---------------------------------------------------------------------------


def _wrap_bare_function(fn: Any, settings: dict) -> Any:
    name = getattr(fn, "__name__", None)
    config = next((m for m in _get_methods(settings) if m["call"] == name), None)
    if config is None:
        return fn
    return _make_wrapper(fn, config, name, settings)
