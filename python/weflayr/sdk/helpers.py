"""Weflayr SDK — provider-agnostic telemetry helpers.

This module contains the generic building blocks shared by every connector.
Individual connectors (e.g. ``weflayr.sdk.mistralai``) import from here and
only implement the provider-specific bits.

Available helpers
-----------------
- :func:`post_sync`    — fire-and-forget HTTP POST (sync)
- :func:`post_async`   — fire-and-forget HTTP POST (async)
- :func:`track_sync`   — wrap a sync call with before/after/error telemetry
- :func:`track_async`  — wrap an async call with before/after/error telemetry
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

import httpx

INTAKE_URL: str = os.environ.get("WEFLAYR_INTAKE_URL", "http://127.0.0.1:8123")
"""Default intake API URL, overridden by the ``WEFLAYR_INTAKE_URL`` env var."""

CLIENT_ID: str = os.environ.get("WEFLAYR_CLIENT_ID", "unknown_client")
"""Default client ID, overridden by the ``WEFLAYR_CLIENT_ID`` env var."""

CLIENT_SECRET: str = os.environ.get("WEFLAYR_CLIENT_SECRET", "")
"""Default bearer token, overridden by the ``WEFLAYR_CLIENT_SECRET`` env var."""


def _build_url(base_url: str, client_id: str) -> str:
    """Build the intake endpoint URL for a given client.

    Args:
        base_url: Intake API base URL (e.g. ``"http://127.0.0.1:8000"``).
        client_id: Client identifier included in the path.

    Returns:
        ``"{base_url}/{client_id}/"``
    """
    return f"{base_url.rstrip('/')}/{client_id}/"


def _auth_headers(bearer_token: str) -> dict:
    """Build the Authorization header dict.

    Args:
        bearer_token: The bearer token to include.

    Returns:
        ``{"Authorization": "Bearer {bearer_token}"}``
    """
    return {"Authorization": f"Bearer {bearer_token}"}


def post_sync(url: str, payload: dict, headers: dict | None = None) -> None:
    """POST *payload* to *url* in a background daemon thread.

    Returns immediately — never blocks the caller or raises exceptions.

    Args:
        url: HTTP endpoint to POST to.
        payload: JSON-serialisable dict sent as the request body.
        headers: Optional HTTP headers (e.g. Authorization).
    """
    def _send() -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(url, json=payload, headers=headers or {})
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


async def post_async(url: str, payload: dict, headers: dict | None = None) -> None:
    """Await a POST of *payload* to *url*, swallowing all exceptions.

    Args:
        url: HTTP endpoint to POST to.
        payload: JSON-serialisable dict sent as the request body.
        headers: Optional HTTP headers (e.g. Authorization).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload, headers=headers or {})
    except Exception:
        pass


def _error_payload(exc: Exception) -> dict:
    """Extract a provider-agnostic error dict from any exception.

    Args:
        exc: The exception raised by the provider SDK.

    Returns:
        A dict with ``error_type``, ``error_message``, and ``status_code``.
    """
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "status_code": getattr(exc, "status_code", None),
    }


def track_sync(
    url: str,
    call: str,
    before: dict,
    fn,
    after_extra,
    *,
    input_text: str | None = None,
    response_transform=None,
    client_id: str = CLIENT_ID,
    bearer_token: str = CLIENT_SECRET,
) -> Any:
    """Wrap a synchronous call with before/after/error telemetry events.

    Builds ``{url}/{client_id}/`` as the endpoint and sends
    ``Authorization: Bearer {bearer_token}`` on every request.

    Events emitted:

    - ``<call>.before``  — always, before the provider call
    - ``<call>.after``   — on success, with timing and token usage
    - ``<call>.error``   — on failure, with timing and error details;
      the original exception is always re-raised after the event is sent

    Args:
        url: Intake API base URL.
        call: Event name prefix (e.g. ``"chat.complete"``).
        before: Extra fields merged into every emitted payload.
        fn: Zero-argument callable that performs the actual provider call.
        after_extra: Callable ``(response) -> dict`` returning fields to merge
            into the *after* payload (e.g. token counts).
        input_text: Optional raw input string. When provided, ``char_count``
            is computed and merged into every emitted payload automatically.
        response_transform: Optional callable ``(response) -> Any``. When
            provided, ``after_extra`` receives the raw *fn* response (useful
            when *fn* returns a richer format than the caller expects), and
            the transformed value is returned to the caller instead.
        client_id: Client identifier used in the endpoint path.
        bearer_token: Bearer token sent in the Authorization header.

    Returns:
        The value returned by *fn*, or ``response_transform(response)`` if
        *response_transform* is provided.

    Raises:
        Exception: Whatever *fn* raises, unmodified.
    """
    if input_text is not None:
        before = {**before, "char_count": len(input_text)}
    endpoint = _build_url(url, client_id)
    headers = _auth_headers(bearer_token)
    request_id = str(uuid.uuid4())

    post_sync(endpoint, {"event_id": request_id, "event_type": f"{call}.before", **before}, headers)

    start = time.perf_counter()
    try:
        response = fn()
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        post_sync(endpoint, {"event_id": request_id, "event_type": f"{call}.error",
                             **before, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, headers)
        raise

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    post_sync(endpoint, {"event_id": request_id, "event_type": f"{call}.after",
                         **before, "elapsed_ms": elapsed_ms, **after_extra(response)}, headers)
    return response_transform(response) if response_transform is not None else response


async def track_async(
    url: str,
    call: str,
    before: dict,
    fn,
    after_extra,
    *,
    input_text: str | None = None,
    response_transform=None,
    client_id: str = CLIENT_ID,
    bearer_token: str = CLIENT_SECRET,
) -> Any:
    """Wrap an async call with before/after/error telemetry events.

    Async equivalent of :func:`track_sync`.

    Args:
        url: Intake API base URL.
        call: Event name prefix (e.g. ``"chat.complete_async"``).
        before: Extra fields merged into every emitted payload.
        fn: Zero-argument async callable that performs the actual provider call.
        after_extra: Callable ``(response) -> dict`` returning fields to merge
            into the *after* payload (e.g. token counts).
        input_text: Optional raw input string. When provided, ``char_count``
            is computed and merged into every emitted payload automatically.
        response_transform: Optional callable ``(response) -> Any``. When
            provided, ``after_extra`` receives the raw *fn* response (useful
            when *fn* returns a richer format than the caller expects), and
            the transformed value is returned to the caller instead.
        client_id: Client identifier used in the endpoint path.
        bearer_token: Bearer token sent in the Authorization header.

    Returns:
        The value returned by *fn*, or ``response_transform(response)`` if
        *response_transform* is provided.

    Raises:
        Exception: Whatever *fn* raises, unmodified.
    """
    if input_text is not None:
        before = {**before, "char_count": len(input_text)}
    endpoint = _build_url(url, client_id)
    headers = _auth_headers(bearer_token)
    request_id = str(uuid.uuid4())

    await post_async(endpoint, {"event_id": request_id, "event_type": f"{call}.before", **before}, headers)

    start = time.perf_counter()
    try:
        response = await fn()
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        await post_async(endpoint, {"event_id": request_id, "event_type": f"{call}.error",
                                    **before, "elapsed_ms": elapsed_ms, **_error_payload(exc)}, headers)
        raise

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    await post_async(endpoint, {"event_id": request_id, "event_type": f"{call}.after",
                                **before, "elapsed_ms": elapsed_ms, **after_extra(response)}, headers)
    return response_transform(response) if response_transform is not None else response
