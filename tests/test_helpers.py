"""Unit tests for weflayr.sdk.helpers."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weflayr.sdk.helpers import (
    _auth_headers,
    _build_url,
    _error_payload,
    post_async,
    post_sync,
    track_async,
    track_sync,
)

# Capture helper: collects (payload, headers) tuples from post_sync/post_async calls
def _capture_sync():
    events = []
    return events, lambda url, p, h: events.append((p, h))

def _capture_async():
    events = []
    async def _cap(url, p, h): events.append((p, h))
    return events, _cap


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_appends_client_id_and_trailing_slash(self):
        assert _build_url("http://x", "acme") == "http://x/acme/"

    def test_handles_trailing_slash_in_base(self):
        assert _build_url("http://x/", "acme") == "http://x/acme/"


# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    def test_returns_bearer_header(self):
        assert _auth_headers("my-secret") == {"Authorization": "Bearer my-secret"}


# ---------------------------------------------------------------------------
# post_sync
# ---------------------------------------------------------------------------

class TestPostSync:
    def test_posts_payload_and_headers_to_url(self):
        with patch("weflayr.sdk.helpers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client

            post_sync("http://localhost/", {"key": "value"}, {"Authorization": "Bearer tok"})
            time.sleep(0.05)

            mock_client.post.assert_called_once_with(
                "http://localhost/", json={"key": "value"}, headers={"Authorization": "Bearer tok"}
            )

    def test_defaults_headers_to_empty_dict(self):
        with patch("weflayr.sdk.helpers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client

            post_sync("http://localhost/", {})
            time.sleep(0.05)

            mock_client.post.assert_called_once_with("http://localhost/", json={}, headers={})

    def test_does_not_raise_on_network_error(self):
        with patch("weflayr.sdk.helpers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("network down")
            mock_client_cls.return_value.__enter__.return_value = mock_client
            post_sync("http://localhost/", {})
            time.sleep(0.05)

    def test_returns_immediately_without_blocking(self):
        with patch("weflayr.sdk.helpers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = lambda *a, **kw: time.sleep(5)
            mock_client_cls.return_value.__enter__.return_value = mock_client

            start = time.perf_counter()
            post_sync("http://localhost/", {})
            assert time.perf_counter() - start < 0.5


# ---------------------------------------------------------------------------
# post_async
# ---------------------------------------------------------------------------

class TestPostAsync:
    def test_posts_payload_and_headers_to_url(self):
        async def _run():
            with patch("weflayr.sdk.helpers.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value.__aenter__.return_value = mock_client
                await post_async("http://localhost/", {"key": "value"}, {"Authorization": "Bearer tok"})
                mock_client.post.assert_awaited_once_with(
                    "http://localhost/", json={"key": "value"}, headers={"Authorization": "Bearer tok"}
                )
        asyncio.run(_run())

    def test_defaults_headers_to_empty_dict(self):
        async def _run():
            with patch("weflayr.sdk.helpers.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value.__aenter__.return_value = mock_client
                await post_async("http://localhost/", {})
                mock_client.post.assert_awaited_once_with("http://localhost/", json={}, headers={})
        asyncio.run(_run())

    def test_does_not_raise_on_network_error(self):
        async def _run():
            with patch("weflayr.sdk.helpers.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post.side_effect = Exception("network down")
                mock_cls.return_value.__aenter__.return_value = mock_client
                await post_async("http://localhost/", {})
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# track_sync
# ---------------------------------------------------------------------------

class TestTrackSync:
    def test_returns_fn_result(self):
        with patch("weflayr.sdk.helpers.post_sync"):
            assert track_sync("http://x/", "chat", {}, fn=lambda: 42, after_extra=lambda r: {}) == 42

    def test_calls_fn_once(self):
        fn = MagicMock(return_value="resp")
        with patch("weflayr.sdk.helpers.post_sync"):
            track_sync("http://x/", "chat", {}, fn=fn, after_extra=lambda r: {})
        fn.assert_called_once()

    def test_sends_two_events(self):
        with patch("weflayr.sdk.helpers.post_sync") as mock_post:
            track_sync("http://x/", "chat.complete", {}, fn=lambda: None, after_extra=lambda r: {})
        assert mock_post.call_count == 2

    def test_event_types(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat.complete", {}, fn=lambda: None, after_extra=lambda r: {})
        assert events[0][0]["event_type"] == "chat.complete.before"
        assert events[1][0]["event_type"] == "chat.complete.after"

    def test_same_event_id_in_both_events(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {})
        assert events[0][0]["event_id"] == events[1][0]["event_id"]

    def test_event_id_is_valid_uuid(self):
        import uuid
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {})
        uuid.UUID(events[0][0]["event_id"])

    def test_each_call_gets_unique_event_id(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {})
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {})
        assert events[0][0]["event_id"] != events[2][0]["event_id"]

    def test_before_fields_present_in_both_events(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {"model": "m"}, fn=lambda: None, after_extra=lambda r: {})
        assert events[0][0]["model"] == "m"
        assert events[1][0]["model"] == "m"

    def test_after_extra_merged_into_after_event(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None,
                       after_extra=lambda r: {"prompt_tokens": 10, "completion_tokens": 5})
        assert events[1][0]["prompt_tokens"] == 10

    def test_after_event_contains_elapsed_ms(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {})
        assert isinstance(events[1][0]["elapsed_ms"], float)

    def test_url_includes_client_id(self):
        urls = []
        with patch("weflayr.sdk.helpers.post_sync", side_effect=lambda url, p, h: urls.append(url)):
            track_sync("http://x", "chat", {}, fn=lambda: None, after_extra=lambda r: {}, client_id="acme")
        assert all(u == "http://x/acme/" for u in urls)

    def test_bearer_token_in_headers(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            track_sync("http://x/", "chat", {}, fn=lambda: None, after_extra=lambda r: {},
                       bearer_token="secret-123")
        assert all(e[1] == {"Authorization": "Bearer secret-123"} for e in events)

    def test_fn_exception_propagates(self):
        with patch("weflayr.sdk.helpers.post_sync"):
            with pytest.raises(ValueError, match="boom"):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(ValueError("boom")), after_extra=lambda r: {})

    def test_error_event_sent_on_exception(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            with pytest.raises(RuntimeError):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(RuntimeError("fail")), after_extra=lambda r: {})
        assert len(events) == 2
        assert events[1][0]["event_type"] == "chat.error"

    def test_error_event_shares_event_id(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            with pytest.raises(RuntimeError):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(RuntimeError()), after_extra=lambda r: {})
        assert events[0][0]["event_id"] == events[1][0]["event_id"]

    def test_error_event_contains_error_fields(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            with pytest.raises(ValueError):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(ValueError("bad key")), after_extra=lambda r: {})
        assert events[1][0]["error_type"] == "ValueError"
        assert events[1][0]["error_message"] == "bad key"
        assert "elapsed_ms" in events[1][0]

    def test_error_event_includes_status_code_when_present(self):
        class FakeAPIError(Exception):
            status_code = 401

        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            with pytest.raises(FakeAPIError):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(FakeAPIError("Unauthorized")), after_extra=lambda r: {})
        assert events[1][0]["status_code"] == 401

    def test_no_after_event_on_exception(self):
        events, capture = _capture_sync()
        with patch("weflayr.sdk.helpers.post_sync", side_effect=capture):
            with pytest.raises(RuntimeError):
                track_sync("http://x/", "chat", {}, fn=lambda: (_ for _ in ()).throw(RuntimeError()), after_extra=lambda r: {})
        assert all(e[0]["event_type"] != "chat.after" for e in events)


# ---------------------------------------------------------------------------
# track_async
# ---------------------------------------------------------------------------

class TestTrackAsync:
    def test_returns_fn_result(self):
        async def _run():
            with patch("weflayr.sdk.helpers.post_async", new=AsyncMock()):
                return await track_async("http://x/", "chat", {}, fn=AsyncMock(return_value=42), after_extra=lambda r: {})
        assert asyncio.run(_run()) == 42

    def test_sends_two_events(self):
        async def _run():
            mock_post = AsyncMock()
            with patch("weflayr.sdk.helpers.post_async", mock_post):
                await track_async("http://x/", "chat", {}, fn=AsyncMock(return_value=None), after_extra=lambda r: {})
            return mock_post.await_count
        assert asyncio.run(_run()) == 2

    def test_event_types(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                await track_async("http://x/", "chat.complete_async", {}, fn=AsyncMock(return_value=None), after_extra=lambda r: {})
            return events
        events = asyncio.run(_run())
        assert events[0][0]["event_type"] == "chat.complete_async.before"
        assert events[1][0]["event_type"] == "chat.complete_async.after"

    def test_same_event_id_in_both_events(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                await track_async("http://x/", "chat", {}, fn=AsyncMock(return_value=None), after_extra=lambda r: {})
            return events
        events = asyncio.run(_run())
        assert events[0][0]["event_id"] == events[1][0]["event_id"]

    def test_after_event_contains_elapsed_ms(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                await track_async("http://x/", "chat", {}, fn=AsyncMock(return_value=None), after_extra=lambda r: {})
            return events
        events = asyncio.run(_run())
        assert "elapsed_ms" in events[1][0]

    def test_url_includes_client_id(self):
        async def _run():
            urls = []
            async def cap(url, p, h): urls.append(url)
            with patch("weflayr.sdk.helpers.post_async", side_effect=cap):
                await track_async("http://x", "chat", {}, fn=AsyncMock(return_value=None),
                                  after_extra=lambda r: {}, client_id="acme")
            return urls
        urls = asyncio.run(_run())
        assert all(u == "http://x/acme/" for u in urls)

    def test_bearer_token_in_headers(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                await track_async("http://x/", "chat", {}, fn=AsyncMock(return_value=None),
                                  after_extra=lambda r: {}, bearer_token="tok-456")
            return events
        events = asyncio.run(_run())
        assert all(e[1] == {"Authorization": "Bearer tok-456"} for e in events)

    def test_error_event_sent_on_exception(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                try:
                    await track_async("http://x/", "chat", {}, fn=AsyncMock(side_effect=RuntimeError("fail")), after_extra=lambda r: {})
                except RuntimeError:
                    pass
            return events
        events = asyncio.run(_run())
        assert len(events) == 2
        assert events[1][0]["event_type"] == "chat.error"

    def test_error_event_shares_event_id_async(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                try:
                    await track_async("http://x/", "chat", {}, fn=AsyncMock(side_effect=RuntimeError()), after_extra=lambda r: {})
                except RuntimeError:
                    pass
            return events
        events = asyncio.run(_run())
        assert events[0][0]["event_id"] == events[1][0]["event_id"]

    def test_error_event_contains_error_fields_async(self):
        async def _run():
            events, capture = _capture_async()
            with patch("weflayr.sdk.helpers.post_async", side_effect=capture):
                try:
                    await track_async("http://x/", "chat", {}, fn=AsyncMock(side_effect=ValueError("bad async key")), after_extra=lambda r: {})
                except ValueError:
                    pass
            return events
        events = asyncio.run(_run())
        assert events[1][0]["error_type"] == "ValueError"
        assert events[1][0]["error_message"] == "bad async key"
        assert "elapsed_ms" in events[1][0]


# ---------------------------------------------------------------------------
# _error_payload
# ---------------------------------------------------------------------------

class TestErrorPayload:
    def test_extracts_basic_fields(self):
        result = _error_payload(ValueError("something went wrong"))
        assert result == {"error_type": "ValueError", "error_message": "something went wrong", "status_code": None}

    def test_extracts_status_code(self):
        class APIError(Exception):
            status_code = 403
        assert _error_payload(APIError("Forbidden"))["status_code"] == 403
