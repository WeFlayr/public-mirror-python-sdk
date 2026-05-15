"""Tests for the Weflayr Python SDK."""

import asyncio
import time
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import weflayr._core as _mod
from weflayr._core import (
    _ObjectProxy,
    _apply_policy,
    _auth_headers,
    _build_url,
    _error_payload,
    _post_async,
    _post_sync,
    _sanitize,
    send_event,
    weflayr_instrument,
    weflayr_setup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**kw) -> dict:
    defaults = dict(
        intake_url="http://intake/",
        client_id="cid",
        client_secret="secret",
        methods=[{"call": "chat.completions.create"}],
    )
    defaults.update(kw)
    return defaults


def _reset():
    _mod._state = None


# ---------------------------------------------------------------------------
# HTTP primitives
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_appends_client_id_and_trailing_slash(self):
        assert _build_url("http://x", "acme") == "http://x/acme/"

    def test_handles_trailing_slash_in_base(self):
        assert _build_url("http://x/", "acme") == "http://x/acme/"


class TestAuthHeaders:
    def test_returns_bearer_header(self):
        assert _auth_headers("my-secret") == {"Authorization": "Bearer my-secret"}


class TestPostSync:
    def test_posts_payload_and_headers(self):
        with patch("weflayr._core.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            _post_sync("http://localhost/", {"k": "v"}, {"Authorization": "Bearer tok"})
            time.sleep(0.05)
            mock_client.post.assert_called_once_with(
                "http://localhost/", json={"k": "v"}, headers={"Authorization": "Bearer tok"}
            )

    def test_defaults_headers_to_empty_dict(self):
        with patch("weflayr._core.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            _post_sync("http://localhost/", {})
            time.sleep(0.05)
            mock_client.post.assert_called_once_with("http://localhost/", json={}, headers={})

    def test_does_not_raise_on_network_error(self):
        with patch("weflayr._core.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("network down")
            mock_cls.return_value.__enter__.return_value = mock_client
            _post_sync("http://localhost/", {})
            time.sleep(0.05)

    def test_returns_immediately(self):
        with patch("weflayr._core.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = lambda *a, **kw: time.sleep(5)
            mock_cls.return_value.__enter__.return_value = mock_client
            start = time.perf_counter()
            _post_sync("http://localhost/", {})
            assert time.perf_counter() - start < 0.5


class TestPostAsync:
    def test_posts_payload_and_headers(self):
        async def _run():
            with patch("weflayr._core.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value.__aenter__.return_value = mock_client
                await _post_async("http://localhost/", {"k": "v"}, {"Authorization": "Bearer tok"})
                mock_client.post.assert_awaited_once_with(
                    "http://localhost/", json={"k": "v"}, headers={"Authorization": "Bearer tok"}
                )
        asyncio.run(_run())

    def test_does_not_raise_on_network_error(self):
        async def _run():
            with patch("weflayr._core.httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.post.side_effect = Exception("down")
                mock_cls.return_value.__aenter__.return_value = mock_client
                await _post_async("http://localhost/", {})
        asyncio.run(_run())


class TestErrorPayload:
    def test_extracts_basic_fields(self):
        assert _error_payload(ValueError("oops")) == {
            "error_type": "ValueError", "error_message": "oops", "status_code": None
        }

    def test_extracts_status_code(self):
        class APIError(Exception):
            status_code = 403
        assert _error_payload(APIError("Forbidden"))["status_code"] == 403


# ---------------------------------------------------------------------------
# weflayr_setup
# ---------------------------------------------------------------------------

class TestWeflayrSetup:
    def setup_method(self):
        _reset()

    def test_sets_state(self):
        s = _settings()
        weflayr_setup(s)
        assert _mod._state is s

    def test_raises_on_missing_intake_url(self):
        with pytest.raises(ValueError):
            weflayr_setup({"intake_url": "", "client_id": "x", "client_secret": "y"})

    def test_raises_on_missing_client_id(self):
        with pytest.raises(ValueError):
            weflayr_setup({"intake_url": "http://x/", "client_id": "", "client_secret": "y"})

    def test_raises_on_missing_client_secret(self):
        with pytest.raises(ValueError):
            weflayr_setup({"intake_url": "http://x/", "client_id": "x", "client_secret": ""})

    def test_disabled_resets_state(self):
        weflayr_setup(_settings())
        weflayr_setup(_settings(enabled=False))
        assert _mod._state is None

    def test_mutually_exclusive_fields_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            weflayr_setup(_settings(ignore_fields=lambda d: d, allow_fields=lambda d: d))
        assert any("mutually exclusive" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# weflayr_instrument
# ---------------------------------------------------------------------------

class TestWeflayrInstrument:
    def setup_method(self):
        _reset()

    def test_returns_target_when_disabled(self):
        obj = object()
        assert weflayr_instrument(obj) is obj

    def test_returns_proxy_when_enabled(self):
        weflayr_setup(_settings())
        assert isinstance(weflayr_instrument(SimpleNamespace()), _ObjectProxy)

    def test_wraps_bare_function_by_name(self):
        weflayr_setup(_settings(methods=[{"call": "my_fn"}]))
        def my_fn(**kw): return 1
        assert weflayr_instrument(my_fn) is not my_fn

    def test_returns_function_unchanged_when_no_match(self):
        weflayr_setup(_settings(methods=[{"call": "other"}]))
        def my_fn(): pass
        assert weflayr_instrument(my_fn) is my_fn


# ---------------------------------------------------------------------------
# send_event
# ---------------------------------------------------------------------------

class TestSendEvent:
    def setup_method(self):
        _reset()

    def test_noop_when_not_setup(self):
        with patch("weflayr._core._post_sync") as mock_post:
            send_event("foo", {})
        mock_post.assert_not_called()

    def test_fires_event_when_setup(self):
        weflayr_setup(_settings())
        with patch("weflayr._core._post_sync") as mock_post:
            send_event("my.event", {"k": "v"})
        payload = mock_post.call_args[0][1]
        assert payload["event_type"] == "my.event"
        assert payload["k"] == "v"
        assert "event_id" in payload

    def test_uses_correct_endpoint(self):
        weflayr_setup(_settings(intake_url="http://host", client_id="mycid"))
        with patch("weflayr._core._post_sync") as mock_post:
            send_event("x", {})
        assert mock_post.call_args[0][0] == "http://host/mycid/"

    def test_uses_bearer_token(self):
        weflayr_setup(_settings(client_secret="tok-abc"))
        with patch("weflayr._core._post_sync") as mock_post:
            send_event("x", {})
        assert mock_post.call_args[0][2] == {"Authorization": "Bearer tok-abc"}


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_replaces_bytes(self):
        assert _sanitize(b"audio") is None

    def test_replaces_bytearray(self):
        assert _sanitize(bytearray(b"x")) is None

    def test_replaces_memoryview(self):
        assert _sanitize(memoryview(b"x")) is None

    def test_recurses_into_dict(self):
        assert _sanitize({"a": 1, "b": b"bin"}) == {"a": 1, "b": None}

    def test_recurses_into_list(self):
        assert _sanitize([1, b"x", "y"]) == [1, None, "y"]

    def test_passthrough_scalars(self):
        assert _sanitize(42) == 42
        assert _sanitize("hello") == "hello"
        assert _sanitize(None) is None


# ---------------------------------------------------------------------------
# _apply_policy
# ---------------------------------------------------------------------------

class TestApplyPolicy:
    def test_passthrough_when_no_policy(self):
        assert _apply_policy({"a": 1}, _settings()) == {"a": 1}

    def test_calls_ignore_fields(self):
        s = _settings(ignore_fields=lambda d: {k: v for k, v in d.items() if k != "secret"})
        assert _apply_policy({"x": 1, "secret": "pw"}, s) == {"x": 1}

    def test_calls_allow_fields(self):
        s = _settings(allow_fields=lambda d: {k: v for k, v in d.items() if k == "x"})
        assert _apply_policy({"x": 1, "y": 2}, s) == {"x": 1}

    def test_returns_none_when_both_set(self):
        s = _settings(ignore_fields=lambda d: d, allow_fields=lambda d: d)
        assert _apply_policy({"a": 1}, s) is None


# ---------------------------------------------------------------------------
# _ObjectProxy
# ---------------------------------------------------------------------------

class TestObjectProxy:
    def setup_method(self):
        _reset()

    def test_intercepts_configured_method(self):
        weflayr_setup(_settings(methods=[{"call": "do_thing"}]))
        target = SimpleNamespace(do_thing=lambda: "real")
        proxy = _ObjectProxy(target, "", _mod._state)
        with patch("weflayr._core._post_sync"):
            assert proxy.do_thing() == "real"

    def test_passes_through_unconfigured_attr(self):
        weflayr_setup(_settings(methods=[{"call": "other"}]))
        target = SimpleNamespace(value=42)
        assert _ObjectProxy(target, "", _mod._state).value == 42

    def test_wraps_nested_dot_path(self):
        weflayr_setup(_settings(methods=[{"call": "chat.completions.create"}]))
        inner = MagicMock()
        inner.create = MagicMock(return_value="resp")
        target = SimpleNamespace(chat=SimpleNamespace(completions=inner))
        proxy = _ObjectProxy(target, "", _mod._state)
        with patch("weflayr._core._post_sync"):
            assert proxy.chat.completions.create(model="gpt-4o", messages=[]) == "resp"

    def test_strips_tags_before_underlying_call(self):
        received = {}
        def fake_fn(**kw): received.update(kw); return "ok"
        weflayr_setup(_settings(methods=[{"call": "do_it"}]))
        proxy = _ObjectProxy(SimpleNamespace(do_it=fake_fn), "", _mod._state)
        with patch("weflayr._core._post_sync"):
            proxy.do_it(x=1, __weflayr_tags={"env": "prod"})
        assert "__weflayr_tags" not in received
        assert received["x"] == 1

    def test_setattr_delegates_to_target(self):
        weflayr_setup(_settings())
        target = SimpleNamespace(val=1)
        _ObjectProxy(target, "", _mod._state).val = 99
        assert target.val == 99


# ---------------------------------------------------------------------------
# Sync wrapper — event lifecycle
# ---------------------------------------------------------------------------

class TestSyncWrapper:
    def setup_method(self):
        _reset()

    def _captured(self):
        events = []
        def cap(url, payload, headers): events.append(payload)
        return events, cap

    def test_fires_before_and_after(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn(model="x")
        assert [e["event_type"] for e in events] == ["before", "after"]

    def test_same_event_id(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn()
        assert events[0]["event_id"] == events[1]["event_id"]

    def test_after_contains_elapsed_ms(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn()
        assert "elapsed_ms" in events[1]

    def test_tags_in_before(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn(__weflayr_tags={"env": "prod"})
        assert events[0]["tags"] == {"env": "prod"}

    def test_after_on_error_with_error_fields(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: (_ for _ in ()).throw(ValueError("bad"))), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            with pytest.raises(ValueError):
                proxy.fn()
        after = events[-1]
        assert after["event_type"] == "after"
        assert after["error_type"] == "ValueError"
        assert after["error_message"] == "bad"

    def test_error_re_raised(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))), "", _mod._state)
        with patch("weflayr._core._post_sync"):
            with pytest.raises(RuntimeError, match="boom"):
                proxy.fn()

    def test_light_mode_skips_before(self):
        weflayr_setup(_settings(event_mode="light", methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn()
        types = [e["event_type"] for e in events]
        assert "before" not in types
        assert "after" in types

    def test_middleware_extra_in_after(self):
        def mw(args, resp): return {} if resp is None else {"tok": 42}
        weflayr_setup(_settings(methods=[{"call": "fn", "middleware": mw}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn()
        assert events[-1]["tok"] == 42

    def test_both_content_policies_block_all_events(self):
        weflayr_setup(_settings(ignore_fields=lambda d: d, allow_fields=lambda d: d,
                                methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        with patch("weflayr._core._post_sync") as mock_post:
            proxy.fn()
        mock_post.assert_not_called()

    def test_ignore_fields_applied(self):
        weflayr_setup(_settings(
            ignore_fields=lambda d: {k: v for k, v in d.items() if k != "secret"},
            methods=[{"call": "fn"}],
        ))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn(secret="pw", x=1)
        assert "secret" not in events[0]
        assert events[0].get("x") == 1

    def test_binary_payload_replaced_with_none(self):
        weflayr_setup(_settings(methods=[{"call": "fn"}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: "ok"), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            proxy.fn(audio=b"\x00\xff")
        assert events[0]["audio"] is None


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class TestAsyncWrapper:
    def setup_method(self):
        _reset()

    def _captured(self):
        events = []
        async def cap(url, payload, headers): events.append(payload)
        return events, cap

    def test_fires_before_and_after(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn"}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value="ok")), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                await proxy.fn(model="x")
            return [e["event_type"] for e in events]
        assert asyncio.run(_run()) == ["before", "after"]

    def test_same_event_id(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn"}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value="ok")), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                await proxy.fn()
            return events
        events = asyncio.run(_run())
        assert events[0]["event_id"] == events[1]["event_id"]

    def test_after_on_exception(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn"}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(side_effect=ValueError("async boom"))), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                with pytest.raises(ValueError):
                    await proxy.fn()
            return events
        events = asyncio.run(_run())
        assert events[-1]["error_type"] == "ValueError"

    def test_light_mode_skips_before(self):
        async def _run():
            weflayr_setup(_settings(event_mode="light", methods=[{"call": "fn"}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value="ok")), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                await proxy.fn()
            return [e["event_type"] for e in events]
        types = asyncio.run(_run())
        assert "before" not in types
        assert "after" in types

    def test_tags_stripped(self):
        async def _run():
            received = {}
            async def fake_fn(**kw): received.update(kw); return "ok"
            weflayr_setup(_settings(methods=[{"call": "fn"}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=fake_fn), "", _mod._state)
            with patch("weflayr._core._post_async", new=AsyncMock()):
                await proxy.fn(x=1, __weflayr_tags={"env": "test"})
            return received
        received = asyncio.run(_run())
        assert "__weflayr_tags" not in received
        assert received["x"] == 1


# ---------------------------------------------------------------------------
# Sync streaming proxy
# ---------------------------------------------------------------------------

class TestSyncStreamProxy:
    def setup_method(self):
        _reset()

    def _captured(self):
        events = []
        def cap(url, payload, headers): events.append(payload)
        return events, cap

    def _acc_factory(self, emit_on: set | None = None, extra: dict | None = None):
        _emit = emit_on or set()
        _extra = extra or {}
        class Acc:
            def __init__(self): self._n = 0
            def on_chunk(self, chunk): self._n += 1; return self._n in _emit
            def finalize(self): return _extra
        return Acc

    def test_fires_stream_start_and_after(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([1, 2])), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            list(proxy.fn())
        types = [e["event_type"] for e in events]
        assert "stream_start" in types
        assert "after" in types

    def test_yields_all_chunks(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([10, 20, 30])), "", _mod._state)
        with patch("weflayr._core._post_sync"):
            assert list(proxy.fn()) == [10, 20, 30]

    def test_stream_pending_when_on_chunk_true(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory(emit_on={2})}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([1, 2, 3])), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            list(proxy.fn())
        assert any(e["event_type"] == "stream_pending" for e in events)

    def test_no_stream_pending_when_never_true(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([1, 2])), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            list(proxy.fn())
        assert not any(e["event_type"] == "stream_pending" for e in events)

    def test_finalize_merged_into_after(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory(extra={"tok": 7})}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([1])), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            list(proxy.fn())
        after = next(e for e in events if e["event_type"] == "after")
        assert after["tok"] == 7

    def test_after_not_fired_twice_with_context_manager(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
        class FakeStream:
            def __iter__(self): return iter([1, 2])
            def __enter__(self): return self
            def __exit__(self, *a): pass
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: FakeStream()), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            stream = proxy.fn()
            with stream:
                list(stream)
        assert sum(1 for e in events if e["event_type"] == "after") == 1

    def test_stream_start_emitted_in_light_mode(self):
        weflayr_setup(_settings(event_mode="light",
                                methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: iter([1])), "", _mod._state)
        events, cap = self._captured()
        with patch("weflayr._core._post_sync", side_effect=cap):
            list(proxy.fn())
        assert any(e["event_type"] == "stream_start" for e in events)

    def test_stream_path_resolves_attribute(self):
        weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory(), "stream_path": "stream"}]))
        response = SimpleNamespace(stream=iter([1, 2, 3]))
        proxy = _ObjectProxy(SimpleNamespace(fn=lambda **kw: response), "", _mod._state)
        with patch("weflayr._core._post_sync"):
            assert list(proxy.fn()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Async streaming proxy
# ---------------------------------------------------------------------------

class TestAsyncStreamProxy:
    def setup_method(self):
        _reset()

    def _captured(self):
        events = []
        async def cap(url, payload, headers): events.append(payload)
        return events, cap

    def _acc_factory(self, emit_on: set | None = None, extra: dict | None = None):
        _emit = emit_on or set()
        _extra = extra or {}
        class Acc:
            def __init__(self): self._n = 0
            def on_chunk(self, chunk): self._n += 1; return self._n in _emit
            def finalize(self): return _extra
        return Acc

    def _agen(self, items):
        async def _g():
            for i in items: yield i
        return _g()

    def test_fires_stream_start_and_after(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value=self._agen([1, 2]))), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                result = await proxy.fn()
                _ = [c async for c in result]
            return [e["event_type"] for e in events]
        types = asyncio.run(_run())
        assert "stream_start" in types
        assert "after" in types

    def test_yields_all_chunks(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value=self._agen([10, 20, 30]))), "", _mod._state)
            with patch("weflayr._core._post_async", new=AsyncMock()):
                result = await proxy.fn()
                return [c async for c in result]
        assert asyncio.run(_run()) == [10, 20, 30]

    def test_stream_pending_on_chunk_true(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory(emit_on={1})}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value=self._agen([1, 2]))), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                result = await proxy.fn()
                _ = [c async for c in result]
            return [e["event_type"] for e in events]
        assert "stream_pending" in asyncio.run(_run())

    def test_finalize_merged_into_after(self):
        async def _run():
            weflayr_setup(_settings(methods=[{"call": "fn", "stream_middleware": self._acc_factory(extra={"tokens": 5})}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value=self._agen([1]))), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                result = await proxy.fn()
                _ = [c async for c in result]
            return events
        events = asyncio.run(_run())
        assert next(e for e in events if e["event_type"] == "after")["tokens"] == 5

    def test_stream_start_emitted_in_light_mode(self):
        async def _run():
            weflayr_setup(_settings(event_mode="light",
                                    methods=[{"call": "fn", "stream_middleware": self._acc_factory()}]))
            proxy = _ObjectProxy(SimpleNamespace(fn=AsyncMock(return_value=self._agen([1]))), "", _mod._state)
            events, cap = self._captured()
            with patch("weflayr._core._post_async", side_effect=cap):
                result = await proxy.fn()
                _ = [c async for c in result]
            return [e["event_type"] for e in events]
        assert "stream_start" in asyncio.run(_run())


# ---------------------------------------------------------------------------
# Bare function wrapping
# ---------------------------------------------------------------------------

class TestBareFunctionWrapping:
    def setup_method(self):
        _reset()

    def test_wraps_matching_function(self):
        weflayr_setup(_settings(methods=[{"call": "my_func"}]))
        def my_func(**kw): return "result"
        wrapped = weflayr_instrument(my_func)
        with patch("weflayr._core._post_sync"):
            assert wrapped() == "result"

    def test_fires_before_and_after(self):
        events = []
        def cap(url, p, h): events.append(p)
        weflayr_setup(_settings(methods=[{"call": "my_func"}]))
        def my_func(**kw): return 42
        wrapped = weflayr_instrument(my_func)
        with patch("weflayr._core._post_sync", side_effect=cap):
            wrapped()
        types = [e["event_type"] for e in events]
        assert "before" in types
        assert "after" in types

    def test_passes_through_unmatched_function(self):
        weflayr_setup(_settings(methods=[{"call": "other"}]))
        def my_func(): return "untouched"
        assert weflayr_instrument(my_func) is my_func
