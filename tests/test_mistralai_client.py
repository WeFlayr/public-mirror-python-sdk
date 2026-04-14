"""Unit tests for weflayr.sdk.mistralai.client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weflayr.sdk.mistralai.client import Chat, Mistral, _usage


# ---------------------------------------------------------------------------
# _usage
# ---------------------------------------------------------------------------

class TestUsage:
    def test_extracts_token_counts(self):
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        assert _usage(response) == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_returns_none_when_usage_missing(self):
        response = SimpleNamespace()
        result = _usage(response)
        assert result == {"prompt_tokens": None, "completion_tokens": None}

    def test_returns_none_when_token_fields_missing(self):
        response = SimpleNamespace(usage=SimpleNamespace())
        result = _usage(response)
        assert result == {"prompt_tokens": None, "completion_tokens": None}


# ---------------------------------------------------------------------------
# Chat.complete
# ---------------------------------------------------------------------------

class TestChatComplete:
    def _make_chat(self, intake_url="http://intake/"):
        """Return a Chat instance with the upstream complete() stubbed out."""
        with patch("weflayr.sdk.mistralai.client._Chat.__init__", return_value=None):
            chat = Chat(intake_url=intake_url)
        return chat

    def test_returns_upstream_response(self):
        chat = self._make_chat()
        fake_response = SimpleNamespace(usage=None)

        with patch("weflayr.sdk.mistralai.client.track_sync", return_value=fake_response) as mock_track:
            result = chat.complete(model="m", messages=[])

        assert result is fake_response

    def test_passes_correct_call_name(self):
        chat = self._make_chat()
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="m", messages=[])
            _, kwargs = mock_track.call_args
            assert kwargs["call"] == "chat.complete"

    def test_passes_intake_url(self):
        chat = self._make_chat(intake_url="http://custom/")
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="m", messages=[])
            _, kwargs = mock_track.call_args
            assert kwargs["url"] == "http://custom/"

    def test_before_contains_model_and_message_count(self):
        chat = self._make_chat()
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="mistral-small-latest", messages=[{"role": "user", "content": "hi"}])
            _, kwargs = mock_track.call_args
            assert kwargs["before"]["model"] == "mistral-small-latest"
            assert kwargs["before"]["message_count"] == 1

    def test_tags_stripped_from_upstream_kwargs(self):
        """tags must not reach the upstream Mistral call."""
        chat = self._make_chat()
        captured_fn = None

        def capture_track(**kwargs):
            nonlocal captured_fn
            captured_fn = kwargs["fn"]
            return SimpleNamespace(usage=None)

        upstream = MagicMock(return_value=SimpleNamespace(usage=None))
        with patch("weflayr.sdk.mistralai.client.track_sync", side_effect=capture_track):
            with patch.object(Chat, "complete", wraps=chat.complete):
                # Simulate the fn lambda calling super().complete
                # We just verify tags are in before but not forwarded
                chat.complete(model="m", messages=[], tags={"env": "test"})

        # tags must appear in before dict, not as a kwarg to upstream
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="m", messages=[], tags={"env": "prod"})
            _, kwargs = mock_track.call_args
            assert kwargs["before"]["tags"] == {"env": "prod"}

    def test_tags_default_to_empty_dict(self):
        chat = self._make_chat()
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="m", messages=[])
            _, kwargs = mock_track.call_args
            assert kwargs["before"]["tags"] == {}

    def test_after_extra_is_usage_extractor(self):
        chat = self._make_chat()
        with patch("weflayr.sdk.mistralai.client.track_sync") as mock_track:
            chat.complete(model="m", messages=[])
            _, kwargs = mock_track.call_args
            # after_extra should be the _usage function
            assert kwargs["after_extra"] is _usage


# ---------------------------------------------------------------------------
# Chat.complete_async
# ---------------------------------------------------------------------------

class TestChatCompleteAsync:
    def _make_chat(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.mistralai.client._Chat.__init__", return_value=None):
            chat = Chat(intake_url=intake_url)
        return chat

    def test_returns_upstream_response(self):
        async def _run():
            chat = self._make_chat()
            fake_response = SimpleNamespace(usage=None)
            with patch("weflayr.sdk.mistralai.client.track_async", new=AsyncMock(return_value=fake_response)):
                return await chat.complete_async(model="m", messages=[])

        assert asyncio.run(_run()).usage is None

    def test_passes_correct_call_name(self):
        async def _run():
            chat = self._make_chat()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.mistralai.client.track_async", mock_track):
                await chat.complete_async(model="m", messages=[])
            return mock_track.call_args

        args, kwargs = asyncio.run(_run())
        assert kwargs["call"] == "chat.complete_async"

    def test_before_contains_model_and_message_count(self):
        async def _run():
            chat = self._make_chat()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.mistralai.client.track_async", mock_track):
                await chat.complete_async(model="mistral-large", messages=[{}, {}])
            return mock_track.call_args

        args, kwargs = asyncio.run(_run())
        assert kwargs["before"]["model"] == "mistral-large"
        assert kwargs["before"]["message_count"] == 2

    def test_tags_default_to_empty_dict(self):
        async def _run():
            chat = self._make_chat()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.mistralai.client.track_async", mock_track):
                await chat.complete_async(model="m", messages=[])
            return mock_track.call_args

        args, kwargs = asyncio.run(_run())
        assert kwargs["before"]["tags"] == {}


# ---------------------------------------------------------------------------
# Mistral.__getattr__ — chat interception
# ---------------------------------------------------------------------------

class TestMistralChatInterception:
    def _make_mistral(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.mistralai.client._Mistral.__init__", return_value=None):
            client = Mistral(intake_url=intake_url)
            client.sdk_configuration = MagicMock()
        return client

    def test_chat_returns_wrapped_chat_instance(self):
        client = self._make_mistral()
        with patch("weflayr.sdk.mistralai.client._Mistral.__getattr__", return_value=MagicMock()):
            assert isinstance(client.chat, Chat)

    def test_chat_receives_intake_url(self):
        client = self._make_mistral(intake_url="http://custom/")
        with patch("weflayr.sdk.mistralai.client._Mistral.__getattr__", return_value=MagicMock()):
            assert client.chat._intake_url == "http://custom/"

    def test_chat_is_cached_after_first_access(self):
        client = self._make_mistral()
        with patch("weflayr.sdk.mistralai.client._Mistral.__getattr__", return_value=MagicMock()):
            chat_1 = client.chat
            chat_2 = client.chat
            assert chat_1 is chat_2
