"""Unit tests for weflayr.sdk.openai.client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weflayr.sdk.openai.client import (
    AsyncCompletions,
    AsyncEmbeddings,
    AsyncOpenAI,
    AsyncResponses,
    AsyncSpeech,
    AsyncTextCompletions,
    AsyncTranscriptions,
    AsyncTranslations,
    Completions,
    Embeddings,
    OpenAI,
    Responses,
    Speech,
    TextCompletions,
    Transcriptions,
    Translations,
    _convert_translation_response,
    _embedding_usage,
    _response_usage,
    _stt_usage,
    _text_completion_usage,
    _translation_verbose_usage,
    _tts_usage,
    _usage,
)


# ---------------------------------------------------------------------------
# _usage  (chat completions)
# ---------------------------------------------------------------------------


class TestUsage:
    def test_extracts_token_counts(self):
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8))
        assert _usage(response) == {"prompt_tokens": 20, "completion_tokens": 8}

    def test_returns_none_when_usage_missing(self):
        assert _usage(SimpleNamespace()) == {"prompt_tokens": None, "completion_tokens": None}

    def test_returns_none_when_token_fields_missing(self):
        assert _usage(SimpleNamespace(usage=SimpleNamespace())) == {"prompt_tokens": None, "completion_tokens": None}


# ---------------------------------------------------------------------------
# _text_completion_usage
# ---------------------------------------------------------------------------


class TestTextCompletionUsage:
    def test_extracts_token_counts(self):
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        assert _text_completion_usage(response) == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_returns_none_when_usage_missing(self):
        assert _text_completion_usage(SimpleNamespace()) == {"prompt_tokens": None, "completion_tokens": None}

    def test_returns_none_when_token_fields_missing(self):
        assert _text_completion_usage(SimpleNamespace(usage=SimpleNamespace())) == {
            "prompt_tokens": None,
            "completion_tokens": None,
        }


# ---------------------------------------------------------------------------
# _embedding_usage
# ---------------------------------------------------------------------------


class TestEmbeddingUsage:
    def test_extracts_prompt_and_total_tokens(self):
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, total_tokens=12))
        assert _embedding_usage(response) == {"prompt_tokens": 12, "total_tokens": 12}

    def test_returns_none_when_usage_missing(self):
        assert _embedding_usage(SimpleNamespace()) == {"prompt_tokens": None, "total_tokens": None}

    def test_returns_none_when_token_fields_missing(self):
        assert _embedding_usage(SimpleNamespace(usage=SimpleNamespace())) == {
            "prompt_tokens": None,
            "total_tokens": None,
        }


# ---------------------------------------------------------------------------
# _response_usage
# ---------------------------------------------------------------------------


class TestResponseUsage:
    def test_extracts_input_output_and_cached_tokens(self):
        details = SimpleNamespace(cached_tokens=4)
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=20, output_tokens=8, input_tokens_details=details))
        assert _response_usage(response) == {"input_tokens": 20, "output_tokens": 8, "cached_tokens": 4}

    def test_returns_none_when_usage_missing(self):
        assert _response_usage(SimpleNamespace()) == {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
        }

    def test_returns_none_when_details_missing(self):
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        result = _response_usage(response)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["cached_tokens"] is None


# ---------------------------------------------------------------------------
# Completions.create (sync chat)
# ---------------------------------------------------------------------------


class TestCompletionsCreate:
    def _make_completions(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Completions.__init__", return_value=None):
            return Completions(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make_completions()
        fake = SimpleNamespace(usage=None)
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="gpt-4o-mini", messages=[]) is fake

    def test_passes_correct_call_name(self):
        c = self._make_completions()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", messages=[])
            assert mock_track.call_args.kwargs["call"] == "chat.completions.create"

    def test_passes_intake_url(self):
        c = self._make_completions(intake_url="http://custom/")
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", messages=[])
            assert mock_track.call_args.kwargs["url"] == "http://custom/"

    def test_before_contains_model_and_message_count(self):
        c = self._make_completions()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])
            before = mock_track.call_args.kwargs["before"]
            assert before["model"] == "gpt-4o"
            assert before["message_count"] == 2

    def test_tags_forwarded_in_before(self):
        c = self._make_completions()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", messages=[], tags={"env": "prod"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"env": "prod"}

    def test_tags_default_to_empty_dict(self):
        c = self._make_completions()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", messages=[])
            assert mock_track.call_args.kwargs["before"]["tags"] == {}

    def test_after_extra_is_usage_extractor(self):
        c = self._make_completions()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", messages=[])
            assert mock_track.call_args.kwargs["after_extra"] is _usage


# ---------------------------------------------------------------------------
# AsyncCompletions.create (async chat)
# ---------------------------------------------------------------------------


class TestAsyncCompletionsCreate:
    def _make_completions(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncCompletions.__init__", return_value=None):
            return AsyncCompletions(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        async def _run():
            c = self._make_completions()
            fake = SimpleNamespace(usage=None)
            with patch("weflayr.sdk.openai.client.track_async", new=AsyncMock(return_value=fake)):
                return await c.create(model="m", messages=[])

        assert asyncio.run(_run()).usage is None

    def test_passes_correct_call_name(self):
        async def _run():
            c = self._make_completions()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", messages=[])
            return mock_track.call_args

        assert asyncio.run(_run()).kwargs["call"] == "chat.completions.create_async"

    def test_before_contains_model_and_message_count(self):
        async def _run():
            c = self._make_completions()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="gpt-4o-mini", messages=[{}, {}, {}])
            return mock_track.call_args.kwargs["before"]

        before = asyncio.run(_run())
        assert before["model"] == "gpt-4o-mini"
        assert before["message_count"] == 3

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make_completions()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", messages=[])
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# TextCompletions.create (sync legacy)
# ---------------------------------------------------------------------------


class TestTextCompletionsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._TextCompletions.__init__", return_value=None):
            return TextCompletions(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = SimpleNamespace(usage=None)
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="gpt-3.5-turbo-instruct", prompt="hi") is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", prompt="hello")
            assert mock_track.call_args.kwargs["call"] == "completions.create"

    def test_before_contains_model_and_prompt_length_string(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", prompt="hello world")
            before = mock_track.call_args.kwargs["before"]
            assert before["model"] == "m"
            assert before["prompt_length"] == len("hello world")

    def test_before_prompt_length_for_list(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", prompt=["a", "b", "c"])
            assert mock_track.call_args.kwargs["before"]["prompt_length"] == 3

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", prompt="x", tags={"k": "v"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"k": "v"}

    def test_after_extra_is_text_completion_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", prompt="x")
            assert mock_track.call_args.kwargs["after_extra"] is _text_completion_usage


# ---------------------------------------------------------------------------
# AsyncTextCompletions.create
# ---------------------------------------------------------------------------


class TestAsyncTextCompletionsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncTextCompletions.__init__", return_value=None):
            return AsyncTextCompletions(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", prompt="x")
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "completions.create_async"

    def test_before_contains_prompt_length(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", prompt="hello")
            return mock_track.call_args.kwargs["before"]

        before = asyncio.run(_run())
        assert before["prompt_length"] == len("hello")

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", prompt="x")
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# Embeddings.create (sync)
# ---------------------------------------------------------------------------


class TestEmbeddingsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Embeddings.__init__", return_value=None):
            return Embeddings(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = SimpleNamespace(usage=None)
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="text-embedding-3-small", input="hi") is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="hello")
            assert mock_track.call_args.kwargs["call"] == "embeddings.create"

    def test_before_input_count_string(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="hello")
            assert mock_track.call_args.kwargs["before"]["input_count"] == 1

    def test_before_input_count_list(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input=["a", "b", "c"])
            assert mock_track.call_args.kwargs["before"]["input_count"] == 3

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="x", tags={"env": "test"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"env": "test"}

    def test_after_extra_is_embedding_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="x")
            assert mock_track.call_args.kwargs["after_extra"] is _embedding_usage


# ---------------------------------------------------------------------------
# AsyncEmbeddings.create
# ---------------------------------------------------------------------------


class TestAsyncEmbeddingsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncEmbeddings.__init__", return_value=None):
            return AsyncEmbeddings(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input="x")
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "embeddings.create_async"

    def test_before_input_count_list(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input=["a", "b"])
            return mock_track.call_args.kwargs["before"]["input_count"]

        assert asyncio.run(_run()) == 2

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input="x")
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# Responses.create (sync)
# ---------------------------------------------------------------------------


class TestResponsesCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Responses.__init__", return_value=None):
            return Responses(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = SimpleNamespace(usage=None)
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="gpt-4o", input="hello") is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="hello")
            assert mock_track.call_args.kwargs["call"] == "responses.create"

    def test_before_input_count_string(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="hello")
            assert mock_track.call_args.kwargs["before"]["input_count"] == 1

    def test_before_input_count_list(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])
            assert mock_track.call_args.kwargs["before"]["input_count"] == 2

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="x", tags={"feature": "chat"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"feature": "chat"}

    def test_after_extra_is_response_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="m", input="x")
            assert mock_track.call_args.kwargs["after_extra"] is _response_usage


# ---------------------------------------------------------------------------
# AsyncResponses.create
# ---------------------------------------------------------------------------


class TestAsyncResponsesCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncResponses.__init__", return_value=None):
            return AsyncResponses(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input="x")
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "responses.create_async"

    def test_before_input_count_list(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input=[{}, {}])
            return mock_track.call_args.kwargs["before"]["input_count"]

        assert asyncio.run(_run()) == 2

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="m", input="x")
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# _tts_usage
# ---------------------------------------------------------------------------


class TestTtsUsage:
    def test_returns_empty_dict_always(self):
        assert _tts_usage(None) == {}
        assert _tts_usage(SimpleNamespace(anything="ignored")) == {}


# ---------------------------------------------------------------------------
# Speech.create (sync)
# ---------------------------------------------------------------------------


class TestSpeechCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Speech.__init__", return_value=None):
            return Speech(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = object()
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="tts-1", voice="alloy", input="Hello") is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1", voice="alloy", input="Hello")
            assert mock_track.call_args.kwargs["call"] == "audio.speech.create"

    def test_before_contains_model_voice_and_char_count(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1-hd", voice="nova", input="Hello world")
            before = mock_track.call_args.kwargs["before"]
            assert before["model"] == "tts-1-hd"
            assert before["voice"] == "nova"
            assert mock_track.call_args.kwargs["input_text"] == "Hello world"

    def test_char_count_empty_input(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1", voice="alloy", input="")
            assert mock_track.call_args.kwargs["input_text"] == ""

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1", voice="alloy", input="hi", tags={"env": "prod"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"env": "prod"}

    def test_tags_default_to_empty_dict(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1", voice="alloy", input="hi")
            assert mock_track.call_args.kwargs["before"]["tags"] == {}

    def test_after_extra_is_tts_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="tts-1", voice="alloy", input="hi")
            assert mock_track.call_args.kwargs["after_extra"] is _tts_usage


# ---------------------------------------------------------------------------
# AsyncSpeech.create
# ---------------------------------------------------------------------------


class TestAsyncSpeechCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncSpeech.__init__", return_value=None):
            return AsyncSpeech(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="tts-1", voice="alloy", input="hi")
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "audio.speech.create_async"

    def test_before_contains_char_count(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="tts-1", voice="alloy", input="Hello!")
            return mock_track.call_args.kwargs

        kwargs = asyncio.run(_run())
        assert kwargs["input_text"] == "Hello!"
        assert kwargs["before"]["voice"] == "alloy"

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="tts-1", voice="alloy", input="hi")
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# _stt_usage
# ---------------------------------------------------------------------------


class TestSttUsage:
    def test_tokens_variant(self):
        details = SimpleNamespace(audio_tokens=50)
        usage = SimpleNamespace(type="tokens", input_tokens=80, input_token_details=details)
        result = _stt_usage(SimpleNamespace(usage=usage))
        assert result == {"usage_type": "tokens", "input_tokens": 80, "audio_tokens": 50, "duration_seconds": None}

    def test_duration_variant(self):
        usage = SimpleNamespace(type="duration", seconds=12.5)
        result = _stt_usage(SimpleNamespace(usage=usage))
        assert result == {"usage_type": "duration", "input_tokens": None, "audio_tokens": None, "duration_seconds": 12.5}

    def test_missing_usage(self):
        result = _stt_usage(SimpleNamespace())
        assert result == {"usage_type": None, "input_tokens": None, "audio_tokens": None, "duration_seconds": None}

    def test_missing_input_token_details(self):
        usage = SimpleNamespace(type="tokens", input_tokens=10)
        result = _stt_usage(SimpleNamespace(usage=usage))
        assert result["audio_tokens"] is None
        assert result["input_tokens"] == 10

    def test_unknown_usage_type(self):
        usage = SimpleNamespace(type="unknown")
        result = _stt_usage(SimpleNamespace(usage=usage))
        assert result == {"usage_type": None, "input_tokens": None, "audio_tokens": None, "duration_seconds": None}


# ---------------------------------------------------------------------------
# Transcriptions.create (sync)
# ---------------------------------------------------------------------------


class TestTranscriptionsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Transcriptions.__init__", return_value=None):
            return Transcriptions(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = SimpleNamespace(usage=None, text="hello")
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="whisper-1", file=MagicMock()) is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["call"] == "audio.transcriptions.create"

    def test_before_contains_model_and_language(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="gpt-4o-mini-transcribe", file=MagicMock(), language="fr")
            before = mock_track.call_args.kwargs["before"]
            assert before["model"] == "gpt-4o-mini-transcribe"
            assert before["language"] == "fr"

    def test_language_defaults_to_none(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["before"]["language"] is None

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock(), tags={"env": "test"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"env": "test"}

    def test_after_extra_is_stt_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["after_extra"] is _stt_usage


# ---------------------------------------------------------------------------
# AsyncTranscriptions.create
# ---------------------------------------------------------------------------


class TestAsyncTranscriptionsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncTranscriptions.__init__", return_value=None):
            return AsyncTranscriptions(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "audio.transcriptions.create_async"

    def test_before_contains_model_and_language(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock(), language="es")
            return mock_track.call_args.kwargs["before"]

        before = asyncio.run(_run())
        assert before["model"] == "whisper-1"
        assert before["language"] == "es"

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}


# ---------------------------------------------------------------------------
# Translations.create (sync)
# ---------------------------------------------------------------------------


class TestTranslationsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._Translations.__init__", return_value=None):
            return Translations(MagicMock(), intake_url=intake_url)

    def test_returns_upstream_response(self):
        c = self._make()
        fake = SimpleNamespace(usage=None, text="hello")
        with patch("weflayr.sdk.openai.client.track_sync", return_value=fake):
            assert c.create(model="whisper-1", file=MagicMock()) is fake

    def test_call_name(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["call"] == "audio.translations.create"

    def test_before_contains_model(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["before"]["model"] == "whisper-1"

    def test_tags_stripped_and_forwarded(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock(), tags={"env": "prod"})
            assert mock_track.call_args.kwargs["before"]["tags"] == {"env": "prod"}

    def test_after_extra_is_translation_verbose_usage(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            assert mock_track.call_args.kwargs["after_extra"] is _translation_verbose_usage

    def test_verbose_json_forced(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client._Translations.create") as mock_super, \
             patch("weflayr.sdk.helpers.post_sync"):
            mock_super.return_value = SimpleNamespace(text="hi", duration=1.0, segments=[])
            c.create(model="whisper-1", file=MagicMock())
            assert mock_super.call_args.kwargs["response_format"] == "verbose_json"

    def test_response_format_not_leaked_to_caller_default(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            transform = mock_track.call_args.kwargs["response_transform"]
            assert transform is not None

    def test_response_transform_converts_json_format(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock())
            transform = mock_track.call_args.kwargs["response_transform"]
        verbose = SimpleNamespace(text="translated text", duration=3.0, segments=[])
        result = transform(verbose)
        assert result.text == "translated text"

    def test_response_transform_returns_text_for_text_format(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock(), response_format="text")
            transform = mock_track.call_args.kwargs["response_transform"]
        verbose = SimpleNamespace(text="translated text", duration=3.0, segments=[])
        assert transform(verbose) == "translated text"

    def test_response_transform_passthrough_for_verbose_json(self):
        c = self._make()
        with patch("weflayr.sdk.openai.client.track_sync") as mock_track:
            c.create(model="whisper-1", file=MagicMock(), response_format="verbose_json")
            transform = mock_track.call_args.kwargs["response_transform"]
        verbose = SimpleNamespace(text="hi", duration=2.0, segments=[])
        assert transform(verbose) is verbose


# ---------------------------------------------------------------------------
# _translation_verbose_usage
# ---------------------------------------------------------------------------


class TestTranslationVerboseUsage:
    def _seg(self, tokens):
        return SimpleNamespace(tokens=tokens)

    def test_extracts_duration(self):
        verbose = SimpleNamespace(duration=7.5, segments=[])
        assert _translation_verbose_usage(verbose)["duration_seconds"] == 7.5

    def test_completion_tokens_sum_across_segments(self):
        verbose = SimpleNamespace(duration=1.0, segments=[
            self._seg([1, 2, 3]),
            self._seg([4, 5]),
        ])
        assert _translation_verbose_usage(verbose)["completion_tokens"] == 5

    def test_completion_tokens_empty_segments(self):
        verbose = SimpleNamespace(duration=1.0, segments=[])
        assert _translation_verbose_usage(verbose)["completion_tokens"] == 0

    def test_none_duration_when_absent(self):
        verbose = SimpleNamespace(segments=[])
        assert _translation_verbose_usage(verbose)["duration_seconds"] is None


# ---------------------------------------------------------------------------
# _convert_translation_response
# ---------------------------------------------------------------------------


class TestConvertTranslationResponse:
    def _verbose(self, text="hello"):
        return SimpleNamespace(text=text, duration=2.0, segments=[])

    def test_none_format_returns_translation_object(self):
        from openai.types.audio import Translation
        result = _convert_translation_response(self._verbose("hi"), None)
        assert isinstance(result, Translation)
        assert result.text == "hi"

    def test_json_format_returns_translation_object(self):
        from openai.types.audio import Translation
        result = _convert_translation_response(self._verbose("hi"), "json")
        assert isinstance(result, Translation)
        assert result.text == "hi"

    def test_verbose_json_format_returns_verbose_unchanged(self):
        verbose = self._verbose("hi")
        assert _convert_translation_response(verbose, "verbose_json") is verbose

    def test_text_format_returns_string(self):
        result = _convert_translation_response(self._verbose("bonjour"), "text")
        assert result == "bonjour"

    def test_srt_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _convert_translation_response(self._verbose(), "srt")

    def test_vtt_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _convert_translation_response(self._verbose(), "vtt")


# ---------------------------------------------------------------------------
# AsyncTranslations.create
# ---------------------------------------------------------------------------


class TestAsyncTranslationsCreate:
    def _make(self, intake_url="http://intake/"):
        with patch("weflayr.sdk.openai.client._AsyncTranslations.__init__", return_value=None):
            return AsyncTranslations(MagicMock(), intake_url=intake_url)

    def test_call_name(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["call"]

        assert asyncio.run(_run()) == "audio.translations.create_async"

    def test_tags_default_to_empty_dict(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["before"]["tags"]

        assert asyncio.run(_run()) == {}

    def test_after_extra_is_translation_verbose_usage(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["after_extra"]

        assert asyncio.run(_run()) is _translation_verbose_usage

    def test_response_transform_provided(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock())
            return mock_track.call_args.kwargs["response_transform"]

        assert asyncio.run(_run()) is not None

    def test_response_transform_converts_text_format(self):
        async def _run():
            c = self._make()
            mock_track = AsyncMock()
            with patch("weflayr.sdk.openai.client.track_async", mock_track):
                await c.create(model="whisper-1", file=MagicMock(), response_format="text")
            return mock_track.call_args.kwargs["response_transform"]

        transform = asyncio.run(_run())
        verbose = SimpleNamespace(text="async result", duration=1.0, segments=[])
        assert transform(verbose) == "async result"


# ---------------------------------------------------------------------------
# OpenAI — resource injection
# ---------------------------------------------------------------------------


class TestOpenAI:
    def test_chat_completions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None):
                client = OpenAI.__new__(OpenAI)
                client.chat = MagicMock()
                OpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.chat.completions, Completions)

    def test_completions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None):
                with patch.object(TextCompletions, "__init__", return_value=None):
                    client = OpenAI.__new__(OpenAI)
                    client.chat = MagicMock()
                    OpenAI.__init__(client, intake_url="http://x/")
                    assert isinstance(client.completions, TextCompletions)

    def test_embeddings_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None):
                with patch.object(TextCompletions, "__init__", return_value=None):
                    with patch.object(Embeddings, "__init__", return_value=None):
                        client = OpenAI.__new__(OpenAI)
                        client.chat = MagicMock()
                        OpenAI.__init__(client, intake_url="http://x/")
                        assert isinstance(client.embeddings, Embeddings)

    def test_responses_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None):
                with patch.object(TextCompletions, "__init__", return_value=None):
                    with patch.object(Embeddings, "__init__", return_value=None):
                        with patch.object(Responses, "__init__", return_value=None):
                            client = OpenAI.__new__(OpenAI)
                            client.chat = MagicMock()
                            OpenAI.__init__(client, intake_url="http://x/")
                            assert isinstance(client.responses, Responses)

    def test_speech_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None):
                with patch.object(TextCompletions, "__init__", return_value=None):
                    with patch.object(Embeddings, "__init__", return_value=None):
                        with patch.object(Responses, "__init__", return_value=None):
                            with patch.object(Speech, "__init__", return_value=None):
                                client = OpenAI.__new__(OpenAI)
                                client.chat = MagicMock()
                                client.audio = MagicMock()
                                OpenAI.__init__(client, intake_url="http://x/")
                                assert isinstance(client.audio.speech, Speech)

    def test_transcriptions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Transcriptions, "__init__", return_value=None):
                client = OpenAI.__new__(OpenAI)
                client.chat = MagicMock()
                client.audio = MagicMock()
                OpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.audio.transcriptions, Transcriptions)

    def test_translations_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Translations, "__init__", return_value=None):
                client = OpenAI.__new__(OpenAI)
                client.chat = MagicMock()
                client.audio = MagicMock()
                OpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.audio.translations, Translations)

    def test_intake_url_forwarded_to_completions(self):
        with patch("weflayr.sdk.openai.client._OpenAI.__init__", return_value=None):
            with patch.object(Completions, "__init__", return_value=None) as mock_init:
                client = OpenAI.__new__(OpenAI)
                client.chat = MagicMock()
                OpenAI.__init__(client, intake_url="http://custom/")
                assert mock_init.call_args.kwargs["intake_url"] == "http://custom/"


# ---------------------------------------------------------------------------
# AsyncOpenAI — resource injection
# ---------------------------------------------------------------------------


class TestAsyncOpenAI:
    def test_chat_completions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None):
                client = AsyncOpenAI.__new__(AsyncOpenAI)
                client.chat = MagicMock()
                AsyncOpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.chat.completions, AsyncCompletions)

    def test_completions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None):
                with patch.object(AsyncTextCompletions, "__init__", return_value=None):
                    client = AsyncOpenAI.__new__(AsyncOpenAI)
                    client.chat = MagicMock()
                    AsyncOpenAI.__init__(client, intake_url="http://x/")
                    assert isinstance(client.completions, AsyncTextCompletions)

    def test_embeddings_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None):
                with patch.object(AsyncTextCompletions, "__init__", return_value=None):
                    with patch.object(AsyncEmbeddings, "__init__", return_value=None):
                        client = AsyncOpenAI.__new__(AsyncOpenAI)
                        client.chat = MagicMock()
                        AsyncOpenAI.__init__(client, intake_url="http://x/")
                        assert isinstance(client.embeddings, AsyncEmbeddings)

    def test_responses_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None):
                with patch.object(AsyncTextCompletions, "__init__", return_value=None):
                    with patch.object(AsyncEmbeddings, "__init__", return_value=None):
                        with patch.object(AsyncResponses, "__init__", return_value=None):
                            client = AsyncOpenAI.__new__(AsyncOpenAI)
                            client.chat = MagicMock()
                            AsyncOpenAI.__init__(client, intake_url="http://x/")
                            assert isinstance(client.responses, AsyncResponses)

    def test_speech_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None):
                with patch.object(AsyncTextCompletions, "__init__", return_value=None):
                    with patch.object(AsyncEmbeddings, "__init__", return_value=None):
                        with patch.object(AsyncResponses, "__init__", return_value=None):
                            with patch.object(AsyncSpeech, "__init__", return_value=None):
                                client = AsyncOpenAI.__new__(AsyncOpenAI)
                                client.chat = MagicMock()
                                client.audio = MagicMock()
                                AsyncOpenAI.__init__(client, intake_url="http://x/")
                                assert isinstance(client.audio.speech, AsyncSpeech)

    def test_transcriptions_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncTranscriptions, "__init__", return_value=None):
                client = AsyncOpenAI.__new__(AsyncOpenAI)
                client.chat = MagicMock()
                client.audio = MagicMock()
                AsyncOpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.audio.transcriptions, AsyncTranscriptions)

    def test_translations_is_wrapped_instance(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncTranslations, "__init__", return_value=None):
                client = AsyncOpenAI.__new__(AsyncOpenAI)
                client.chat = MagicMock()
                client.audio = MagicMock()
                AsyncOpenAI.__init__(client, intake_url="http://x/")
                assert isinstance(client.audio.translations, AsyncTranslations)

    def test_intake_url_forwarded_to_async_completions(self):
        with patch("weflayr.sdk.openai.client._AsyncOpenAI.__init__", return_value=None):
            with patch.object(AsyncCompletions, "__init__", return_value=None) as mock_init:
                client = AsyncOpenAI.__new__(AsyncOpenAI)
                client.chat = MagicMock()
                AsyncOpenAI.__init__(client, intake_url="http://custom/")
                assert mock_init.call_args.kwargs["intake_url"] == "http://custom/"
