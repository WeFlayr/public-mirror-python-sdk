"""Weflayr instrumented wrapper for the Mistral AI SDK.

Connector: **Mistral AI**
Provider:  https://mistral.ai

Available connectors
--------------------
- :class:`Mistral` — top-level client (drop-in for ``mistralai.Mistral``)

Available methods
-----------------
Via ``client.chat``:

- :meth:`Chat.complete`       — synchronous chat completion with telemetry
- :meth:`Chat.complete_async` — async chat completion with telemetry

Example::

    from weflayr.sdk.mistralai.client import Mistral

    client = Mistral(api_key="sk-...")
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

from typing import Any

from mistralai.client import Mistral as _Mistral
from mistralai.client.chat import Chat as _Chat

from weflayr.sdk.helpers import CLIENT_ID, CLIENT_SECRET, INTAKE_URL, track_async, track_sync


def _usage(response) -> dict:
    """Extract token usage from a Mistral chat response.

    Args:
        response: A ``ChatCompletionResponse`` returned by the Mistral SDK.

    Returns:
        A dict with ``prompt_tokens`` and ``completion_tokens`` (both ``int | None``).
    """
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


class Chat(_Chat):
    """Instrumented Mistral chat client.

    Args:
        *args: Forwarded to the upstream ``mistralai.client.chat.Chat``.
        intake_url: Weflayr intake API base URL.
        client_id: Client identifier sent in the endpoint path.
        bearer_token: Bearer token for the Authorization header.
        **kwargs: Forwarded to the upstream ``mistralai.client.chat.Chat``.
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
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def complete(self, **kwargs: Any):
        """Send a synchronous chat completion request with telemetry.

        Args:
            model (str): Mistral model identifier (e.g. ``"mistral-small-latest"``).
            messages (list[dict]): Conversation history in OpenAI message format.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``mistralai`` ``chat.complete()``.

        Returns:
            ``ChatCompletionResponse``: The upstream Mistral response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return track_sync(
            url=self._intake_url,
            call="chat.complete",
            before={"model": kwargs.get("model"), "message_count": len(kwargs.get("messages", [])), "tags": tags},
            fn=lambda: super(Chat, self).complete(**kwargs),
            after_extra=_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )

    async def complete_async(self, **kwargs: Any):
        """Send an async chat completion request with telemetry.

        Args:
            model (str): Mistral model identifier (e.g. ``"mistral-small-latest"``).
            messages (list[dict]): Conversation history in OpenAI message format.
            tags (dict, optional): Arbitrary key/value metadata. Stripped before the upstream call.
            **kwargs: Any other kwargs accepted by ``mistralai`` ``chat.complete_async()``.

        Returns:
            ``ChatCompletionResponse``: The upstream Mistral response, unmodified.
        """
        tags = kwargs.pop("tags", {})
        return await track_async(
            url=self._intake_url,
            call="chat.complete_async",
            before={"model": kwargs.get("model"), "message_count": len(kwargs.get("messages", [])), "tags": tags},
            fn=lambda: super(Chat, self).complete_async(**kwargs),
            after_extra=_usage,
            client_id=self._client_id,
            bearer_token=self._bearer_token,
        )


class Mistral(_Mistral):
    """Drop-in replacement for ``mistralai.client.Mistral`` with Weflayr telemetry.

    Args:
        api_key (str): Your Mistral API key.
        intake_url (str, optional): Weflayr intake API base URL.
        client_id (str, optional): Client identifier sent in the endpoint path.
        bearer_token (str, optional): Bearer token for the Authorization header.
        **kwargs: Forwarded unchanged to ``mistralai.client.Mistral``.

    Attributes:
        chat (:class:`Chat`): Instrumented chat client.
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
        self._intake_url = intake_url
        self._client_id = client_id
        self._bearer_token = bearer_token

    def __getattr__(self, name: str) -> Any:
        instance = super().__getattr__(name)
        if name == "chat":
            instance = Chat(
                self.sdk_configuration,
                parent_ref=self,
                intake_url=self._intake_url,
                client_id=self._client_id,
                bearer_token=self._bearer_token,
            )
            object.__setattr__(self, name, instance)
        return instance
