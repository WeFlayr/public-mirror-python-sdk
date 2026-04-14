# Weflayr SDK

Drop-in instrumented wrappers for AI clients. Add telemetry to your LLM calls in one line - no changes to your existing code structure required.

## How it works

Weflayr wraps the official provider SDKs and automatically fires telemetry events to your Weflayr intake API before, after, and on error for every LLM call. Your application code stays identical.

```
your code  →  weflayr.sdk.openai.OpenAI  →  openai.OpenAI  →  OpenAI API
                       ↓
              Weflayr Intake API
              (before / after / error events)
```

## Installation

```bash
pip install weflayr
```

## Quickstart

### Configuration

Set the following environment variables:

| Variable | Description | Default |
|---|---|---|
| `WEFLAYR_INTAKE_URL` | Your Weflayr intake API base URL | `https://api.weflayr.com` |
| `WEFLAYR_CLIENT_ID` | Your client identifier | `unknown_client` |
| `WEFLAYR_CLIENT_SECRET` | Your bearer token | _(empty)_ |

```bash
export WEFLAYR_INTAKE_URL="https://api.weflayr.com"
export WEFLAYR_CLIENT_ID="my-app"
export WEFLAYR_CLIENT_SECRET="your-secret-token"
```

---

## OpenAI

Drop-in replacement for `openai.OpenAI` and `openai.AsyncOpenAI`.

### Sync

```python
from weflayr.sdk.openai.client import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

### Async

```python
import asyncio
from weflayr.sdk.openai.client import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")

async def main():
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Tagging calls

Pass a `tags` dict to attach arbitrary metadata to your telemetry events:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this."}],
    tags={"feature": "summarization", "user_id": "u_123"},
)
```

---

## Mistral AI

Drop-in replacement for `mistralai.client.Mistral`.

### Sync

```python
from weflayr.sdk.mistralai.client import Mistral

client = Mistral(api_key="sk-...")

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

### Async

```python
import asyncio
from weflayr.sdk.mistralai.client import Mistral

client = Mistral(api_key="sk-...")

async def main():
    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Tagging calls

```python
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Translate this."}],
    tags={"feature": "translation", "env": "production"},
)
```

---

## Telemetry events

For every LLM call, Weflayr sends up to three events to your intake API:

| Event | When | Payload includes |
|---|---|---|
| `<call>.before` | Before the provider call | `model`, `message_count`, `tags` |
| `<call>.after` | On success | `model`, `elapsed_ms`, `prompt_tokens`, `completion_tokens` |
| `<call>.error` | On failure | `model`, `elapsed_ms`, `error_type`, `error_message`, `status_code` |

Events are sent **fire-and-forget** in background threads — they never block your application or raise exceptions.

---

## Advanced: per-client configuration

Override the intake URL, client ID, and bearer token directly on the client instead of using environment variables:

```python
from weflayr.sdk.openai.client import OpenAI

client = OpenAI(
    api_key="sk-...",
    intake_url="https://api.weflayr.com",
    client_id="my-service",
    bearer_token="my-secret",
)
```

---

## License

[Elastic License 2.0](LICENSE) — free to use, modifications and redistribution not permitted.
