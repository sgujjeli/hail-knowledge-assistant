"""
LLM orchestration — Azure OpenAI GPT-4o (primary) + Anthropic Claude Sonnet (fallback).

Primary endpoint:
  POST https://{resource}.openai.azure.com/openai/deployments/gpt-4o/chat/completions
       ?api-version=2024-02-01

Streaming uses Server-Sent Events (SSE): each token arrives as a delta chunk.
The FastAPI route wraps this in a StreamingResponse with text/event-stream content type.

Hallucination controls enforced here:
  - temperature=0 — deterministic, no creativity
  - System prompt (in prompts.py) — ONLY-use-context instruction
  - max_tokens=1200 — prevents runaway verbose answers
  - stream=True — yields tokens; client sees partial answers fast
"""
import asyncio
from collections.abc import AsyncGenerator
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from core.config import settings


def _build_azure_client() -> AsyncAzureOpenAI:
    """
    Prefer Managed Identity / DefaultAzureCredential (no key in .env).
    Falls back to api_key if AZURE_OPENAI_API_KEY is explicitly set.

    Local dev:  run `az login` once — DefaultAzureCredential picks up your session.
    Production: assign Managed Identity with 'Cognitive Services OpenAI User' role.
    """
    if settings.azure_openai_api_key:
        # Explicit key present — use it (useful for CI or quick testing)
        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


# ── Azure OpenAI async client ────────────────────────────────────────────────
_azure_client = _build_azure_client()


async def stream_azure_openai(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Azure OpenAI GPT-4o.
    Retries up to 3 times with exponential backoff on rate limit (429).
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stream = await _azure_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                temperature=0,
                max_tokens=1200,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return  # success — exit retry loop
        except Exception as e:
            from openai import RateLimitError
            if isinstance(e, RateLimitError) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait)
                continue
            raise


async def generate_non_streaming(messages: list[dict]) -> str:
    """
    Non-streaming completion — used by RAGAS evaluation pipeline where
    we need the full response before computing faithfulness scores.
    """
    response = await _azure_client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=messages,
        temperature=0,
        max_tokens=1200,
        stream=False,
    )
    return response.choices[0].message.content


# ── Anthropic Claude Sonnet fallback ────────────────────────────────────────
async def stream_claude_sonnet(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Fallback: stream from Anthropic Claude Sonnet via the Anthropic SDK.
    Activates when ANTHROPIC_API_KEY is set and Azure OpenAI fails.

    Endpoint: https://api.anthropic.com/v1/messages
    """
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Separate system message from user/assistant turns
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=system_msg,
        messages=user_messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_with_fallback(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Try Azure OpenAI first.
    Falls back to Claude Sonnet only when ENABLE_ANTHROPIC_FALLBACK=true in .env
    and ANTHROPIC_API_KEY is set.
    """
    try:
        async for token in stream_azure_openai(messages):
            yield token
    except Exception as azure_err:
        if settings.enable_anthropic_fallback and settings.anthropic_api_key:
            async for token in stream_claude_sonnet(messages):
                yield token
        else:
            raise azure_err
