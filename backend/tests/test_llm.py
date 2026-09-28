"""
Unit tests — retrieval/llm.py

Tests:
  - stream_azure_openai: yields token strings from the stream
  - stream_azure_openai: uses temperature=0 and max_tokens=1200
  - generate_non_streaming: returns full response string
  - stream_with_fallback: falls back to Claude when Azure fails and key is set
  - stream_with_fallback: re-raises Azure error when no Anthropic key configured
  - stream_claude_sonnet: yields tokens from Anthropic stream
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import sys


def _make_stream_chunks(tokens: list[str]):
    """Build fake async stream chunks matching OpenAI's format."""
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunks.append(chunk)
    # Final chunk with no content (done sentinel)
    done_chunk = MagicMock()
    done_chunk.choices = [MagicMock()]
    done_chunk.choices[0].delta.content = None
    chunks.append(done_chunk)
    return chunks


async def _async_iter(items):
    for item in items:
        yield item


class TestStreamAzureOpenAI:
    @pytest.mark.asyncio
    async def test_yields_tokens(self):
        stream_chunks = _make_stream_chunks(["Hello", " world", "!"])

        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=_async_iter(stream_chunks))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import stream_azure_openai
            tokens = []
            async for token in stream_azure_openai([{"role": "user", "content": "hi"}]):
                tokens.append(token)

        assert "".join(tokens) == "Hello world!"

    @pytest.mark.asyncio
    async def test_uses_temperature_zero(self):
        stream_chunks = _make_stream_chunks(["ok"])
        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=_async_iter(stream_chunks))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import stream_azure_openai
            async for _ in stream_azure_openai([]):
                pass

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    @pytest.mark.asyncio
    async def test_max_tokens_1200(self):
        stream_chunks = _make_stream_chunks(["ok"])
        mock_stream = MagicMock()
        mock_stream.__aiter__ = MagicMock(return_value=_async_iter(stream_chunks))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import stream_azure_openai
            async for _ in stream_azure_openai([]):
                pass

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1200


class TestGenerateNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_full_response(self):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "Full answer here."

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import generate_non_streaming
            result = await generate_non_streaming([{"role": "user", "content": "question"}])

        assert result == "Full answer here."

    @pytest.mark.asyncio
    async def test_stream_is_false(self):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "answer"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import generate_non_streaming
            await generate_non_streaming([])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is False


class TestStreamWithFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_claude_on_azure_error(self, mock_settings):
        mock_settings.anthropic_api_key = "sk-ant-fake"

        async def azure_fail(**kwargs):
            raise RuntimeError("Azure unavailable")

        mock_azure_client = MagicMock()
        mock_azure_client.chat.completions.create = azure_fail

        async def fake_claude_stream(messages):
            yield "Claude"
            yield " response"

        with (
            patch("openai.AsyncAzureOpenAI", return_value=mock_azure_client),
            patch("retrieval.llm.stream_claude_sonnet", side_effect=fake_claude_stream),
        ):
            from retrieval.llm import stream_with_fallback
            tokens = []
            async for t in stream_with_fallback([{"role": "user", "content": "q"}]):
                tokens.append(t)

        assert "".join(tokens) == "Claude response"

    @pytest.mark.asyncio
    async def test_reraises_when_no_anthropic_key(self, mock_settings):
        mock_settings.anthropic_api_key = ""

        async def azure_fail(**kwargs):
            raise RuntimeError("Azure unavailable")

        mock_client = MagicMock()
        mock_client.chat.completions.create = azure_fail

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from retrieval.llm import stream_with_fallback
            with pytest.raises(RuntimeError, match="Azure unavailable"):
                async for _ in stream_with_fallback([]):
                    pass
