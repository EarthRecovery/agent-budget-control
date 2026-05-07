from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import inspect
import os
import asyncio
import random
import time

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from together import AsyncTogether
from tqdm.auto import tqdm

from .model_capabilities import is_openai_reasoning_model_name

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str
    provider_name: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(
            {
                key: val
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        )
    return str(value)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_flag_is_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_usage(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None

    raw_usage = _jsonable(usage)
    if not isinstance(raw_usage, dict):
        raw_usage = {"value": raw_usage}

    input_tokens = _as_int(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
    )
    output_tokens = _as_int(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
    )
    total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw": raw_usage,
    }


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(0.0, float(retry_after_ms) / 1000.0)
        except (TypeError, ValueError):
            return None

    return None


async def _maybe_aclose_client(client: Any) -> None:
    if client is None:
        return

    close_method = getattr(client, "close", None)
    if close_method is None:
        return

    close_result = close_method()
    if inspect.isawaitable(close_result):
        await close_result

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

    async def close(self) -> None:
        return None

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    provider_name = "openai"
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")

        # Disable SDK-level retries so batch retry behavior is controlled in ConcurrentLLM.
        self._client_kwargs = {
            "api_key": self.api_key,
            "max_retries": 0,
        }
        if timeout is not None:
            self._client_kwargs["timeout"] = timeout
        self.client = None
        self._client_loop_id = None

    def _get_client(self) -> AsyncOpenAI:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncOpenAI(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None

    def _normalize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(kwargs)
        if self._uses_responses_api():
            if "max_completion_tokens" in normalized and "max_output_tokens" not in normalized:
                normalized["max_output_tokens"] = normalized.pop("max_completion_tokens")
            if "max_tokens" in normalized and "max_output_tokens" not in normalized:
                normalized["max_output_tokens"] = normalized.pop("max_tokens")
            reasoning_effort = normalized.pop("reasoning_effort", None)
            if reasoning_effort is not None and "reasoning" not in normalized:
                normalized["reasoning"] = {"effort": str(reasoning_effort)}
        else:
            needs_max_completion_tokens = self.model_name.startswith("gpt-5") or self.model_name.startswith("o")
            if needs_max_completion_tokens and "max_tokens" in normalized and "max_completion_tokens" not in normalized:
                normalized["max_completion_tokens"] = normalized.pop("max_tokens")
        return normalized

    def _uses_responses_api(self) -> bool:
        normalized = self.model_name.strip().lower()
        return "codex" in normalized

    def _normalize_response_input_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            normalized_blocks: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    normalized_block = dict(block)
                    block_type = normalized_block.get("type")
                    if block_type == "text":
                        normalized_block["type"] = "input_text"
                    normalized_blocks.append(normalized_block)
                elif block is not None:
                    normalized_blocks.append({"type": "input_text", "text": str(block)})
            return normalized_blocks
        if isinstance(content, dict):
            normalized_block = dict(content)
            if normalized_block.get("type") == "text":
                normalized_block["type"] = "input_text"
            return [normalized_block]
        if content is None:
            return ""
        return str(content)

    def _messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            normalized_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": self._normalize_response_input_content(message.get("content")),
                }
            )
        return normalized_messages

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        output_items = getattr(response, "output", None) or []
        collected: List[str] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            item_content = getattr(item, "content", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
                item_content = item.get("content")
            if item_type != "message" or not item_content:
                continue
            for block in item_content:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                    block_text = block.get("text")
                if block_type == "output_text" and block_text is not None:
                    collected.append(str(block_text))
        if collected:
            return "".join(collected)
        raise ValueError("Responses API response did not include output_text")

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized = [dict(message) for message in messages]
        if "o1-mini" in self.model_name and normalized and normalized[0].get("role") == "system":
            normalized = normalized[1:]
        if is_openai_reasoning_model_name(self.model_name):
            for message in normalized:
                if message.get("role") == "system":
                    message["role"] = "developer"
        return normalized

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        kwargs = self._normalize_kwargs(kwargs)
        messages = self._normalize_messages(messages)

        if self._uses_responses_api():
            response = await self._get_client().responses.create(
                model=self.model_name,
                input=self._messages_to_responses_input(messages),
                **kwargs
            )
            status = getattr(response, "status", None)
            incomplete_details = getattr(response, "incomplete_details", None)
            incomplete_reason = getattr(incomplete_details, "reason", None)
            if incomplete_reason is None and isinstance(incomplete_details, dict):
                incomplete_reason = incomplete_details.get("reason")
            if status == "incomplete" and incomplete_reason == "max_output_tokens":
                raise ValueError("Content filtered or length exceeded")
            content = self._extract_response_text(response)
        else:
            response = await self._get_client().chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            content = response.choices[0].message.content

        if not self._uses_responses_api() and finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=content,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class OpenRouterProvider(LLMProvider):
    """OpenRouter via the OpenAI-compatible endpoint."""
    provider_name = "openrouter"

    def __init__(
        self,
        model_name: str = "qwen/qwen3.6-plus",
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and not found in environment variables")

        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_X_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self._client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "max_retries": 0,
        }
        if default_headers:
            self._client_kwargs["default_headers"] = default_headers
        if timeout is not None:
            self._client_kwargs["timeout"] = timeout
        self.client = None
        self._client_loop_id = None

    def _get_client(self) -> AsyncOpenAI:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncOpenAI(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None

    def _uses_gemini_prompt_caching_breakpoints(self) -> bool:
        return self.model_name.startswith("google/gemini-")

    def _message_has_explicit_cache_control(self, message: Dict[str, Any]) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        for block in content:
            if isinstance(block, dict) and block.get("cache_control") is not None:
                return True
        return False

    def _normalize_content_blocks(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    blocks.append(dict(item))
                else:
                    blocks.append({"type": "text", "text": str(item)})
            return blocks
        if isinstance(content, dict):
            return [dict(content)]
        return [{"type": "text", "text": "" if content is None else str(content)}]

    def _apply_gemini_cache_breakpoint(
        self,
        messages: List[Dict[str, Any]],
        cache_control: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        if any(self._message_has_explicit_cache_control(message) for message in messages):
            return messages

        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            normalized_message = dict(message)
            normalized_message["content"] = self._normalize_content_blocks(
                normalized_message.get("content")
            )
            normalized_messages.append(normalized_message)

        target_message = normalized_messages[-1]
        blocks = list(target_message.get("content") or [])
        cache_payload = dict(cache_control)

        for block_idx in range(len(blocks) - 1, -1, -1):
            block = blocks[block_idx]
            if not isinstance(block, dict):
                continue
            if block.get("type", "text") != "text":
                continue
            updated_block = dict(block)
            updated_block["cache_control"] = cache_payload
            blocks[block_idx] = updated_block
            target_message["content"] = blocks
            return normalized_messages

        blocks.append({"type": "text", "text": "", "cache_control": cache_payload})
        target_message["content"] = blocks
        return normalized_messages

    def _normalize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(kwargs)
        thinking_mode = normalized.pop("thinking_mode", None)
        cache_control = normalized.pop("cache_control", None)

        extra_body = normalized.get("extra_body")
        if extra_body is None:
            extra_body = {}
        elif not isinstance(extra_body, dict):
            raise ValueError("OpenRouter extra_body must be a dict")
        else:
            extra_body = dict(extra_body)

        if thinking_mode is not None:
            reasoning = extra_body.get("reasoning")
            if reasoning is None:
                reasoning = {}
            elif not isinstance(reasoning, dict):
                raise ValueError("OpenRouter extra_body.reasoning must be a dict")
            else:
                reasoning = dict(reasoning)
            reasoning.setdefault("effort", str(thinking_mode))
            extra_body["reasoning"] = reasoning

        explicit_prompt_cache_control = cache_control
        if explicit_prompt_cache_control is None:
            explicit_prompt_cache_control = extra_body.get("cache_control")

        if (
            explicit_prompt_cache_control is not None
            and self._uses_gemini_prompt_caching_breakpoints()
        ):
            normalized["_openrouter_prompt_cache_control"] = dict(explicit_prompt_cache_control)
            extra_body.pop("cache_control", None)
        elif cache_control is not None:
            extra_body["cache_control"] = cache_control

        if extra_body:
            normalized["extra_body"] = extra_body

        return normalized

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        kwargs = self._normalize_kwargs(kwargs)
        prompt_cache_control = kwargs.pop("_openrouter_prompt_cache_control", None)
        if prompt_cache_control is not None:
            messages = self._apply_gemini_cache_breakpoint(messages, prompt_cache_control)
        response = await self._get_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class GeminiProvider(LLMProvider):
    """Gemini API provider via Google's OpenAI-compatible endpoint."""
    provider_name = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables")

        self._client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "max_retries": 0,
        }
        if timeout is not None:
            self._client_kwargs["timeout"] = timeout
        self.client = None
        self._client_loop_id = None

    def _get_client(self) -> AsyncOpenAI:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncOpenAI(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self._get_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation"""
    provider_name = "deepseek"
    
    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and not found in environment variables")

        # Disable SDK-level retries so batch retry behavior is controlled in ConcurrentLLM.
        self._client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://api.deepseek.com",
            "max_retries": 0,
        }
        if timeout is not None:
            self._client_kwargs["timeout"] = timeout
        self.client = None
        self._client_loop_id = None

    def _get_client(self) -> AsyncOpenAI:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncOpenAI(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self._get_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation
    Refer to https://github.com/anthropics/anthropic-sdk-python
    """
    provider_name = "anthropic"
    
    def __init__(self, model_name: str = "claude-3.5-sonnet-20240620", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self._client_kwargs = {"api_key": self.api_key}
        self.client = None
        self._client_loop_id = None
        self.prompt_cache_enabled = _env_flag_is_enabled("ANTHROPIC_PROMPT_CACHE", True)

    def _get_client(self) -> AsyncAnthropic:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncAnthropic(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    def _normalize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(kwargs)
        prompt_cache = normalized.pop("prompt_cache", None)
        if prompt_cache is not None:
            self_prompt_cache_enabled = bool(prompt_cache)
        else:
            self_prompt_cache_enabled = self.prompt_cache_enabled
        normalized["_anthropic_prompt_cache_enabled"] = self_prompt_cache_enabled
        # New Claude 4.x/5.x models reject/deprecate the temperature parameter.
        if (
            self.model_name.startswith("claude-opus-4")
            or self.model_name.startswith("claude-opus-5")
            or self.model_name.startswith("claude-sonnet-4")
            or self.model_name.startswith("claude-sonnet-5")
        ):
            normalized.pop("temperature", None)
        thinking = normalized.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise ValueError("Anthropic thinking must be a dict")
            thinking_type = str(thinking.get("type", "enabled"))
            normalized_thinking = {"type": thinking_type}
            if thinking_type == "enabled":
                budget_tokens = thinking.get("budget_tokens")
                if budget_tokens is not None:
                    normalized_thinking["budget_tokens"] = int(budget_tokens)
            elif thinking_type == "adaptive":
                display = thinking.get("display")
                if display is not None:
                    normalized_thinking["display"] = str(display)
            else:
                raise ValueError(f"Unsupported Anthropic thinking type: {thinking_type}")
            normalized["thinking"] = normalized_thinking
        output_config = normalized.get("output_config")
        if output_config is not None:
            if not isinstance(output_config, dict):
                raise ValueError("Anthropic output_config must be a dict")
            normalized_output_config = {}
            effort = output_config.get("effort")
            if effort is not None:
                normalized_output_config["effort"] = str(effort)
            output_format = output_config.get("format")
            if output_format is not None:
                normalized_output_config["format"] = output_format
            normalized["output_config"] = normalized_output_config
        return normalized

    def _make_text_block(self, text: str, *, cache: bool = False) -> Dict[str, Any]:
        block: Dict[str, Any] = {"type": "text", "text": text}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def _prepare_anthropic_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        prompt_cache_enabled: bool,
    ) -> tuple[Any, List[Dict[str, Any]]]:
        system_content = ""
        chat_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(
                    {
                        "role": "assistant" if msg["role"] == "assistant" else "user",
                        "content": [self._make_text_block(msg["content"])],
                    }
                )

        if not prompt_cache_enabled:
            return system_content, chat_messages

        # Anthropic prompt caching works via cache breakpoints on content blocks.
        # By default, cache the largest reusable prefix while leaving the final
        # user query uncached. This is usually the estimation instruction in our
        # workloads and changes most frequently between requests.
        if len(chat_messages) >= 2:
            chat_messages[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        elif system_content:
            system_content = [self._make_text_block(system_content, cache=True)]
        elif chat_messages:
            chat_messages[0]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        return system_content, chat_messages

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        kwargs = self._normalize_kwargs(kwargs)
        prompt_cache_enabled = bool(kwargs.pop("_anthropic_prompt_cache_enabled", self.prompt_cache_enabled))
        system_content, chat_messages = self._prepare_anthropic_messages(
            messages,
            prompt_cache_enabled=prompt_cache_enabled,
        )

        response = await self._get_client().messages.create(
            model=self.model_name,
            system=system_content,
            messages=chat_messages,
            **kwargs
        )
        if response.stop_reason == "max_tokens":
            raise ValueError("Max tokens exceeded")
        return LLMResponse(
            content=response.content[0].text,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class TogetherProvider(LLMProvider):
    """Together AI API provider implementation"""
    provider_name = "together"
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided and not found in environment variables")
        
        self._client_kwargs = {"api_key": self.api_key}
        self.client = None
        self._client_loop_id = None

    def _get_client(self) -> AsyncTogether:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncTogether(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self._get_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model,
            provider_name=self.provider_name,
            usage=_normalize_usage(getattr(response, "usage", None)),
            request_id=getattr(response, "id", None),
        )

class ConcurrentLLM:
    """Unified concurrent interface for multiple LLM providers"""
    
    def __init__(self, provider: Union[str, LLMProvider], model_name: Optional[str] = None, 
                api_key: Optional[str] = None, max_concurrency: int = 4,
                timeout: Optional[Union[float, httpx.Timeout]] = None):
        """
        Initialize the concurrent LLM client.
        
        Args:
            provider: Either a provider instance or a string ('openai', 'anthropic', 'together')
            model_name: Model name (if provider is a string)
            api_key: API key (if provider is a string)
            max_concurrency: Maximum number of concurrent requests
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            if provider.lower() == "openai":
                self.provider = OpenAIProvider(model_name or "gpt-4o", api_key, timeout=timeout)
            elif provider.lower() == "openrouter":
                self.provider = OpenRouterProvider(
                    model_name or "qwen/qwen3.6-plus", api_key, timeout=timeout
                )
            elif provider.lower() == "gemini":
                self.provider = GeminiProvider(
                    model_name or "gemini-2.5-pro", api_key, timeout=timeout
                )
            elif provider.lower() == "deepseek":
                self.provider = DeepSeekProvider(
                    model_name or "deepseek-reasoner", api_key, timeout=timeout
                )
            elif provider.lower() == "anthropic":
                self.provider = AnthropicProvider(model_name or "claude-3-7-sonnet-20250219", api_key)
            elif provider.lower() == "together":
                self.provider = TogetherProvider(model_name or "meta-llama/Llama-3-70b-chat-hf", api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Store max_concurrency but don't create the semaphore yet
        self.max_concurrency = max_concurrency
        self._semaphore = None
    
    @property
    def semaphore(self):
        """
        Lazy initialization of the semaphore.
        This ensures the semaphore is created in the event loop where it's used.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response with concurrency control"""
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)

    def _build_failure_result(self, messages: List[Dict[str, str]], error: Exception) -> Dict[str, Any]:
        status_code = getattr(error, "status_code", None)
        error_code = getattr(error, "code", None)
        error_type = getattr(error, "type", None)
        exception_class = type(error).__name__
        error_message = str(error)
        error_body = getattr(error, "body", None)
        retry_after_seconds = _extract_retry_after_seconds(error)

        if isinstance(error_body, dict):
            error_payload = error_body.get("error", error_body)
            if isinstance(error_payload, dict):
                error_code = error_code or error_payload.get("code")
                error_type = error_type or error_payload.get("type")
                error_message = error_payload.get("message", error_message)

        retryable = True
        if isinstance(error, ValueError):
            retryable = False
        if status_code is not None:
            try:
                status_code_int = int(status_code)
            except (TypeError, ValueError):
                status_code_int = None
            if status_code_int is not None and status_code_int < 500 and status_code_int not in (408, 409, 429):
                retryable = False

        lowered_error = error_message.lower()
        if (
            "invalid_prompt" in lowered_error
            or "invalid prompt" in lowered_error
            or "usage policy" in lowered_error
        ):
            retryable = False
            error_code = error_code or "invalid_prompt"

        return {
            "messages": messages,
            "response": "",
            "model": getattr(self.provider, "model_name", None),
            "provider": getattr(self.provider, "provider_name", None),
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "exception_class": exception_class,
            "error_code": error_code,
            "status_code": status_code,
            "retryable": retryable,
            "retry_after_seconds": retry_after_seconds,
            "usage": None,
            "request_id": getattr(error, "request_id", None),
        }

    def _build_interaction_record(
        self,
        *,
        attempt: int,
        success: bool,
        model: Optional[str],
        usage: Optional[Dict[str, Any]],
        request_id: Optional[str],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        exception_class: Optional[str] = None,
        error_code: Optional[str] = None,
        status_code: Optional[Any] = None,
        retryable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        usage = usage or {}
        return {
            "attempt": int(attempt),
            "success": bool(success),
            "provider": getattr(self.provider, "provider_name", None),
            "model": model,
            "request_id": request_id,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "usage": usage or None,
            "error": error,
            "error_type": error_type,
            "exception_class": exception_class,
            "error_code": error_code,
            "status_code": status_code,
            "retryable": retryable,
        }

    def _compute_retry_sleep_seconds(
        self,
        *,
        retry_count: int,
        retry_failure_results: List[Dict[str, Any]],
        initial_retry_delay_seconds: float,
        max_retry_delay_seconds: float,
        retry_jitter_seconds: float,
    ) -> float:
        retry_after_hints = [
            float(result["retry_after_seconds"])
            for result in retry_failure_results
            if result.get("retry_after_seconds") is not None
        ]
        if retry_after_hints:
            return min(max_retry_delay_seconds, max(retry_after_hints))

        base_delay = min(
            max_retry_delay_seconds,
            initial_retry_delay_seconds * (2 ** max(0, retry_count)),
        )
        if retry_jitter_seconds <= 0:
            return base_delay

        jitter = random.uniform(0.0, min(retry_jitter_seconds, base_delay))
        return min(max_retry_delay_seconds, base_delay + jitter)

    async def _run_batch_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        progress_bar: Optional[tqdm],
        generate_kwargs: Dict[str, Any],
        max_retries: int,
        initial_retry_delay_seconds: float,
        max_retry_delay_seconds: float,
        retry_jitter_seconds: float,
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        results = [None] * len(messages_list)
        position_map = {id(messages): i for i, messages in enumerate(messages_list)}
        latest_failure_results: Dict[int, Dict[str, Any]] = {}
        attempt_histories: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(len(messages_list))
        }

        self._semaphore = None
        current_batch = messages_list.copy()
        retry_count = 0
        next_batch: List[List[Dict[str, str]]] = []

        try:
            while current_batch and retry_count < max_retries:
                batch_results = []
                failures = []
                retry_failure_results = []
                attempt_number = retry_count + 1

                tasks_with_messages = [
                    (msg, asyncio.create_task(self.generate(msg, **generate_kwargs)))
                    for msg in current_batch
                ]
                for messages, task in tasks_with_messages:
                    try:
                        response = await task
                        position = position_map[id(messages)]
                        attempt_histories[position].append(
                            self._build_interaction_record(
                                attempt=attempt_number,
                                success=True,
                                model=response.model_name,
                                usage=response.usage,
                                request_id=response.request_id,
                            )
                        )
                        batch_results.append((position, {
                            "messages": messages,
                            "response": response.content,
                            "model": response.model_name,
                            "provider": response.provider_name,
                            "success": True,
                            "usage": response.usage,
                            "request_id": response.request_id,
                            "attempts": list(attempt_histories[position]),
                        }))
                    except Exception as e:
                        position = position_map[id(messages)]
                        failure_result = self._build_failure_result(messages, e)
                        print(
                            "[DEBUG] request failed: "
                            f"exception_class={failure_result.get('exception_class')} "
                            f"status_code={failure_result.get('status_code')} "
                            f"error_type={failure_result.get('error_type')} "
                            f"error_code={failure_result.get('error_code')} "
                            f"request_id={failure_result.get('request_id')} "
                            f"retryable={failure_result.get('retryable')} "
                            f"message={failure_result.get('error')}"
                        )
                        attempt_histories[position].append(
                            self._build_interaction_record(
                                attempt=attempt_number,
                                success=False,
                                model=failure_result.get("model"),
                                usage=failure_result.get("usage"),
                                request_id=failure_result.get("request_id"),
                                error=failure_result.get("error"),
                                error_type=failure_result.get("error_type"),
                                exception_class=failure_result.get("exception_class"),
                                error_code=failure_result.get("error_code"),
                                status_code=failure_result.get("status_code"),
                                retryable=failure_result.get("retryable"),
                            )
                        )
                        latest_failure_results[position] = failure_result
                        if failure_result["retryable"]:
                            failures.append(messages)
                            retry_failure_results.append(failure_result)
                        else:
                            failure_result["attempts"] = list(attempt_histories[position])
                            batch_results.append((position, failure_result))

                for position, result in batch_results:
                    results[position] = result
                if progress_bar is not None and batch_results:
                    progress_bar.update(len(batch_results))
                    progress_bar.set_postfix(retries=retry_count, refresh=False)

                if failures:
                    sleep_seconds = self._compute_retry_sleep_seconds(
                        retry_count=retry_count,
                        retry_failure_results=retry_failure_results,
                        initial_retry_delay_seconds=initial_retry_delay_seconds,
                        max_retry_delay_seconds=max_retry_delay_seconds,
                        retry_jitter_seconds=retry_jitter_seconds,
                    )
                    retry_count += 1
                    next_batch = failures
                    position_map = {
                        id(messages): position_map[id(messages)]
                        for messages in next_batch
                    }
                    current_batch = next_batch
                    if retry_count < max_retries:
                        print(
                            f"[DEBUG] retrying {len(next_batch)} failed messages after "
                            f"{sleep_seconds:.2f}s backoff; retry_count={retry_count}"
                        )
                        await asyncio.sleep(sleep_seconds)
                else:
                    current_batch = []

            unresolved_failures = list(current_batch) if current_batch and retry_count >= max_retries else []
            for messages in unresolved_failures:
                position = position_map[id(messages)]
                failure_result = dict(
                    latest_failure_results.get(position)
                    or self._build_failure_result(messages, RuntimeError("Max retries exceeded"))
                )
                failure_result["retryable"] = False
                failure_result["error"] = f'{failure_result.get("error", "Max retries exceeded")} (max retries exceeded)'
                failure_result["attempts"] = list(attempt_histories[position])
                results[position] = failure_result
            if progress_bar is not None and unresolved_failures:
                progress_bar.update(len(unresolved_failures))
                progress_bar.set_postfix(retries=retry_count, refresh=False)

            for idx, result in enumerate(results):
                if result is None:
                    results[idx] = {
                        "messages": messages_list[idx],
                        "response": "",
                        "model": getattr(self.provider, "model_name", None),
                        "provider": getattr(self.provider, "provider_name", None),
                        "success": False,
                        "error": "No response generated.",
                        "error_type": None,
                        "error_code": None,
                        "status_code": None,
                        "retryable": False,
                        "usage": None,
                        "request_id": None,
                        "attempts": list(attempt_histories[idx]),
                    }

            return results, unresolved_failures
        finally:
            await self.provider.close()
    
    def run_batch(self, 
                messages_list: List[List[Dict[str, str]]], 
                progress_bar: Optional[tqdm] = None,
                **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Process a batch with retries inside one event loop."""

        generate_kwargs = dict(kwargs)
        max_retries = int(generate_kwargs.pop("max_retries", 100))
        initial_retry_delay_seconds = float(generate_kwargs.pop("initial_retry_delay_seconds", 1.0))
        max_retry_delay_seconds = float(generate_kwargs.pop("max_retry_delay_seconds", 30.0))
        retry_jitter_seconds = float(generate_kwargs.pop("retry_jitter_seconds", 1.0))
        return asyncio.run(
            self._run_batch_async(
                messages_list=messages_list,
                progress_bar=progress_bar,
                generate_kwargs=generate_kwargs,
                max_retries=max_retries,
                initial_retry_delay_seconds=initial_retry_delay_seconds,
                max_retry_delay_seconds=max_retry_delay_seconds,
                retry_jitter_seconds=retry_jitter_seconds,
            )
        )



if __name__ == "__main__":
    # llm = ConcurrentLLM(provider="openai", model_name="gpt-4o")
    # llm = ConcurrentLLM(provider="anthropic", model_name="claude-3-5-sonnet-20240620")
    llm = ConcurrentLLM(provider="together", model_name="Qwen/Qwen2.5-7B-Instruct-Turbo")
    messages = [
        [{"role": "user", "content": "what is 2+2?"}],
        [{"role": "user", "content": "what is 2+3?"}],
        [{"role": "user", "content": "what is 2+4?"}],
        [{"role": "user", "content": "what is 2+5?"}],
        [{"role": "user", "content": "what is 2+6?"}],
        [{"role": "user", "content": "what is 2+7?"}],
        [{"role": "user", "content": "what is 2+8?"}],
        [{"role": "user", "content": "what is 2+9?"}],
        [{"role": "user", "content": "what is 2+10?"}],
        [{"role": "user", "content": "what is 2+11?"}],
        [{"role": "user", "content": "what is 2+12?"}],
        [{"role": "user", "content": "what is 2+13?"}],
        [{"role": "user", "content": "what is 2+14?"}],
        [{"role": "user", "content": "what is 2+15?"}],
        [{"role": "user", "content": "what is 2+16?"}],
        [{"role": "user", "content": "what is 2+17?"}],
    ]
    response = llm.run_batch(messages, max_tokens=100)
    print(f"final response: {response}")
