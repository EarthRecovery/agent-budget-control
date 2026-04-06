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
        needs_max_completion_tokens = self.model_name.startswith("gpt-5") or self.model_name.startswith("o")
        if needs_max_completion_tokens and "max_tokens" in normalized and "max_completion_tokens" not in normalized:
            normalized["max_completion_tokens"] = normalized.pop("max_tokens")
        return normalized

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

    def _get_client(self) -> AsyncAnthropic:
        loop_id = id(asyncio.get_running_loop())
        if self.client is None or self._client_loop_id != loop_id:
            self.client = AsyncAnthropic(**self._client_kwargs)
            self._client_loop_id = loop_id
        return self.client

    async def close(self) -> None:
        if self.client is None:
            return
        await _maybe_aclose_client(self.client)
        self.client = None
        self._client_loop_id = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message if present
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                # Map to Anthropic's format
                chat_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
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
