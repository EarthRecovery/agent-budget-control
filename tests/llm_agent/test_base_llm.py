import asyncio

from ragen.llm_agent import base_llm
from ragen.llm_agent.base_llm import ConcurrentLLM, LLMProvider, LLMResponse


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, scripted_results):
        self.model_name = "fake-model"
        self._scripted_results = list(scripted_results)

    async def generate(self, messages, **kwargs):
        result = self._scripted_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeAPIError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        status_code=None,
        request_id=None,
        headers=None,
        error_type=None,
        error_code=None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response = type("Response", (), {"headers": headers or {}})()
        self.type = error_type
        self.code = error_code


def test_run_batch_records_usage_and_attempts():
    provider = FakeProvider(
        [
            LLMResponse(
                content="ok",
                model_name="fake-model",
                provider_name="fake",
                usage={
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                },
                request_id="req_ok",
            )
        ]
    )
    llm = ConcurrentLLM(provider=provider, max_concurrency=1)

    results, unresolved = llm.run_batch([[{"role": "user", "content": "hello"}]], max_retries=1)

    assert unresolved == []
    assert results[0]["success"] is True
    assert results[0]["usage"]["total_tokens"] == 17
    assert len(results[0]["attempts"]) == 1
    assert results[0]["attempts"][0]["input_tokens"] == 12
    assert results[0]["attempts"][0]["request_id"] == "req_ok"


def test_run_batch_preserves_retry_attempt_history(monkeypatch):
    async def fake_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(base_llm.asyncio, "sleep", fake_sleep)
    provider = FakeProvider(
        [
            RuntimeError("temporary failure"),
            LLMResponse(
                content="ok",
                model_name="fake-model",
                provider_name="fake",
                usage={
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "total_tokens": 27,
                },
                request_id="req_retry",
            ),
        ]
    )
    llm = ConcurrentLLM(provider=provider, max_concurrency=1)

    results, unresolved = llm.run_batch([[{"role": "user", "content": "hello"}]], max_retries=2)

    assert unresolved == []
    assert results[0]["success"] is True
    assert len(results[0]["attempts"]) == 2
    assert results[0]["attempts"][0]["success"] is False
    assert results[0]["attempts"][1]["success"] is True
    assert results[0]["attempts"][1]["total_tokens"] == 27


def test_run_batch_uses_retry_after_header_and_records_exception_class(monkeypatch):
    sleep_calls = []

    async def fake_sleep(delay, *_args, **_kwargs):
        sleep_calls.append(delay)

    monkeypatch.setattr(base_llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(base_llm.random, "uniform", lambda *_args, **_kwargs: 0.0)
    provider = FakeProvider(
        [
            FakeAPIError(
                "rate limited",
                status_code=429,
                request_id="req_rate_limit",
                headers={"retry-after": "7"},
                error_type="rate_limit_error",
                error_code="rate_limit_exceeded",
            ),
            LLMResponse(
                content="ok",
                model_name="fake-model",
                provider_name="fake",
                usage={
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "total_tokens": 11,
                },
                request_id="req_ok_after_retry",
            ),
        ]
    )
    llm = ConcurrentLLM(provider=provider, max_concurrency=1)

    results, unresolved = llm.run_batch([[{"role": "user", "content": "hello"}]], max_retries=2)

    assert unresolved == []
    assert sleep_calls == [7.0]
    assert results[0]["success"] is True
    assert results[0]["attempts"][0]["exception_class"] == "FakeAPIError"
    assert results[0]["attempts"][0]["status_code"] == 429
    assert results[0]["attempts"][0]["request_id"] == "req_rate_limit"


def test_openai_provider_rebuilds_async_client_when_event_loop_changes(monkeypatch):
    created_clients = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

    monkeypatch.setattr(base_llm, "AsyncOpenAI", FakeAsyncOpenAI)
    provider = base_llm.OpenAIProvider(model_name="gpt-4o", api_key="test-key")

    async def get_client_identity():
        return id(provider._get_client())

    first_client = asyncio.run(get_client_identity())
    second_client = asyncio.run(get_client_identity())

    assert len(created_clients) == 2
    assert first_client != second_client


def test_openrouter_gemini_moves_cache_control_into_last_message_block(monkeypatch):
    captured_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            captured_calls.append(kwargs)
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"finish_reason": "stop", "message": message})()
            return type(
                "Response",
                (),
                {"choices": [choice], "model": kwargs["model"], "usage": None, "id": "req_gemini"},
            )()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr(base_llm, "AsyncOpenAI", FakeAsyncOpenAI)
    provider = base_llm.OpenRouterProvider(
        model_name="google/gemini-2.5-pro",
        api_key="test-key",
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "assistant reply"},
        {"role": "user", "content": "latest turn"},
    ]

    asyncio.run(
        provider.generate(
            messages,
            extra_body={"cache_control": {"type": "ephemeral"}},
        )
    )

    assert len(captured_calls) == 1
    request = captured_calls[0]
    assert "extra_body" not in request or "cache_control" not in request.get("extra_body", {})
    assert request["messages"][-1]["content"] == [
        {
            "type": "text",
            "text": "latest turn",
            "cache_control": {"type": "ephemeral"},
        }
    ]
