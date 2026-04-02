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
    monkeypatch.setattr(base_llm.time, "sleep", lambda *_args, **_kwargs: None)
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
