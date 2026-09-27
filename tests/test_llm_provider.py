import pytest
from src.llm_provider import invoke_with_retry, LLMInvocationError
from src.llm_provider import validate_provider_config


class _FlakyChain:
    """Fake chain that fails a fixed number of times, then succeeds."""
    def __init__(self, fail_times: int, result="ok"):
        self.fail_times = fail_times
        self.calls = 0
        self.result = result

    def invoke(self, _input):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"simulated transient failure #{self.calls}")
        return self.result


class _AlwaysFailsChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, _input):
        self.calls += 1
        raise RuntimeError("simulated permanent failure")


def test_invoke_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("src.llm_provider.time.sleep", lambda _seconds: None)

    chain = _FlakyChain(fail_times=2, result="success")
    result = invoke_with_retry(chain, {"x": 1}, max_attempts=3, base_delay=0.01)

    assert result == "success"
    assert chain.calls == 3


def test_invoke_with_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("src.llm_provider.time.sleep", lambda _seconds: None)

    chain = _AlwaysFailsChain()
    with pytest.raises(LLMInvocationError):
        invoke_with_retry(chain, {"x": 1}, max_attempts=3, base_delay=0.01)

    assert chain.calls == 3


def test_validate_provider_config_passes_with_openai_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-test")
    validate_provider_config()  # should not raise


def test_validate_provider_config_fails_with_missing_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        validate_provider_config()


def test_validate_provider_config_fails_with_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError):
        validate_provider_config()