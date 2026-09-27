import logging
import os
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_groq import ChatGroq
import time



class Provider(str, Enum):
    OPENAI = "openai"
    WATSONX = "watsonx"
    GROQ = "groq"


def get_llm(temperature: float = 0.0):
    """
    Factory function: returns a configured chat model based on
    the LLM_PROVIDER environment variable.
    """
    # Defaulting to openrouter so the pipeline runs without extra config
    provider = os.environ.get("LLM_PROVIDER", Provider.OPENAI.value)

    if provider == Provider.OPENAI.value:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export it or add it to your .env file."
            )
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=api_key
        )

    elif provider == Provider.WATSONX.value:
        api_key = os.environ.get("WATSONX_APIKEY")
        project_id = os.environ.get("WATSONX_PROJECT_ID")
        missing = [
            name for name, val in [
                ("WATSONX_APIKEY", api_key),
                ("WATSONX_PROJECT_ID", project_id),
            ] if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required Watsonx env var(s): {', '.join(missing)}. "
                "Export them or add them to your .env file."
            )

        model_id = os.environ.get("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct")
        url = os.environ.get("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
        return ChatWatsonx(
            apikey=api_key,
            project_id=project_id,
            url=url,
            model_id=model_id,
            params={"temperature": temperature},
        )

    elif provider == Provider.GROQ.value:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it or add it to your .env file."
            )
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")



logger = logging.getLogger(__name__)


class LLMInvocationError(Exception):
    """Raised when an LLM call fails after all retry attempts are exhausted."""
    pass


def invoke_with_retry(chain, input_data, max_attempts: int = 3, base_delay: float = 1.0):
    """
    Invokes a LangChain runnable chain with exponential backoff.

    Retries transient failures (rate limits, timeouts, connection drops) up to
    max_attempts times. If every attempt fails, raises LLMInvocationError with
    the original exception attached, so callers get one clear error instead of
    a raw stack trace surfacing from deep inside LangGraph's execution.

    Note: this currently retries on ANY exception raised by chain.invoke(),
    which is intentionally simple. If you start seeing it waste retries on
    non-transient errors (e.g. a malformed prompt or an auth failure), narrow
    the except clause to the specific exception types your provider raises
    (e.g. openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError).
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(input_data)
        except Exception as exc:
            last_exception = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, max_attempts, exc, delay
            )
            time.sleep(delay)

    raise LLMInvocationError(
        f"LLM call failed after {max_attempts} attempts: {last_exception}"
    ) from last_exception


def validate_provider_config() -> None:
    """
    Checks that the currently configured LLM_PROVIDER has everything it
    needs (a valid provider name and the required credentials) before the
    graph starts running. Raises EnvironmentError/ValueError immediately
    with a clear message, instead of letting a misconfigured provider
    fail deep inside a LangGraph node.
    """
    provider = os.environ.get("LLM_PROVIDER", Provider.OPENAI.value)

    if provider == Provider.OPENAI.value:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing. "
                "Add it to your .env file."
            )
    elif provider == Provider.WATSONX.value:
        missing = [
            name for name in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID")
            if not os.environ.get(name)
        ]
        if missing:
            raise EnvironmentError(
                f"LLM_PROVIDER is set to 'watsonx' but missing: {', '.join(missing)}. "
                "Add them to your .env file."
            )
    elif provider == Provider.GROQ.value:
        if not os.environ.get("GROQ_API_KEY"):
            raise EnvironmentError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be one of: "
            f"{', '.join(p.value for p in Provider)}"
        )