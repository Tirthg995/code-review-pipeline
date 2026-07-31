import os
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx


class Provider(str, Enum):
    OPENAI = "openai"
    WATSONX = "watsonx"


def get_llm(temperature: float = 0.0):
    """
    Factory function: returns a configured chat model based on
    the LLM_PROVIDER environment variable.
    """
    provider = os.environ.get("LLM_PROVIDER", Provider.WATSONX.value)

    if provider == Provider.OPENAI.value:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=temperature)

    elif provider == Provider.WATSONX.value:
        model_id = os.environ.get("WATSONX_MODEL", "meta-llama/llama-3-3-70b-instruct")
        return ChatWatsonx(
            apikey=os.environ.get("WATSONX_APIKEY"),
            project_id=os.environ.get("WATSONX_PROJECT_ID"),
            url=os.environ.get("WATSONX_URL"),
            model_id=model_id,
            params={"temperature": temperature},
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")