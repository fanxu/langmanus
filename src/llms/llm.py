from google.protobuf.any import is_type
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models.tongyi import ChatTongyi
from src.llms.litellm_v2 import ChatLiteLLMV2 as ChatLiteLLM
from src.config import load_yaml_config
from typing import Optional
from litellm import LlmProviders
from pathlib import Path
from typing import Dict, Any

from src.config import (
    REASONING_MODEL,
    BASIC_MODEL,
    VL_MODEL,
)
from src.config.agents import LLMType


def create_openai_llm(
    model: str,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True,
        stream_usage=True,
        **kwargs,
    )


def create_deepseek_llm(
    model: str,
    temperature: float = 0.0,
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration
    """
    return ChatDeepSeek(
        model=model,
        temperature=temperature,
        streaming=True,
        **kwargs,
    )


def create_vertex_ai_llm(
    model: str,
    temperature: float = 0.0,
    **kwargs,
) -> ChatVertexAI:
    """
    Create a ChatVertexAI instance with the specified configuration
    """    
    return ChatVertexAI(
        model=model,
        temperature=temperature,
        streaming=True,
        **kwargs,
    )


def create_anthropic_llm(
    model: str,
    temperature: float = 0.0,
    **kwargs,
) -> ChatAnthropic:
    """
    Create a ChatAnthropic instance with the specified configuration
    """
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        streaming=True,
        max_tokens=4096,
        **kwargs,
    )


def create_tongyi_llm(
    model: str,
    temperature: float = 0.0,
    **kwargs,
) -> ChatTongyi:
    """
    Create a ChatTongyi instance with the specified configuration
    """
    return ChatTongyi(
        model=model,
        temperature=temperature,
        streaming=True,
        **kwargs,
    )


# def create_azure_llm(
#     azure_deployment: str,
#     azure_endpoint: str,
#     api_version: str,
#     api_key: str,
#     temperature: float = 0.0,
# ) -> AzureChatOpenAI:
#     """
#     create azure llm instance with specified configuration
#     """
#     return AzureChatOpenAI(
#         azure_deployment=azure_deployment,
#         azure_endpoint=azure_endpoint,
#         api_version=api_version,
#         api_key=api_key,
#         temperature=temperature,
#     )


# def create_litellm_model(
#     model: str,
#     base_url: Optional[str] = None,
#     api_key: Optional[str] = None,
#     temperature: float = 0.0,
#     **kwargs,
# ) -> ChatLiteLLM:
#     """
#     Support various different model's through LiteLLM's capabilities.
#     """

#     llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

#     if base_url:  # This will handle None or empty string
#         llm_kwargs["api_base"] = base_url

#     if api_key:  # This will handle None or empty string
#         llm_kwargs["api_key"] = api_key

#     return ChatLiteLLM(**llm_kwargs)


# Cache for LLM instances
_llm_cache: dict[LLMType, ChatOpenAI | ChatDeepSeek | ChatAnthropic | ChatVertexAI | ChatTongyi] = (
    {}
)


def is_litellm_model(model_name: str) -> bool:
    """
    Check if the model name indicates it should be handled by LiteLLM.

    Args:
        model_name: The name of the model to check

    Returns:
        bool: True if the model should be handled by LiteLLM, False otherwise
    """
    return (
        model_name
        and "/" in model_name
        and model_name.split("/")[0] in [p.value for p in LlmProviders]
    )


def get_model_provider(model_name: str) -> tuple[str, str]:
    """
    Get the provider and name of the model
    
    Args:
        model_name: The full model name in format "provider/model"
        
    Returns:
        tuple: (provider, model_name)
    """
    parts = model_name.split("/", 1)
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else ""
    return provider, model


def _create_llm_use_env(
    llm_type: LLMType,
) -> ChatOpenAI | ChatDeepSeek | ChatAnthropic | ChatVertexAI | ChatTongyi:
    if llm_type == "reasoning":
        model_provider, model_name = get_model_provider(REASONING_MODEL)
        if model_provider == LlmProviders.DEEPSEEK:
            llm = create_deepseek_llm(
                model=model_name,
            )
        elif model_provider == LlmProviders.ANTHROPIC:
            llm = create_anthropic_llm(
                model=model_name,
                # must set temperature to 1.0 to enable thinking
                temperature=1.0,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
        elif model_provider == "dashscope":
            llm = create_tongyi_llm(
                model=model_name,
            )
        else:
            raise ValueError(f"LLM model not supported: {REASONING_MODEL}")
    elif llm_type == "basic":
        model_provider, model_name = get_model_provider(BASIC_MODEL)
        if model_provider == LlmProviders.OPENAI:
            llm = create_openai_llm(
                model=model_name,
            )
        elif model_provider == LlmProviders.VERTEX_AI:
            llm = create_vertex_ai_llm(
                model=model_name,
                # thinking_budget=None,
            )
        elif model_provider == "anthropic":
            llm = create_anthropic_llm(
                model=model_name,
            )
        else:
            raise ValueError(f"LLM model not supported: {BASIC_MODEL}")
    elif llm_type == "vision":
        model_provider, model_name = get_model_provider(VL_MODEL)
        if model_provider == LlmProviders.OPENAI:
            llm = create_openai_llm(
                model=model_name,
            )
        elif model_provider == LlmProviders.VERTEX_AI:
            llm = create_vertex_ai_llm(
                model=model_name,
                thinking_budget=None,
            )
        else:
            raise ValueError(f"LLM model not supported: {VL_MODEL}")
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    return llm


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> ChatLiteLLM:
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL"),
        "basic": conf.get("BASIC_MODEL"),
        "vision": conf.get("VISION_MODEL"),
    }
    llm_conf = llm_type_map.get(llm_type)
    if not llm_conf:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM Conf: {llm_type}")
    return ChatLiteLLM(**llm_conf)


def get_llm_by_type(
    llm_type: LLMType,
) -> ChatOpenAI | ChatDeepSeek | AzureChatOpenAI | ChatLiteLLM:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    use_conf = conf.get("USE_CONF", False)
    if use_conf:
        llm = _create_llm_use_conf(llm_type, conf)
    else:
        llm = _create_llm_use_env(llm_type)

    _llm_cache[llm_type] = llm
    return llm


# Initialize LLMs for different purposes - now these will be cached
reasoning_llm = get_llm_by_type("reasoning")
basic_llm = get_llm_by_type("basic")
vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    # stream = reasoning_llm.stream("what is mcp?")
    # full_response = ""
    # for chunk in stream:
    #     full_response += chunk.content
    # print(full_response)

    print(basic_llm.invoke("Hello"))
    # print(vl_llm.invoke("Hello"))
