import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from schemas import LLMConfig, Provider

# Mismo patrón de registro del Módulo 1: proveedor -> (clase cliente, variable de entorno con la API key).
_REGISTRY: dict[Provider, tuple[type[BaseChatModel], str]] = {
    "openai": (ChatOpenAI, "OPENAI_API_KEY"),
    "anthropic": (ChatAnthropic, "ANTHROPIC_API_KEY"),
}


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea el chat model de LangChain del proveedor configurado, validando antes que exista su API key."""
    client_cls, api_key_env = _REGISTRY[config.provider]
    if not os.getenv(api_key_env):
        raise ValueError(f"{api_key_env} is not set in the environment or .env file.")

    # max_retries cubre errores de red/rate limit a nivel SDK; los errores de formato se reintentan en la cadena.
    # No se envía temperature: los modelos Claude actuales la rechazan.
    options = {"max_retries": config.max_retries} if config.max_retries is not None else {}
    return client_cls(model=config.model, max_tokens=config.max_tokens, **options)
