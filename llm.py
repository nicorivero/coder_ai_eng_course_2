import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from schemas import LLMConfig, Provider

# Mismo patrón de registro del Módulo 1: proveedor -> (clase cliente, variable con la API key, método de salida
# estructurada). Métodos elegidos para que el truncado y los errores de parseo lleguen a la validación de la cadena.
_REGISTRY: dict[Provider, tuple[type[BaseChatModel], str, str]] = {
    # json_schema de OpenAI usa parse(), que lanza LengthFinishReasonError antes de que la cadena pueda reintentar.
    "openai": (ChatOpenAI, "OPENAI_API_KEY", "function_calling"),
    # Los modelos Claude actuales no admiten tool calling forzado; json_schema usa structured outputs nativo.
    "anthropic": (ChatAnthropic, "ANTHROPIC_API_KEY", "json_schema"),
}


def structured_output_method(provider: Provider) -> str:
    """Devuelve el método de with_structured_output() a usar con el proveedor."""
    return _REGISTRY[provider][2]


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """Crea el chat model de LangChain del proveedor configurado, validando antes que exista su API key."""
    client_cls, api_key_env, _ = _REGISTRY[config.provider]
    if not os.getenv(api_key_env):
        raise ValueError(f"{api_key_env} is not set in the environment or .env file.")

    # max_retries cubre errores de red/rate limit a nivel SDK; los errores de formato se reintentan en la cadena.
    # No se envía temperature: los modelos Claude actuales la rechazan.
    options = {"max_retries": config.max_retries} if config.max_retries is not None else {}
    return client_cls(model=config.model, max_tokens=config.max_tokens, **options)
