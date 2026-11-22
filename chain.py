import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from llm import build_chat_model
from schemas import ExtraccionTecnica, LLMConfig

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

# Motivos de corte que indican respuesta truncada: normalizados por LangChain ("length", "context_exceeded")
# y crudos del proveedor como respaldo (Chat Completions: "length", Anthropic: "max_tokens").
_TRUNCATION_REASONS = {"length", "context_exceeded", "max_tokens"}

# Métodos con los que el truncado y los errores de parseo llegan a _ensure_complete en vez de lanzarse dentro
# del modelo (json_schema de OpenAI lanza LengthFinishReasonError; tool calling forzado no anda en Claude actual).
_STRUCTURED_OUTPUT_METHOD = {"openai": "function_calling", "anthropic": "json_schema"}

FORMAT_INSTRUCTIONS = """\
- tecnologias: nombres de tecnologías, lenguajes, frameworks, bases de datos, colas, servicios cloud o \
herramientas mencionados en el texto, escritos con su nombre propio (ej. "PostgreSQL", "FastAPI"). \
Si el texto no nombra ninguna de forma explícita, incluí la categoría técnica más concreta que se pueda \
inferir (ej. "base de datos relacional"). Nunca devuelvas la lista vacía.
- nivel_de_criticidad:
  - alta: caída de servicio, pérdida o corrupción de datos, fallas de seguridad o impacto directo en producción.
  - media: degradación de rendimiento, errores parciales o intermitentes, riesgos técnicos a corto plazo.
  - baja: descripciones informativas, mejoras o situaciones sin impacto inmediato.
- resumen_tecnico: una o dos oraciones en español que indiquen el componente involucrado y el problema \
o la característica principal."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software senior que extrae entidades técnicas de textos sin procesar "
            "(descripciones de arquitectura, logs de error, incidentes).\n"
            "Seguí estas reglas para cada campo:\n{format_instructions}",
        ),
        ("human", "Texto a analizar:\n\n{text}"),
    ]
).partial(format_instructions=FORMAT_INSTRUCTIONS)


class InvalidOutputError(Exception):
    """Salida del LLM truncada, mal formada o que no cumple el schema; es la única excepción que se reintenta."""


def _ensure_complete(result: dict[str, Any]) -> ExtraccionTecnica:
    """Valida la salida cruda del modelo antes de aceptarla: detecta truncado por tokens y errores de parseo."""
    raw = result["raw"]
    metadata = raw.response_metadata
    # El campo normalizado unifica proveedores y APIs (la Responses API de OpenAI reporta finish_reason="incomplete").
    finish_reason = (
        metadata.get("normalized_finish_reason") or metadata.get("finish_reason") or metadata.get("stop_reason")
    )
    usage = raw.usage_metadata or {}
    logger.info("Respuesta recibida (finish_reason=%s, output_tokens=%s)", finish_reason, usage.get("output_tokens"))

    # Se chequea antes del parseo: un JSON cortado puede fallar con un error engañoso o venir incompleto.
    if finish_reason in _TRUNCATION_REASONS:
        logger.warning("Respuesta truncada por límite de tokens; se descarta el intento")
        raise InvalidOutputError(f"respuesta truncada (finish_reason={finish_reason})")

    if result["parsing_error"] is not None or result["parsed"] is None:
        logger.warning("La salida no cumple el schema: %s", result["parsing_error"])
        raise InvalidOutputError(f"salida inválida: {result['parsing_error']}") from result["parsing_error"]

    logger.info("Salida validada contra el schema %s", ExtraccionTecnica.__name__)
    return result["parsed"]


def build_chain(config: LLMConfig) -> Runnable:
    """Arma la cadena LCEL: prompt | modelo con salida estructurada | validación, con reintento automático."""
    model = build_chat_model(config)
    # include_raw=True devuelve el AIMessage original (para leer finish_reason) y captura errores de parseo.
    structured_model = model.with_structured_output(
        ExtraccionTecnica, include_raw=True, method=_STRUCTURED_OUTPUT_METHOD[config.provider]
    )
    chain = PROMPT | structured_model | RunnableLambda(_ensure_complete)
    # Solo se reintentan salidas inválidas; los errores de red/API ya los reintenta el SDK (max_retries).
    return chain.with_retry(
        retry_if_exception_type=(InvalidOutputError,),
        stop_after_attempt=MAX_ATTEMPTS,
        wait_exponential_jitter=True,
    )


@lru_cache(maxsize=1)
def _load_config() -> LLMConfig:
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER"),
        model=os.getenv("LLM_MODEL"),
        max_tokens=os.getenv("LLM_MAX_TOKENS", "1024"),
    )


# Cacheada para crear el cliente y la cadena una sola vez y reutilizarlos entre llamadas.
@lru_cache(maxsize=1)
def _get_chain() -> Runnable:
    return build_chain(_load_config())


async def process_text(text: str) -> ExtraccionTecnica:
    """Extrae entidades técnicas de un texto libre y devuelve el objeto validado; relanza el error si se agotan los intentos."""
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    config = _load_config()
    logger.info("Procesando texto (%d caracteres) con %s/%s", len(text), config.provider, config.model)
    try:
        result = await _get_chain().ainvoke({"text": text})
    except InvalidOutputError as e:
        logger.error("Extracción fallida tras %d intentos: %s", MAX_ATTEMPTS, e)
        raise
    except Exception:
        logger.exception("Error no recuperable durante la extracción")
        raise

    logger.info("Extracción exitosa: %d tecnologías, criticidad %s", len(result.tecnologias), result.nivel_de_criticidad)
    return result
