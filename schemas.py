from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Provider = Literal["openai", "anthropic"]


class Criticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Contrato de salida del pipeline. Las descripciones de los campos se envían al LLM como parte del schema."""

    tecnologias: list[str] = Field(
        min_length=1,
        description="Tecnologías, lenguajes, frameworks, bases de datos o servicios mencionados en el texto.",
    )
    nivel_de_criticidad: Criticidad = Field(
        description="Impacto técnico de lo descripto: baja, media o alta.",
    )
    resumen_tecnico: str = Field(
        min_length=20,
        description="Resumen técnico breve (una o dos oraciones) en español.",
    )

    @field_validator("tecnologias")
    @classmethod
    def normalizar_tecnologias(cls, value: list[str]) -> list[str]:
        # Limpia espacios, descarta vacíos y duplicados (sin distinguir mayúsculas) manteniendo el orden.
        # Si la lista queda vacía falla la validación, lo que dispara un reintento de la cadena.
        vistas: set[str] = set()
        resultado: list[str] = []
        for item in (v.strip() for v in value):
            if item and item.lower() not in vistas:
                vistas.add(item.lower())
                resultado.append(item)
        if not resultado:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return resultado


class LLMConfig(BaseModel):
    provider: Provider
    model: str = Field(min_length=1)
    max_tokens: int = Field(default=1024, gt=0)
    max_retries: int | None = Field(default=None, ge=0)  # None mantiene el default del SDK
