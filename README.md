# coder_ai_eng_course_2

Entrega 2 del Curso AI Engineer de Coderhouse: **Pipeline de Extracción de Entidades Técnicas**.

El pipeline recibe un párrafo de texto sin procesar (una descripción de arquitectura, un log de error, el reporte de un incidente) y devuelve un objeto **validado con Pydantic** con las tecnologías involucradas, el nivel de criticidad y un resumen técnico. Está construido con **LangChain (LCEL)**, funciona con **OpenAI o Anthropic** según la configuración y reintenta automáticamente cuando el LLM devuelve una salida mal formada, incompleta o truncada.

## Ejemplo de salida

Entrada:

> Nuestra API está construida con FastAPI y usa Redis como caché de sesiones. La persistencia se hace en PostgreSQL a través de SQLAlchemy. En los picos de tráfico el pool de conexiones a la base se agota y las respuestas superan los 5 segundos, lo que provoca timeouts en el frontend.

Salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "SQLAlchemy"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API FastAPI con caché en Redis y persistencia en PostgreSQL; el pool de conexiones se agota en picos de tráfico y genera timeouts."
}
```

## Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `schemas.py` | Contrato de salida (`ExtraccionTecnica`, `Criticidad`) y configuración del LLM (`LLMConfig`). |
| `llm.py` | Fábrica del chat model de LangChain según el proveedor configurado. |
| `chain.py` | Prompt, cadena LCEL, validación de `finish_reason`, reintentos y la función `process_text()`. |
| `run_example.py` | Mini-script de prueba con tres casos. |
| `.env.example` | Variables de entorno necesarias. |

## Cómo funciona

```
texto ──► ChatPromptTemplate ──► LLM.with_structured_output(ExtraccionTecnica, include_raw=True)
                                              │
                                              ▼
                                  _ensure_complete (finish_reason + parseo)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                 salida válida                                 InvalidOutputError
                      │                                               │
                      ▼                                               ▼
             ExtraccionTecnica                     .with_retry() vuelve a ejecutar la cadena
                                                   (hasta MAX_ATTEMPTS, con backoff)
```

### 1. Contrato de salida (`schemas.py`)

`ExtraccionTecnica` define la estructura que el LLM debe devolver:

- `tecnologias: list[str]`: no puede estar vacía (`min_length=1`). Un `field_validator` limpia espacios, descarta strings vacíos y elimina duplicados sin distinguir mayúsculas, manteniendo el orden. Si después de limpiar la lista queda vacía, la validación falla.
- `nivel_de_criticidad: Criticidad`: enum con los valores `baja`, `media` y `alta`. Cualquier otro valor es rechazado.
- `resumen_tecnico: str`: con un largo mínimo para evitar resúmenes vacíos o triviales.

Las `description` de cada campo se envían al modelo como parte del schema, así que también funcionan como instrucciones.

`LLMConfig` valida la configuración leída del `.env` (proveedor, modelo, `max_tokens`, `max_retries`), siguiendo el mismo diseño del Módulo 1.

### 2. Cliente LLM (`llm.py`)

Reutiliza el patrón del Módulo 1: un registro `proveedor -> (clase cliente, variable de la API key)`. `build_chat_model()` verifica que la API key exista antes de crear el cliente (`ChatOpenAI` o `ChatAnthropic`) y le pasa `model`, `max_tokens` y `max_retries`. Para cambiar de proveedor alcanza con modificar `LLM_PROVIDER` y `LLM_MODEL` en el `.env`.

### 3. Prompt template (`chain.py`)

Se usa `ChatPromptTemplate` con dos variables:

- `{text}`: el texto de entrada, en el mensaje del usuario.
- `{format_instructions}`: las reglas por campo (qué cuenta como tecnología, criterios para cada nivel de criticidad, idioma y formato del resumen). Se inyectan con `.partial()` desde la constante `FORMAT_INSTRUCTIONS`.

No hay f-strings dentro de la cadena: LangChain resuelve las variables, lo que mantiene el prompt modular y permite cambiar las instrucciones sin tocar la cadena.

### 4. Cadena LCEL (`chain.py`)

```python
structured_model = model.with_structured_output(ExtraccionTecnica, include_raw=True)
chain = PROMPT | structured_model | RunnableLambda(_ensure_complete)
```

`with_structured_output()` hace que el proveedor devuelva datos ajustados al schema (tool calling / JSON schema) y los parsea a `ExtraccionTecnica`. Con `include_raw=True` la salida incluye también el mensaje original del modelo y el error de parseo, si lo hubo, en lugar de lanzarlo directamente.

### 5. Detección de respuestas incompletas (`finish_reason`)

El paso `_ensure_complete` revisa la respuesta cruda **antes** de aceptar el objeto:

1. Lee el motivo de corte (`finish_reason` en OpenAI, `stop_reason` en Anthropic). Si es `length` o `max_tokens`, el modelo se quedó sin tokens y el JSON puede estar cortado: se lanza `InvalidOutputError`.
2. Si hubo error de parseo o validación de Pydantic, o no se obtuvo objeto, también se lanza `InvalidOutputError`.
3. Si todo está bien, devuelve el `ExtraccionTecnica` validado.

### 6. Resiliencia (`.with_retry()`)

```python
chain.with_retry(
    retry_if_exception_type=(InvalidOutputError,),
    stop_after_attempt=MAX_ATTEMPTS,
    wait_exponential_jitter=True,
)
```

Hay dos niveles de reintento separados:

- **Cadena (`with_retry`)**: reintenta solo ante `InvalidOutputError` (JSON mal formado, incompleto, truncado o que no cumple el schema), con backoff exponencial. Errores como una API key inválida no se reintentan porque no se resuelven repitiendo la llamada.
- **SDK (`max_retries`)**: los clientes de OpenAI y Anthropic ya reintentan errores de red, rate limit y errores 5xx.

### 7. Función asíncrona `process_text()`

```python
from chain import process_text

result = await process_text("texto a analizar")
```

- Rechaza textos vacíos.
- Ejecuta la cadena con `.ainvoke()`; la cadena se construye una sola vez y se reutiliza entre llamadas.
- Devuelve el `ExtraccionTecnica` validado o relanza el error si se agotaron los intentos.

### 8. Logs

Con `logging` en nivel `INFO` se puede seguir todo el proceso:

```
INFO chain: Procesando texto (312 caracteres) con anthropic/claude-opus-5-5
INFO chain: Respuesta recibida (finish_reason=tool_use, output_tokens=142)
INFO chain: Salida validada contra el schema ExtraccionTecnica
INFO chain: Extracción exitosa: 4 tecnologías, criticidad alta
```

Cuando un intento falla se ve un `WARNING` con el motivo (truncado o error de schema), seguido de un nuevo intento; si se agotan los intentos queda un `ERROR` con el detalle.

## Instalación y ejecución

Requiere Python 3.11 o superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Completar en `.env` la API key del proveedor elegido:

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | API key del proveedor que se vaya a usar. |
| `LLM_PROVIDER` | `openai` o `anthropic`. |
| `LLM_MODEL` | Modelo del proveedor elegido. |
| `LLM_MAX_TOKENS` | Tope de tokens de salida. |

Ejecutar el mini-script de prueba:

```powershell
python run_example.py
```

Procesa tres casos e imprime el JSON validado de cada uno:

1. **Descripción de arquitectura**: API con FastAPI, Redis y PostgreSQL con un cuello de botella en conexiones.
2. **Log de error**: excepción de `asyncpg` en un worker que consume de Kafka.
3. **Texto ambiguo (prueba de estrés)**: un reporte vago sin tecnologías explícitas ni impacto claro, para observar si el modelo genera una salida válida o si el validador fuerza reintentos.

### Probar la detección de truncado

Para ver en acción la detección de `finish_reason` y los reintentos, bajar `LLM_MAX_TOKENS` a un valor muy chico (por ejemplo `20`) y volver a ejecutar el script. En los logs se verán los `WARNING` de respuesta truncada en cada intento y, al agotarse los intentos, el `ERROR` final con `InvalidOutputError`.
