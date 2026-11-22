# coder_ai_eng_course_2

Entrega 2 del Curso AI Engineer de Coderhouse: **Pipeline de Extracción de Entidades Técnicas**.

El pipeline recibe un párrafo de texto sin procesar (una descripción de arquitectura, un log de error, el reporte de un incidente) y devuelve un objeto **validado con Pydantic** con las tecnologías involucradas, el nivel de criticidad y un resumen técnico. Está construido con **LangChain (LCEL)**, funciona con **OpenAI o Anthropic** según la configuración y reintenta automáticamente cuando el LLM devuelve una salida mal formada, incompleta o truncada.

## Ejemplo de salida

Entrada:

> Nuestra API está construida con FastAPI y usa Redis como caché de sesiones. La persistencia se hace en PostgreSQL a través de SQLAlchemy. En los picos de tráfico el pool de conexiones a la base se agota y las respuestas superan los 5 segundos, lo que provoca timeouts en el frontend.

Salida real (OpenAI, `gpt-6.1-terra`):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "SQLAlchemy"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "En los picos de tráfico se agota el pool de conexiones a PostgreSQL, lo que eleva los tiempos de respuesta de la API FastAPI por encima de cinco segundos y provoca timeouts en el frontend."
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
structured_model = model.with_structured_output(
    ExtraccionTecnica, include_raw=True, method=_STRUCTURED_OUTPUT_METHOD[config.provider]
)
chain = PROMPT | structured_model | RunnableLambda(_ensure_complete)
```

`with_structured_output()` hace que el proveedor devuelva datos ajustados al schema y los parsea a `ExtraccionTecnica`. Con `include_raw=True` la salida incluye también el mensaje original del modelo y el error de parseo, si lo hubo, en lugar de lanzarlo directamente.

El método de salida estructurada se elige por proveedor:

| Proveedor | Método | Motivo |
|---|---|---|
| OpenAI | `function_calling` | El método `json_schema` usa `parse()` del SDK, que lanza `LengthFinishReasonError` ante un truncado antes de que la cadena pueda inspeccionarlo y reintentar. Con tool calling (no estricto) el `finish_reason` llega intacto. |
| Anthropic | `json_schema` | Structured outputs nativo. Los modelos Claude actuales no admiten tool calling forzado, y en ese modo LangChain lanza la excepción dentro de la llamada al modelo, sin pasar por la validación ni el reintento. |

En ambos casos los errores de truncado y de parseo llegan al paso de validación, que decide si reintentar.

### 5. Detección de respuestas incompletas (`finish_reason`)

El paso `_ensure_complete` revisa la respuesta cruda **antes** de aceptar el objeto:

1. Lee el motivo de corte. Usa primero `normalized_finish_reason`, el campo que LangChain unifica para todos los proveedores (la Responses API de OpenAI, por ejemplo, reporta `finish_reason="incomplete"`), y como respaldo `finish_reason` (OpenAI) o `stop_reason` (Anthropic). Si es `length`, `max_tokens` o `context_exceeded`, la respuesta se cortó y el JSON puede estar incompleto: se lanza `InvalidOutputError`.
2. Si el motivo es `refusal` (el modelo se negó a responder), se lanza `RefusedOutputError`.
3. Si hubo error de parseo o validación de Pydantic, o no se obtuvo objeto, se lanza `InvalidOutputError`. Cuando no hay error de parseo pero tampoco objeto, el mensaje indica el motivo de corte para facilitar el diagnóstico.
4. Si todo está bien, devuelve el `ExtraccionTecnica` validado.

Gracias al método elegido por proveedor (sección anterior), el motivo de corte llega a este paso tanto en OpenAI como en Anthropic, en lugar de convertirse en una excepción del SDK.

### 6. Resiliencia (`.with_retry()`)

```python
chain.with_retry(
    retry_if_exception_type=(InvalidOutputError,),
    stop_after_attempt=MAX_ATTEMPTS,
    wait_exponential_jitter=True,
)
```

Hay dos niveles de reintento separados:

- **Cadena (`with_retry`)**: reintenta solo ante `InvalidOutputError` (JSON mal formado, incompleto, truncado o que no cumple el schema), con backoff exponencial. Errores como una API key inválida no se reintentan porque no se resuelven repitiendo la llamada. Tampoco se reintentan las negativas del modelo (`RefusedOutputError`): repetir el mismo pedido da el mismo resultado y solo gasta tokens.
- **SDK (`max_retries`)**: los clientes de OpenAI y Anthropic ya reintentan errores de red, rate limit y errores 5xx.

### 7. Función asíncrona `process_text()`

```python
from chain import process_text

result = await process_text("texto a analizar")
```

- Rechaza textos vacíos.
- Ejecuta la cadena con `.ainvoke()`; la cadena se construye una sola vez y se reutiliza entre llamadas.
- Devuelve el `ExtraccionTecnica` validado o relanza el error si se agotaron los intentos, si el modelo se negó a responder o si la configuración es inválida (en todos los casos queda registrado en los logs).

### 8. Logs

Con `logging` en nivel `INFO` se puede seguir todo el proceso (salida real del primer caso):

```
INFO chain: Procesando texto (283 caracteres) con openai/gpt-6.1-terra
INFO chain: Respuesta recibida (finish_reason=tool_call, output_tokens=114)
INFO chain: Salida validada contra el schema ExtraccionTecnica
INFO chain: Extracción exitosa: 4 tecnologías, criticidad media
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
| `LLM_PROVIDER` | `openai` o `anthropic` (sin distinguir mayúsculas). |
| `LLM_MODEL` | Modelo del proveedor elegido. |
| `LLM_MAX_TOKENS` | Tope de tokens de salida; si se deja vacío se usa el valor por defecto de `LLMConfig`. |

Si falta una variable obligatoria o tiene un valor inválido, `process_text` registra el error en los logs y lo relanza.

Ejecutar el mini-script de prueba:

```powershell
python run_example.py
```

Procesa tres casos e imprime el JSON validado de cada uno:

1. **Descripción de arquitectura**: API con FastAPI, Redis y PostgreSQL con un cuello de botella en conexiones.
2. **Log de error**: excepción de `asyncpg` en un worker que consume de Kafka.
3. **Texto ambiguo (prueba de estrés)**: un reporte vago sin tecnologías explícitas ni impacto claro, para observar si el modelo genera una salida válida o si el validador fuerza reintentos.

Resultados reales de los casos 2 y 3 (OpenAI, `gpt-6.1-terra`; el caso 1 está en [Ejemplo de salida](#ejemplo-de-salida)). Los tres se validaron en el primer intento:

```json
{
  "tecnologias": ["asyncpg", "Kafka"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "El consumidor de pagos falló al insertar un registro por una violación de clave única. Se agotaron cinco reintentos y, al no confirmarse el offset de Kafka, el mensaje se volverá a entregar."
}
```

```json
{
  "tecnologias": ["base de datos", "red"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "El sistema presenta lentitud intermitente por la mañana, que se resuelve sola. Se sospecha de la base de datos o la red, pero la causa no está confirmada."
}
```

En el texto ambiguo, como no hay tecnologías explícitas, el modelo devolvió categorías técnicas inferidas (según las instrucciones del prompt) en lugar de una lista vacía, y el resumen deja claro que la causa no está confirmada.

### Probar la detección de truncado

Para ver en acción la detección de `finish_reason` y los reintentos, bajar `LLM_MAX_TOKENS` a un valor muy chico (por ejemplo `20`) y volver a ejecutar el script. Salida real con `LLM_MAX_TOKENS=20` sobre el log de error:

```
INFO chain: Procesando texto (405 caracteres) con openai/gpt-6.1-terra
INFO chain: Respuesta recibida (finish_reason=length, output_tokens=20)
WARNING chain: Respuesta truncada por límite de tokens; se descarta el intento
INFO chain: Respuesta recibida (finish_reason=length, output_tokens=20)
WARNING chain: Respuesta truncada por límite de tokens; se descarta el intento
INFO chain: Respuesta recibida (finish_reason=length, output_tokens=20)
WARNING chain: Respuesta truncada por límite de tokens; se descarta el intento
ERROR chain: Extracción fallida tras 3 intentos: respuesta truncada (finish_reason=length)
```

Cada intento truncado se descarta antes de parsear, la cadena reintenta con backoff y, al agotarse los intentos, `process_text` relanza `InvalidOutputError`.
