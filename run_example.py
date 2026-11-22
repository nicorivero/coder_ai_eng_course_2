import asyncio
import logging

from dotenv import load_dotenv

from chain import process_text

EXAMPLES = {
    "Descripción de arquitectura": (
        "Nuestra API está construida con FastAPI y usa Redis como caché de sesiones. La persistencia se hace "
        "en PostgreSQL a través de SQLAlchemy. En los picos de tráfico el pool de conexiones a la base se agota "
        "y las respuestas superan los 5 segundos, lo que provoca timeouts en el frontend."
    ),
    "Log de error": (
        "2026-11-20 03:14:07 ERROR [payments-worker] Unhandled exception in consumer\n"
        "Traceback (most recent call last):\n"
        '  File "/app/worker.py", line 88, in handle_message\n'
        "    await db.execute(insert_payment, payload)\n"
        "asyncpg.exceptions.UniqueViolationError: duplicate key value violates unique constraint \"payments_pkey\"\n"
        "Kafka offset not committed; message will be redelivered. Retries exhausted after 5 attempts."
    ),
    # Prueba de estrés: texto vago, sin tecnologías explícitas ni impacto claro.
    "Texto ambiguo": (
        "A veces el sistema anda lento a la mañana, pero después se arregla solo. "
        "Alguien dijo que podría ser algo de la base o de la red, no estamos seguros."
    ),
}


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Silencia los logs HTTP de los SDKs para que se vean solo los del pipeline (las versiones actuales usan httpx2).
    for http_logger in ("httpx", "httpx2"):
        logging.getLogger(http_logger).setLevel(logging.WARNING)

    for title, text in EXAMPLES.items():
        print(f"\n=== {title} ===")
        try:
            result = await process_text(text)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
