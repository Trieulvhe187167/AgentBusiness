"""
Standalone background worker process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import uuid

from app.background_jobs import background_worker_loop
from app.config import settings
from app.database import init_db
from app.scheduled_sync import scheduled_sync_loop

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app.worker")


async def initialize_worker_runtime() -> None:
    settings.ensure_dirs()
    settings.validate_runtime_settings()
    await init_db()

    try:
        from app.embeddings import get_dimension, warm_up_model
        from app.vector_store import vector_store

        dim = get_dimension()
        vector_store.initialize(expected_dim=dim)
        await asyncio.to_thread(warm_up_model)
        logger.info("Worker vector store ready: backend=%s dim=%s", vector_store.backend_name, dim)
    except Exception as err:
        logger.error("Worker runtime initialization failed: %s", err, exc_info=True)
        raise


async def main() -> None:
    await initialize_worker_runtime()
    worker_id = os.environ.get("RAG_BACKGROUND_WORKER_ID") or f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    tasks = [asyncio.create_task(background_worker_loop(worker_id=worker_id))]
    if settings.scheduled_sync_enabled:
        tasks.append(asyncio.create_task(scheduled_sync_loop()))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
