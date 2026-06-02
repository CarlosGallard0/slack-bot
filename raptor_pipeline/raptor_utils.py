import uuid
import asyncio
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

# =====================================================
# ID GENERATION
# =====================================================


def generate_node_id(level: int, index: int) -> str:
    """
    Genera un ID único y legible para un nodo RAPTOR
    """
    return f"L{level}_N{index}_{uuid.uuid4().hex[:8]}"


# =====================================================
# ASYNC HELPERS
# =====================================================


def safe_async_run(coro: Coroutine) -> Any:
    """
    Ejecuta corrutinas de forma segura tanto en scripts
    como en entornos con event loop activo (Jupyter).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.create_task(coro)


# =====================================================
# TEXT HELPERS
# =====================================================


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """
    Limita texto a un tamaño seguro para LLMs
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# =====================================================
# LOGGING HELPERS
# =====================================================


def log_banner(logger: logging.Logger, title: str) -> None:
    """
    Imprime un banner visual en logs
    """
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
