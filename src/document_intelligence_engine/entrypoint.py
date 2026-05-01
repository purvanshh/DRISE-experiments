"""Project entrypoint."""

from __future__ import annotations

import uvicorn

from document_intelligence_engine.core.config import get_settings


def main() -> int:
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
    )
    return 0
