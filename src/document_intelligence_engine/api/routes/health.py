"""Health routes."""

from __future__ import annotations

from fastapi import APIRouter

from document_intelligence_engine.api.schemas.health import HealthResponse
from document_intelligence_engine.core.config import get_settings


router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.project_name)
