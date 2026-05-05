from __future__ import annotations

import logging

from pythonjsonlogger.json import JsonFormatter

from document_intelligence_engine.core.logging import configure_logging, get_logger


def test_get_logger_returns_named_logger() -> None:
    from document_intelligence_engine.core.config import get_settings
    settings = get_settings()
    original_json = settings.logging.json
    try:
        settings.logging.json = True
        configure_logging()
        logger = get_logger("tests.logger")
        assert logger.name == "tests.logger"
        assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
    finally:
        settings.logging.json = original_json
        configure_logging()
