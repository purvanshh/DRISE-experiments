from __future__ import annotations

import logging

from pythonjsonlogger.json import JsonFormatter

from document_intelligence_engine.core.logging import configure_logging, get_logger


def test_get_logger_returns_named_logger() -> None:
    configure_logging()
    logger = get_logger("tests.logger")
    assert logger.name == "tests.logger"
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
