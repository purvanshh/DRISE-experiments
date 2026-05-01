from __future__ import annotations

from pathlib import Path

from document_intelligence_engine.core.config import AppSettings, get_settings


def test_load_settings_from_yaml() -> None:
    settings = get_settings(Path("configs/config.yaml"))
    assert isinstance(settings, AppSettings)
    assert settings.api.port == 8000
    assert settings.model.layoutlmv3_model_name == "microsoft/layoutlmv3-base"
