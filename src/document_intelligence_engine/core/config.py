"""Centralized application configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
ENV_PREFIX = "DIE_"


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    annotations_dir: Path
    experiment_dir: Path
    artifact_dir: Path
    upload_dir: Path


class APIConfig(BaseModel):
    host: str
    port: int
    reload: bool
    workers: int
    cors_origins: list[str]
    max_upload_size_mb: int
    request_id_header: str = "X-Request-ID"
    rate_limit_per_minute: int = 120
    max_batch_files: int = 10
    batch_concurrency: int = 4


class IngestionConfig(BaseModel):
    max_file_size_mb: int
    pdf_dpi: int
    supported_extensions: list[str]


class OCRConfig(BaseModel):
    backend: str
    language: str
    min_confidence: float
    use_angle_cls: bool = True
    use_gpu: bool = False
    det_limit_side_len: int = 960
    singleton_enabled: bool = True
    batch_size: int = 4
    paddleocr_model_dir: str | None = None
    tesseract_cmd: str | None = None


class ModelConfig(BaseModel):
    layoutlmv3_model_name: str
    version: str = "0.1.0"
    revision: str
    device: str
    max_sequence_length: int
    batch_size: int
    checkpoint_path: str | None = None
    startup_validate_checkpoint: bool = False
    cpu_fallback: bool = True
    use_heuristic_fallback: bool = True


class TrainingConfig(BaseModel):
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_train_samples: int | None = None
    save_dir: str = "experiments/artifacts/cord_finetuned"
    eval_every_n_epochs: int = 1


class PreprocessingConfig(BaseModel):
    target_dpi: int
    max_image_side: int
    grayscale: bool
    resize_max_width: int = 1600
    resize_max_height: int = 1600
    normalize_pixels: bool = True
    noise_reduction: bool = True
    deskew: bool = False
    median_blur_kernel_size: int = 3


class PostprocessingNormalizationConfig(BaseModel):
    date_fields: list[str]
    currency_fields: list[str]
    artifact_map: dict[str, str]


class PostprocessingValidationConfig(BaseModel):
    regex_rules: dict[str, str]
    required_fields: list[str]


class PostprocessingConstraintsConfig(BaseModel):
    amount_tolerance: float
    allow_future_dates: bool
    required_fields: list[str]


class PostprocessingConfidenceConfig(BaseModel):
    min_field_confidence: float
    drop_below_threshold: bool


class PostprocessingConfig(BaseModel):
    field_aliases: dict[str, str]
    normalization: PostprocessingNormalizationConfig
    validation: PostprocessingValidationConfig
    constraints: PostprocessingConstraintsConfig
    confidence: PostprocessingConfidenceConfig


class EvaluationTrackingConfig(BaseModel):
    enabled: bool
    uri: str | None = None
    experiment_name: str = "document-intelligence-benchmarks"


class EvaluationConfig(BaseModel):
    dataset_path: str
    output_dir: str
    partial_match_threshold: float
    confidence_thresholds: list[float]
    error_example_limit: int
    tracking: EvaluationTrackingConfig


class PerformanceConfig(BaseModel):
    enable_profiling: bool = True
    page_batch_size: int = 4
    max_debug_tokens: int = 200
    memory_cleanup_enabled: bool = True


class LoggingConfig(BaseModel):
    level: str
    json: bool
    service_name: str

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log level")
        return normalized


class SecurityConfig(BaseModel):
    allowed_extensions: list[str]
    allowed_content_types: list[str]
    max_pdf_pages: int
    max_image_pixels: int
    filename_pattern: str = Field(
        default=r"[^A-Za-z0-9._-]",
        description="Pattern replaced during filename sanitization.",
    )


class AppSettings(BaseModel):
    project_name: str
    environment: str
    debug: bool
    paths: PathsConfig
    api: APIConfig
    ingestion: IngestionConfig
    ocr: OCRConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig
    evaluation: EvaluationConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    security: SecurityConfig


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_pointer:
        return yaml.safe_load(file_pointer) or {}


def _coerce_env_value(raw_value: str, current_value: Any) -> Any:
    if isinstance(current_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, list):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(current_value, Path):
        return Path(raw_value)
    return raw_value


def _apply_env_overrides(config: dict[str, Any], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        path_parts = env_key[len(prefix) :].lower().split("__")
        target = config
        for part in path_parts[:-1]:
            target = target.setdefault(part, {})

        final_key = path_parts[-1]
        current_value = target.get(final_key, raw_value)
        target[final_key] = _coerce_env_value(raw_value, current_value)

    return config


def _resolve_paths(settings: AppSettings) -> AppSettings:
    for field_name, value in settings.paths.model_dump().items():
        path_value = Path(value)
        resolved = path_value if path_value.is_absolute() else ROOT_DIR / path_value
        setattr(settings.paths, field_name, resolved)
        resolved.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> AppSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    config_data = _apply_env_overrides(_load_yaml_config(path))
    settings = AppSettings.model_validate(config_data)
    return _resolve_paths(settings)
