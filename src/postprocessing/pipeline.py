"""Deterministic post-processing pipeline."""

from __future__ import annotations

from typing import Any

from document_intelligence_engine.core.config import get_settings

from postprocessing.confidence import apply_confidence_policy
from postprocessing.constraints import apply_constraints as apply_constraint_rules
from postprocessing.entity_grouping import group_entities
from postprocessing.normalization import normalize_entities
from postprocessing.recovery import recover_missing_entities
from postprocessing.validation import validate_fields


def postprocess_predictions(
    predictions: list[dict[str, Any]],
    *,
    apply_constraints: bool = True,
    repair_constraints: bool = True,
    ocr_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = get_settings()

    grouped_entities, grouping_errors = group_entities(
        predictions=predictions,
        field_aliases=settings.postprocessing.field_aliases,
    )
    recovered_entities = recover_missing_entities(
        grouped_entities,
        ocr_tokens,
        settings.postprocessing.field_aliases,
    )
    normalized_entities, normalization_errors = normalize_entities(recovered_entities, settings)
    validated_document, validation_errors = validate_fields(normalized_entities, settings)
    constraint_errors: list[dict[str, Any]] = []
    constraint_flags: list[str] = []
    constrained_document = validated_document
    if apply_constraints:
        constrained_document, constraint_errors, constraint_flags = apply_constraint_rules(
            validated_document,
            settings,
            repair=repair_constraints,
        )
    final_document, confidence_errors = apply_confidence_policy(constrained_document, settings)

    final_document["_errors"] = (
        grouping_errors
        + normalization_errors
        + validation_errors
        + constraint_errors
        + confidence_errors
    )
    final_document["_constraint_flags"] = constraint_flags
    return final_document
