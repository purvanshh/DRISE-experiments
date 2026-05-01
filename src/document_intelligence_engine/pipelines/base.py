"""Base classes for experiment extraction pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.experiment_models import ExtractionOutput, ProcessedDocument


class BasePipeline(ABC):
    """Abstract base class for experiment extraction systems."""

    name = "base"

    @abstractmethod
    def run(self, document: ProcessedDocument) -> ExtractionOutput:
        """Extract fields from a processed document."""
