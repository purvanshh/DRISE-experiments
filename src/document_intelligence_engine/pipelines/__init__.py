"""Experiment pipeline abstractions and implementations."""

__all__ = ["BasePipeline", "DRISEPipeline", "LLMOnlyPipeline", "RAGLLMPipeline"]


def __getattr__(name: str):
    if name == "BasePipeline":
        from .base import BasePipeline

        return BasePipeline
    if name == "DRISEPipeline":
        from .drise import DRISEPipeline

        return DRISEPipeline
    if name == "LLMOnlyPipeline":
        from .llm_only import LLMOnlyPipeline

        return LLMOnlyPipeline
    if name == "RAGLLMPipeline":
        from .rag_llm import RAGLLMPipeline

        return RAGLLMPipeline
    raise AttributeError(name)
