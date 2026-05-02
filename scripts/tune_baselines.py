"""Tune experiment baselines on the validation split.

This script supports both real provider-backed runs and offline mock tuning.
When provider credentials are unavailable, it automatically falls back to the
mock backend so prompt and retrieval variants can still be exercised locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.evaluator import Evaluator
from document_intelligence_engine.pipelines.llm_only import LLMOnlyPipeline
from document_intelligence_engine.pipelines.rag_llm import RAGLLMPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LLM-only and RAG baselines on the validation split.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Experiment config path.")
    parser.add_argument("--limit", type=int, default=25, help="Optional document limit for fast tuning.")
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help="Use provider-backed LLM calls when credentials and network are available.",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_tuning.json",
        help="Where to write the tuning summary JSON.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    dataset = load_annotations(experiment["dataset"]["val_annotations"])
    if args.limit > 0:
        dataset = dataset[: args.limit]

    llm_config = dict(experiment.get("llm", {}))
    retrieval_config = dict(experiment.get("retrieval", {}))
    backend = _resolve_backend(str(llm_config.get("backend", "mock")), allow_provider_calls=args.allow_provider_calls)
    llm_config["backend"] = backend
    if backend == "mock":
        retrieval_config["embedder"] = "hashing"

    candidates = _candidate_configs(llm_config, retrieval_config)
    evaluator = Evaluator()
    tuning_report = {
        "backend": backend,
        "sample_count": len(dataset),
        "results": [],
    }

    for candidate in candidates:
        pipeline = _build_pipeline(candidate)
        scores = [_evaluate_document(pipeline, evaluator, document) for document in dataset]
        summary = {
            "name": candidate["name"],
            "system": candidate["system"],
            "config": candidate["config"],
            "field_level_f1": _mean(scores, "field_level_f1"),
            "exact_match": _mean(scores, "exact_match"),
            "schema_valid": _mean(scores, "schema_valid"),
            "hallucination_rate": _mean(scores, "hallucination_rate"),
            "cost_usd": _mean(scores, "cost_usd"),
        }
        tuning_report["results"].append(summary)

    tuning_report["results"].sort(
        key=lambda row: (
            row["system"],
            -row["field_level_f1"],
            -row["exact_match"],
            row["hallucination_rate"],
        )
    )
    tuning_report["best_by_system"] = _best_by_system(tuning_report["results"])

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tuning_report, indent=2), encoding="utf-8")
    print(json.dumps(tuning_report["best_by_system"], indent=2))


def _resolve_backend(configured_backend: str, *, allow_provider_calls: bool) -> str:
    if not allow_provider_calls:
        return "mock"
    if configured_backend == "nvidia" and os.getenv("NVIDIA_API_KEY"):
        return configured_backend
    if configured_backend in {"openai", "openai_compatible"} and os.getenv("OPENAI_API_KEY"):
        return configured_backend
    return "mock"


def _candidate_configs(llm_config: dict[str, Any], retrieval_config: dict[str, Any]) -> list[dict[str, Any]]:
    llm_variants = [
        {
            "name": "llm_only_strict_1500",
            "system": "llm_only",
            "config": {
                **llm_config,
                "prompt_variant": "strict_v1",
                "max_input_tokens": 1500,
            },
        },
        {
            "name": "llm_only_strict_2200",
            "system": "llm_only",
            "config": {
                **llm_config,
                "prompt_variant": "strict_v1",
                "max_input_tokens": 2200,
            },
        },
        {
            "name": "llm_only_baseline_1500",
            "system": "llm_only",
            "config": {
                **llm_config,
                "prompt_variant": "baseline",
                "max_input_tokens": 1500,
            },
        },
    ]
    rag_variants = [
        {
            "name": "rag_strict_k3_c500",
            "system": "rag_llm",
            "config": {
                **llm_config,
                **retrieval_config,
                "prompt_variant": "strict_v1",
                "top_k": 3,
                "chunk_size": 500,
            },
        },
        {
            "name": "rag_strict_k5_c300",
            "system": "rag_llm",
            "config": {
                **llm_config,
                **retrieval_config,
                "prompt_variant": "strict_v1",
                "top_k": 5,
                "chunk_size": 300,
            },
        },
        {
            "name": "rag_baseline_k3_c500",
            "system": "rag_llm",
            "config": {
                **llm_config,
                **retrieval_config,
                "prompt_variant": "baseline",
                "top_k": 3,
                "chunk_size": 500,
            },
        },
    ]
    return llm_variants + rag_variants


def _build_pipeline(candidate: dict[str, Any]) -> Any:
    if candidate["system"] == "llm_only":
        return LLMOnlyPipeline(candidate["config"])
    if candidate["system"] == "rag_llm":
        return RAGLLMPipeline(candidate["config"])
    raise ValueError(f"Unsupported tuning system: {candidate['system']}")


def _evaluate_document(pipeline: Any, evaluator: Evaluator, document: dict[str, Any]) -> dict[str, float]:
    output = pipeline.run(document)
    return evaluator.evaluate(output, document.get("ground_truth"), source_text=str(document.get("ocr_text", "")))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 6)


def _best_by_system(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        system = str(row["system"])
        current = best.get(system)
        if current is None or (
            row["field_level_f1"],
            row["exact_match"],
            -row["hallucination_rate"],
        ) > (
            current["field_level_f1"],
            current["exact_match"],
            -current["hallucination_rate"],
        ):
            best[system] = row
    return best


if __name__ == "__main__":
    main()
