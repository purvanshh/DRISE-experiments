"""Sensitivity and robustness experiments for DRISE baselines."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.evaluator import Evaluator
from document_intelligence_engine.pipelines.drise import DRISEPipeline
from document_intelligence_engine.pipelines.llm_only import LLMOnlyPipeline
from document_intelligence_engine.pipelines.rag_llm import RAGLLMPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temperature and OCR-noise sensitivity sweeps.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/results/sensitivity")
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help="Use provider-backed LLM calls when credentials and network are available.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    sensitivity = dict(experiment.get("sensitivity", {}))
    dataset = load_annotations(experiment["dataset"]["test_annotations"])[: args.limit]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _resolve_backend(
        str(experiment.get("llm", {}).get("backend", "mock")),
        allow_provider_calls=args.allow_provider_calls,
    )
    llm_config = {
        **dict(experiment.get("llm", {})),
        "backend": backend,
    }
    retrieval_config = dict(experiment.get("retrieval", {}))
    if backend == "mock":
        retrieval_config["embedder"] = "hashing"

    evaluator = Evaluator()
    temperature_rows = _run_temperature_sweep(
        dataset,
        evaluator,
        llm_config=llm_config,
        retrieval_config=retrieval_config,
        temperatures=list(sensitivity.get("temperatures", [0.0, 0.3, 0.7])),
    )
    noise_rows = _run_noise_sweep(
        dataset,
        evaluator,
        llm_config=llm_config,
        retrieval_config=retrieval_config,
        drise_config=dict(experiment.get("drise", {})),
        noise_factors=list(sensitivity.get("ocr_noise_factors", [0.0, 0.1, 0.2])),
    )

    payload = {
        "backend": backend,
        "sample_count": len(dataset),
        "temperature_sweep": temperature_rows,
        "noise_sweep": noise_rows,
    }
    (output_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "analysis.md").write_text(_render_markdown(payload), encoding="utf-8")
    _write_temperature_chart(temperature_rows, output_dir / "temperature_sweep.png")
    _write_noise_chart(noise_rows, output_dir / "ocr_noise_sweep.png")
    print(json.dumps({"output_dir": str(output_dir), "backend": backend}, indent=2))


def _run_temperature_sweep(
    dataset: list[dict[str, Any]],
    evaluator: Evaluator,
    *,
    llm_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    temperatures: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        systems = {
            "llm_only": LLMOnlyPipeline({**llm_config, "temperature": temperature, "prompt_variant": "strict_v1"}),
            "rag_llm": RAGLLMPipeline(
                {**llm_config, **retrieval_config, "temperature": temperature, "prompt_variant": "strict_v1"}
            ),
        }
        for system_name, pipeline in systems.items():
            scores = [
                evaluator.evaluate(pipeline.run(document), document.get("ground_truth"), source_text=str(document.get("ocr_text", "")))
                for document in dataset
            ]
            rows.append(
                {
                    "system": system_name,
                    "temperature": temperature,
                    "field_level_f1": _mean(scores, "field_level_f1"),
                    "exact_match": _mean(scores, "exact_match"),
                    "schema_valid": _mean(scores, "schema_valid"),
                }
            )
    return rows


def _run_noise_sweep(
    dataset: list[dict[str, Any]],
    evaluator: Evaluator,
    *,
    llm_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    drise_config: dict[str, Any],
    noise_factors: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    systems = {
        "llm_only": LLMOnlyPipeline({**llm_config, "prompt_variant": "strict_v1"}),
        "rag_llm": RAGLLMPipeline({**llm_config, **retrieval_config, "prompt_variant": "strict_v1"}),
        "drise": DRISEPipeline({**drise_config, "name": "drise"}),
    }
    for noise_factor in noise_factors:
        noisy_documents = [_corrupt_document(document, noise_factor) for document in dataset]
        for system_name, pipeline in systems.items():
            scores = [
                evaluator.evaluate(pipeline.run(document), document.get("ground_truth"), source_text=str(document.get("ocr_text", "")))
                for document in noisy_documents
            ]
            rows.append(
                {
                    "system": system_name,
                    "noise_factor": noise_factor,
                    "field_level_f1": _mean(scores, "field_level_f1"),
                    "exact_match": _mean(scores, "exact_match"),
                    "schema_valid": _mean(scores, "schema_valid"),
                }
            )
    return rows


def _corrupt_document(document: dict[str, Any], noise_factor: float) -> dict[str, Any]:
    if noise_factor <= 0.0:
        return document

    rng = random.Random(f"{document.get('doc_id')}::{noise_factor}")
    corrupted = deepcopy(document)
    corrupted_tokens = []
    for token in list(document.get("ocr_tokens", [])):
        new_token = dict(token)
        if rng.random() < noise_factor:
            original_text = str(new_token.get("text", ""))
            new_token["text"] = _corrupt_text(original_text)
            new_token["confidence"] = max(0.0, float(new_token.get("confidence", 0.0)) - 0.3)
        corrupted_tokens.append(new_token)
    corrupted["ocr_tokens"] = corrupted_tokens
    corrupted["ocr_text"] = "\n".join(token.get("text", "") for token in corrupted_tokens if token.get("text"))
    return corrupted


def _corrupt_text(text: str) -> str:
    if not text:
        return text
    stripped = re.sub(r"[aeiouAEIOU]", "", text)
    if stripped and stripped != text:
        return stripped
    if len(text) > 1:
        return text[:-1] + "#"
    return "#"


def _resolve_backend(configured_backend: str, *, allow_provider_calls: bool) -> str:
    if not allow_provider_calls:
        return "mock"
    if configured_backend == "nvidia" and os.getenv("NVIDIA_API_KEY"):
        return configured_backend
    if configured_backend in {"openai", "openai_compatible"} and os.getenv("OPENAI_API_KEY"):
        return configured_backend
    return "mock"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 6)


def _write_temperature_chart(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    for system_name in sorted({row["system"] for row in rows}):
        points = sorted((row for row in rows if row["system"] == system_name), key=lambda row: row["temperature"])
        axis.plot(
            [row["temperature"] for row in points],
            [row["field_level_f1"] for row in points],
            marker="o",
            label=system_name,
        )
    axis.set_title("Baseline Temperature Sensitivity")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Field-level F1")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_noise_chart(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    for system_name in sorted({row["system"] for row in rows}):
        points = sorted((row for row in rows if row["system"] == system_name), key=lambda row: row["noise_factor"])
        axis.plot(
            [row["noise_factor"] for row in points],
            [row["field_level_f1"] for row in points],
            marker="o",
            label=system_name,
        )
    axis.set_title("OCR Noise Robustness")
    axis.set_xlabel("Noise factor")
    axis.set_ylabel("Field-level F1")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Sensitivity Analysis",
        "",
        f"- Backend: `{payload['backend']}`",
        f"- Sample count: `{payload['sample_count']}`",
        "",
        "## Temperature Sweep",
        "",
        "| System | Temperature | Field F1 | Exact Match | Schema Valid |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["temperature_sweep"]:
        lines.append(
            f"| {row['system']} | {row['temperature']:.1f} | {row['field_level_f1']:.3f} | "
            f"{row['exact_match']:.3f} | {row['schema_valid']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## OCR Noise Sweep",
            "",
            "| System | Noise Factor | Field F1 | Exact Match | Schema Valid |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["noise_sweep"]:
        lines.append(
            f"| {row['system']} | {row['noise_factor']:.1f} | {row['field_level_f1']:.3f} | "
            f"{row['exact_match']:.3f} | {row['schema_valid']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
