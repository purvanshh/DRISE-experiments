"""Run DRISE baseline experiments from a single YAML config."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_intelligence_engine.data.annotation_loader import load_annotations
from document_intelligence_engine.evaluation.report import generate_experiment_report
from document_intelligence_engine.evaluation.runner import ExperimentRunner
from document_intelligence_engine.pipelines.drise import DRISEPipeline
from document_intelligence_engine.pipelines.llm_only import LLMOnlyPipeline
from document_intelligence_engine.pipelines.rag_llm import RAGLLMPipeline


def run_experiments(config_path: str = "configs/experiments.yaml") -> dict[str, Any]:
    config = _load_config(config_path)
    experiment = config["experiment"]
    _set_random_seeds(int(experiment.get("seed", 42)))
    dataset = load_annotations(experiment["dataset"]["test_annotations"])
    systems = _build_systems(experiment)
    runner = ExperimentRunner(
        systems,
        results_dir=experiment.get("output_dir", "experiments/results"),
        resume=bool(experiment.get("resume", True)),
        max_total_cost_usd=float(experiment.get("cost_cap_usd", 30.0)),
    )
    results = runner.run(dataset)
    report = generate_experiment_report(results, output_dir=experiment.get("output_dir", "experiments/results"))

    report_path = Path(experiment.get("output_dir", "experiments/results")).expanduser().resolve() / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_systems(experiment: dict[str, Any]) -> list[Any]:
    system_names = list(experiment.get("systems", []))
    llm_config = dict(experiment.get("llm", {}))
    retrieval_config = dict(experiment.get("retrieval", {}))
    output_dir = str(experiment.get("output_dir", "experiments/results"))

    systems: list[Any] = []
    if "llm_only" in system_names:
        systems.append(LLMOnlyPipeline(llm_config))
    if "rag_llm" in system_names:
        systems.append(RAGLLMPipeline({**llm_config, **retrieval_config}))
    if "drise" in system_names:
        drise_config = {**experiment.get("drise", {}), "results_dir": output_dir, "name": "drise"}
        systems.append(DRISEPipeline(drise_config))
        systems.extend(_build_drise_ablations(experiment, output_dir))
    return systems


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_drise_ablations(experiment: dict[str, Any], output_dir: str) -> list[Any]:
    ablation_config = dict(experiment.get("ablation", {}))
    if not ablation_config.get("enabled", False):
        return []

    drise_defaults = dict(experiment.get("drise", {}))
    ablations: list[Any] = []
    variants = ablation_config.get("variants", {})
    for variant_name, overrides in variants.items():
        if not isinstance(overrides, dict):
            continue
        ablations.append(
            DRISEPipeline(
                {
                    **drise_defaults,
                    **overrides,
                    "results_dir": output_dir,
                    "name": variant_name,
                }
            )
        )
    return ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled DRISE extraction experiments.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to the experiment YAML config.")
    args = parser.parse_args()
    report = run_experiments(args.config)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
