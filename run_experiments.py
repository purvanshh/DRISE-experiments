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
    system_specs = list(experiment.get("systems", []))
    output_dir = str(experiment.get("output_dir", "experiments/results"))

    systems: list[Any] = []
    for system_spec in system_specs:
        pipeline_name, config = _resolve_system_config(system_spec, experiment, output_dir)
        if pipeline_name == "llm_only":
            systems.append(LLMOnlyPipeline(config))
        elif pipeline_name == "rag_llm":
            systems.append(RAGLLMPipeline(config))
        elif pipeline_name == "drise":
            systems.append(DRISEPipeline(config))
            systems.extend(_build_drise_ablations(experiment, output_dir))
    return systems


def _resolve_system_config(system_spec: Any, experiment: dict[str, Any], output_dir: str) -> tuple[str, dict[str, Any]]:
    default_llm_config = dict(experiment.get("llm", {}))
    default_retrieval_config = dict(experiment.get("retrieval", {}))

    if isinstance(system_spec, str):
        pipeline_name = system_spec
        system_name = system_spec
        config_ref = None
        overrides: dict[str, Any] = {}
    elif isinstance(system_spec, dict):
        pipeline_name = str(system_spec.get("pipeline") or system_spec.get("system") or "")
        if not pipeline_name:
            raise ValueError("System config entries must define 'pipeline' or 'system'.")
        system_name = str(system_spec.get("name", pipeline_name))
        config_ref = system_spec.get("config")
        overrides = dict(system_spec.get("overrides", {}))
    else:
        raise ValueError(f"Unsupported system specification: {system_spec!r}")

    llm_config = dict(default_llm_config)
    if config_ref:
        llm_config.update(dict(experiment.get(str(config_ref), {})))

    if pipeline_name == "llm_only":
        return pipeline_name, {**llm_config, **overrides, "name": system_name}
    if pipeline_name == "rag_llm":
        return pipeline_name, {**llm_config, **default_retrieval_config, **overrides, "name": system_name}
    if pipeline_name == "drise":
        return pipeline_name, {**experiment.get("drise", {}), **overrides, "results_dir": output_dir, "name": system_name}
    raise ValueError(f"Unsupported system name: {pipeline_name}")


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
