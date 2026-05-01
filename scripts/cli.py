from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from evaluation.ablation import run_ablation_study
from evaluation.benchmark import run_benchmark
from evaluation.report import generate_report


class PostprocessingPipelineAdapter:
    def postprocess_predictions(self, predictions):
        from postprocessing.pipeline import postprocess_predictions

        return postprocess_predictions(predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Document intelligence CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_document = subparsers.add_parser("parse-document", help="Run the full parsing pipeline.")
    parse_document.add_argument("input_document", help="Path to the input PDF or image.")
    parse_document.add_argument("--debug", action="store_true", help="Include intermediate outputs.")
    parse_document.add_argument("--output", help="Optional output JSON path.")

    run_benchmark_parser = subparsers.add_parser("run-benchmark", help="Run evaluation benchmark.")
    run_benchmark_parser.add_argument(
        "--dataset",
        default=get_settings().evaluation.dataset_path,
        help="Path to benchmark dataset.",
    )
    run_benchmark_parser.add_argument(
        "--report-dir",
        default=get_settings().evaluation.output_dir,
        help="Directory to write benchmark reports.",
    )

    args = parser.parse_args()
    if args.command == "parse-document":
        return _run_parse_document(args.input_document, args.debug, args.output)
    return _run_benchmark(args.dataset, args.report_dir)


def _run_parse_document(input_document: str, debug: bool, output: str | None) -> int:
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)
    result = parser_service.parse_file(input_document, debug=debug)
    payload = json.dumps(result, indent=2)
    if output:
        Path(output).expanduser().resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


def _run_benchmark(dataset: str, report_dir: str) -> int:
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    benchmark_results = run_benchmark(dataset, model_service, PostprocessingPipelineAdapter())
    ablation_results = run_ablation_study(dataset, model_service, PostprocessingPipelineAdapter())
    report = generate_report(benchmark_results, ablation_results=ablation_results, output_dir=report_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
