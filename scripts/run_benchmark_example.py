from __future__ import annotations

from evaluation.ablation import run_ablation_study
from evaluation.benchmark import run_benchmark
from evaluation.report import generate_report


class ExampleModel:
    name = "example-layoutlmv3"

    def predict(self, ocr_tokens):
        _ = ocr_tokens
        return [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.95},
            {"text": "Number", "label": "I-KEY", "confidence": 0.94},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.93},
        ]

    def predict_text_only(self, ocr_tokens):
        return self.predict(ocr_tokens)

    def predict_without_postprocessing(self, ocr_tokens):
        return self.predict(ocr_tokens)


class ExamplePipeline:
    def postprocess_predictions(self, predictions):
        from postprocessing.pipeline import postprocess_predictions

        return postprocess_predictions(predictions)


if __name__ == "__main__":
    dataset_path = "data/annotations/benchmark_samples.json"
    benchmark_results = run_benchmark(dataset_path, ExampleModel(), ExamplePipeline())
    ablation_results = run_ablation_study(dataset_path, ExampleModel(), ExamplePipeline())
    report = generate_report(benchmark_results, ablation_results=ablation_results)
    print(report["json_path"])
