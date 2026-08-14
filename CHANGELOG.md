# Changelog

## v1.2.0 — fine-tuned CORD checkpoint + safe loader

### Added

- `inference_cord_finetuned.py`: safe model loader, receipt inference, locale-aware number parsing, validation-set benchmarking (token + seqeval entity F1) and a confidence-threshold sweep CLI.
- `set_global_threshold()` / `get_global_threshold()` for runtime-adjustable confidence filtering (defaults are resolved at call time).
- `scripts/benchmark_cord_finetuned.py`: token-level benchmark + threshold sweep on the cached `katanaml/cord` test split.
- `LayoutLMv3InferenceService` now loads local checkpoints through the safe loader, falling back to the base processor when `preprocessor_config.json` is missing (transformers-5.x-style checkpoints on a transformers-4.x runtime).

### Benchmark

Fine-tuned CORD checkpoint (`Drise Cord Fine-tuned Checkpoint/`, 5-class BIO, FUNSD-mixed): **0.8576 token-level F1** on the CORD test split (P 0.7513 / R 0.9989), matching the published `0.868` token-F1 ballpark. Note the CORD entity-level F1 is not meaningful on `katanaml/cord` because the model predicts KEY spans that this split does not annotate.

## v1.1.0 — extraction improvements & honest metrics

### Added

- Semantic token categories propagated from LayoutLMv3 through to entity grouping so multi-word field values are preserved.
- FUNSD QUESTION/ANSWER (KEY/VALUE) supervision for fine-tuning via `--include-funsd`, plus support for both CORD dataset formats and token-rendered image fallback.
- Conditional per-field F1 and per-field exact-match contribution reported in experiment summaries.
- Constraint repair that derives a missing line-item quantity from the total when reconciliation is exact.
- `scripts/eval_drise.py` evaluation harness (masked F1, exact match, per-field breakdowns).
- Ground-truth line-item normalization at annotation-load time (dedupe, quantity reconciliation, zero-price handling).

### Changed

- Rewrote heuristic recovery with a locale-aware token parser for receipt line-item layouts, total precedence (GRAND TOTAL > TOTAL > SUBTOTAL), and comma/dot separators.
- Line-item recovery is gated on receipt-likeness so forms no longer produce fabricated items.
- Recovered values carry calibrated confidences anchored to recovery strength.
- Multi-page invoice-number conflicts are detected from OCR tokens directly, independent of entity grouping.

### Fixed

- Category-aware span grouping keeps `Grand Total` / multi-word dates as single spans instead of fragmenting them.
- Structured receipt line items are no longer suppressed by the experiment refinement heuristic.
- `Total Item` / `Total Qty` count lines no longer mistaken for monetary totals.
- Processor is persisted with fine-tuned checkpoints so they load end to end.

### Benchmark

Field-level F1 on the cleaned 201-document ground truth improved from `0.5427` to `0.6168`; document exact match from 11/201 to 51/201; `line_items` conditional F1 from `0.5892` to `0.7370`.

## v1.0-benchmark-final

### Added

- Added a stronger NVIDIA-hosted LLM baseline configuration for benchmark comparisons.
- Added benchmark chart generation for system-level metrics and per-field F1 comparisons.
- Added a benchmark interpretation section that explains the current quality gap against the PRD target.

### Changed

- Hardened `rag_llm` field parsing with JSON repair and field-specific fallback extraction.
- Tightened RAG field prompts so per-field generations return only the value instead of explanations.
- Improved provider retry handling for rate-limited LLM calls during large benchmark runs.
- Updated reproducibility guidance to point to the benchmark lockfile, benchmark container image, and fixed seed handling.
- Documented the clean-container smoke-test caveat for benchmark verification versus full live provider-backed reruns.

### Fixed

- Fixed a `rag_llm` failure mode where repeated scalar outputs such as `null\nnull\n230000` collapsed to empty fields.
- Fixed fallback parsing for partial JSON snippets that previously raised `max() arg is an empty sequence`.
- Fixed benchmark resume handling so named config overrides can be used for additional baseline systems.
