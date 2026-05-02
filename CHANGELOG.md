# Changelog

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
