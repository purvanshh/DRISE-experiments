# DRISE — Document Retrieval & Intelligence with Structured Extraction

This is a comprehensive technical study guide for the DRISE document intelligence system. It explains **what** the system does, **how** it works from first principles, **why** specific architectural decisions were made, and **where** the design has strengths and weaknesses.

---

# 1. Project Overview

## 1.1 What the Project Does

DRISE is a **production-grade document extraction system** that transforms unstructured documents — invoices, receipts, scanned PDFs, images — into validated, structured JSON with per-field confidence scores and cross-field consistency checks.

It accepts documents via a REST API and returns fields like `invoice_number`, `date`, `vendor`, `total_amount`, and `line_items`, each with:
- A **value** (the extracted content)
- A **confidence score** (0.0–1.0, from the ML model)
- A **validity flag** (passed validation or not)
- Optional **constraint flags** (e.g., "line_items_sum_mismatch")

## 1.2 Core Problem It Solves

Document extraction is traditionally solved in one of three flawed ways:

| Approach | What Happens | Why It Fails |
|---|---|---|
| **OCR-only** | Extract raw text, apply regex rules | No spatial awareness — collapses on multi-column layouts, tables, non-linear reading orders |
| **LLM-based** | Prompt an LLM to extract fields | Non-deterministic outputs, hallucination risk, high per-document cost at scale |
| **Template-matching** | Match against known vendor layouts | Brittle — breaks on layout variation, requires per-vendor configuration |

DRISE combines a **layout-aware multimodal transformer** (LayoutLMv3) with a **deterministic post-processing pipeline**. The model understands spatial document structure. The post-processing layer guarantees identical output for identical input — **no variance between runs**.

## 1.3 Key Features and Capabilities

- **Layout-aware extraction**: LayoutLMv3 jointly encodes pixel content, text tokens, and bounding-box geometry, enabling the model to distinguish field labels from values across multi-column, tabular, and non-standard layouts.
- **Deterministic post-processing**: Three-stage pipeline — normalization → validation → constraint enforcement — guarantees reproducibility.
- **Defense-in-depth security**: File uploads validated at extension, MIME type, magic-byte, and size level.
- **Typed data contracts**: All pipeline stages communicate through explicit Pydantic interfaces.
- **Built-in ablation framework**: Controlled experiments with layout removal and constraint removal are implemented out of the box.
- **Multi-model support**: Swap between different LayoutLMv3 checkpoints — the pipeline adapts automatically.
- **Production API**: FastAPI service with structured error mapping, per-request tracing IDs, batch parsing, and health checks.
- **Experiment framework**: Three extraction pipelines (DRISE, LLM-only, RAG+LLM) for controlled benchmarking.

---

# 2. High-Level Architecture

## 2.1 Overall System Design

DRISE follows a **layered pipeline architecture** with clearly separated stages:

```
Upload → Validate → Rasterize → Preprocess → OCR → Model Inference → Post-process → Response
```

Each stage is:
- **Independent** — can be swapped, disabled, or benchmarked individually
- **Typed** — communicates through explicit data contracts (Pydantic models)
- **Instrumented** — logs timing and metadata at each stage

## 2.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (cURL / Frontend)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  /health  │  /v1/documents/parse  │  /v1/documents/batch  │ Errors  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER (Orchestration)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DocumentPipeline  →  DocumentParserService  →  ModelRuntime     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                   │                              │                      │
                   ▼                              ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING STAGES (per-request)                     │
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐   │
│  │ Ingestion │ → │   OCR        │ → │   Model     │ → │ Postprocess  │   │
│  │           │   │   (Paddle)   │   │   (Layout)  │   │ (3 stages)   │   │
│  └──────────┘    └──────────────┘    └─────────────┘    └──────────────┘   │
│       │               │                   │                   │              │
│       ▼               ▼                   ▼                   ▼              │
│  Validation      Token + BBox       Per-token BIO       Normalization     │
│  PDF/Image       extraction         classification      Validation         │
│  sanitization                                            Constraints      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.3 Major Components and How They Interact

| Component | Location | Responsibility | Dependencies |
|---|---|---|---|
| **API Layer** | `src/api/main.py`, `src/document_intelligence_engine/api/` | HTTP request handling, error mapping, validation | FastAPI, Pydantic, ingestion validators |
| **Service Layer** | `src/document_intelligence_engine/services/` | End-to-end pipeline orchestration | Ingestion, OCR, ModelRuntime, Postprocessing |
| **Ingestion** | `src/ingestion/` | File validation, PDF rasterization, page loading | Config, logging |
| **OCR** | `src/ocr/`, `src/document_intelligence_engine/ocr/` | Token extraction with bounding boxes | PaddleOCR or Tesseract |
| **Model Runtime** | `src/document_intelligence_engine/services/model_runtime.py` | Model loading, inference, fallback to heuristic | LayoutLMv3 or heuristic |
| **Multimodal** | `src/document_intelligence_engine/multimodal/` | LayoutLMv3 inference, training, fine-tuning | HuggingFace transformers, PyTorch |
| **Postprocessing** | `src/postprocessing/` | Normalization, validation, constraint enforcement | Config, field aliases |
| **LLM Client** | `src/document_intelligence_engine/llm/` | LLM API calls, caching, JSON parsing | OpenAI SDK or mock |
| **Retrieval** | `src/document_intelligence_engine/retrieval/` | Sentence-BERT embeddings for RAG pipeline | SentenceTransformers |
| **Evaluation** | `src/document_intelligence_engine/evaluation/` | Metrics, benchmarking, experiment runner | Seqeval, jsonschema |
| **Config** | `src/document_intelligence_engine/core/config.py` | YAML + environment variable configuration | Pydantic, PyYAML |

## 2.4 Data Flow Across the System

```
UploadFile
  → validate_upload()           # extension + MIME + magic bytes + size
  → load_pages()                # PDF rasterization or image open
  → ImagePreprocessor           # deterministic page preparation
  → OCRService.extract()        # tokens + bounding boxes + confidence scores
  → LayoutLMv3InferenceService  # per-token field classification (KEY/VALUE/O)
  → normalize_document()        # date / currency / OCR artifact correction
  → validate_document()         # regex + semantic field checks
  → apply_constraints()         # cross-field consistency enforcement
  → DocumentParseResponse       # typed, validated JSON output
```

---

# 3. Why This Architecture?

## 3.1 Why This Architecture Was Chosen

1. **Separation of concerns**: Each pipeline stage can be independently developed, tested, benchmarked, and swapped. This is essential for a research platform comparing three extraction paradigms (DRISE, LLM-only, RAG+LLM).

2. **Determinism as a first-class property**: Document extraction in production systems needs reproducibility. By layering a deterministic post-processing pipeline on top of a probabilistic ML model, DRISE achieves **semantic idempotence** — the same document always produces the same output JSON, even if the model's raw logits vary.

3. **Layout awareness without full end-to-end learning**: Using LayoutLMv3 (a pre-trained multimodal transformer) rather than training a custom model from scratch provides strong baseline performance on receipt/invoice extraction with minimal data.

4. **Type safety as documentation**: Pydantic models in `domain/contracts.py` and `domain/experiment_models.py` serve as both runtime validation and implicit API documentation.

5. **Experiment infrastructure built in**: The system was designed as a **research platform** from day one. The `ExperimentRunner`, three pipeline implementations, ablation framework, and metrics suite make controlled scientific comparison possible without glue code.

## 3.2 Trade-offs

| Dimension | Benefit | Cost |
|---|---|---|
| **Scalability** | Horizontal scaling via FastAPI workers | ML inference is compute-bound on CPU; GPU required for production throughput |
| **Complexity** | Modular stages are independently testable | More code paths, more configuration, more places for subtle bugs |
| **Cost** | LayoutLMv3 runs locally (no per-document API cost) | Heuristic fallback is crude; real model needs GPU |
| **Maintainability** | Typed contracts and explicit stage interfaces | Refactoring any stage requires updating contracts |
| **Reproducibility** | Deterministic post-processing eliminates run-to-run variance | Post-processing can mask model errors rather than exposing them |

## 3.3 When This Architecture Would Fail

1. **Severely degraded scans** (heavy noise, sub-100 DPI, mixed orientation): OCR ceiling — the model cannot recover from garbage input.

2. **Domain-specific documents** (legal contracts, medical records, multilingual): Performance degrades without targeted fine-tuning. The current model is fine-tuned on CORD (receipts) and FUNSD (forms).

3. **Multi-page documents with cross-page references**: Pages are processed independently. A total on page 2 referencing line items on page 1 is not currently resolved.

4. **Complex table structures**: Table cells are extracted, but row/column/span relationships are not reconstructed in the output schema.

5. **High-volume, low-latency requirements**: CPU-bound LayoutLMv3 inference is slow (~300ms per document). Would need GPU deployment and potentially batching.

---

# 4. Tech Stack Breakdown

## 4.1 Languages, Frameworks, Libraries

| Category | Technology | Why It Was Chosen |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem for ML (PyTorch, HuggingFace), OCR (PaddleOCR, Tesseract), and APIs (FastAPI) |
| **Web Framework** | FastAPI | Native async, OpenAPI generation, Pydantic integration, simple dependency injection |
| **ML Model** | LayoutLMv3 (microsoft/layoutlmv3-base) | State-of-the-art multimodal document understanding; jointly encodes text, layout, and image |
| **Fine-tuned Checkpoint** | jinhybr/OCR-LayoutLMv3-Invoice | Pre-trained on invoices; drop-in replacement for base model |
| **OCR Engine** | PaddleOCR (primary), Tesseract (fallback) | PaddleOCR provides better layout preservation and confidence scores |
| **LLM Integration** | OpenAI SDK + custom caching layer | Supports OpenAI, NVIDIA, and OpenAI-compatible endpoints; disk caching for reproducibility |
| **Retrieval** | Sentence-BERT (all-MiniLM-L6-v2) | Lightweight embeddings for RAG pipeline per-field context retrieval |
| **Configuration** | PyYAML + Pydantic | Type-safe config with environment variable overrides (`DIE_` prefix) |
| **Data Validation** | Pydantic | Runtime type checking and serialization for all inter-stage contracts |
| **Evaluation** | seqeval (token-level F1), jsonschema (schema validity), custom metrics | Token classification metrics + document-level exact match + hallucination detection |
| **Deployment** | Docker + Docker Compose | Reproducible environments, optional Redis for async processing |

## 4.2 Alternatives That Could Have Been Used

| Component | What Was Used | Alternative | Why Not Chosen |
|---|---|---|---|
| **ML Model** | LayoutLMv3 | Donut (BROS), FormLayoutLM, custom CNN+BiLSTM | LayoutLMv3 has the best pre-trained layout understanding; Donut requires full fine-tuning |
| **OCR** | PaddleOCR | EasyOCR, cloud APIs (Google Cloud Vision) | PaddleOCR is open-source, self-hosted, and provides per-token confidence |
| **API Framework** | FastAPI | Flask, Django, Starlette | FastAPI's Pydantic integration and async capabilities are a better fit for ML inference pipelines |
| **Configuration** | YAML + Pydantic | Hydra, dataclasses, environment variables only | Pydantic provides strong typing and validation; Hydra adds unnecessary complexity for this scale |
| **Vector Store** | In-memory (for RAG) | Chroma, Weaviate, Pinecone | Simple cosine-similarity retrieval doesn't require a full vector DB for 201-document benchmarks |

## 4.3 Key Dependencies (from requirements.txt)

```
fastapi, uvicorn          # API server
pydantic                  # Data validation and config
torch, transformers       # ML model inference
paddleocr                 # OCR engine
Pillow                    # Image processing
sentence-transformers     # RAG retrieval
openai                    # LLM client
pyyaml                    # Config loading
seqeval                   # Token-level F1 scoring
jsonschema                # Schema validation
python-dotenv             # Environment variables
```

---

# 5. Folder & Code Structure Deep Dive

## 5.1 Project Root Structure

```
.
├── configs/                      # YAML configuration files
│   ├── config.yaml               # Main application config
│   └── experiments.yaml          # Benchmark experiment definitions
├── data/                         # Data directories
│   ├── raw/                      # Source PDFs and images [gitignored]
│   ├── processed/                # Intermediate artifacts [gitignored]
│   └── annotations/              # Ground-truth labels (JSONL) [gitignored]
├── docker/                       # Docker deployment files
│   ├── Dockerfile                # Production container
│   └── docker-compose.yml        # Service orchestration
├── experiments/                  # Experiment outputs
│   ├── results/                  # Benchmark results, charts, reports
│   ├── cache/                    # LLM + retrieval caches [gitignored]
│   └── artifacts/                # Model checkpoints [gitignored]
├── src/
│   ├── document_intelligence_engine/   # Main package (newer, typed)
│   │   ├── api/                       # FastAPI app, routes, schemas
│   │   ├── core/                      # Config, logging, errors
│   │   ├── domain/                    # Typed Pydantic data contracts
│   │   ├── pipelines/                 # DRISE, LLM-only, RAG+LLM pipelines
│   │   ├── services/                  # End-to-end pipeline orchestration
│   │   ├── llm/                       # LLM client, prompts
│   │   ├── multimodal/                # LayoutLMv3 inference, training
│   │   ├── retrieval/                 # Sentence-BERT embedder + retriever
│   │   ├── evaluation/                # Metrics, evaluator, experiment runner
│   │   └── testing/                   # Test harness and fixtures
│   ├── api/                      # FastAPI app entrypoint (legacy location)
│   ├── ingestion/                # File validation, PDF loading
│   ├── ocr/                      # OCR engine integration
│   ├── preprocessing/            # Image normalization
│   └── postprocessing/            # Normalization, validation, constraints
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── load/                     # Load tests
│   ├── security/                 # Security tests
│   └── stress/                   # Stress tests
├── scripts/                      # CLI tools and utilities
├── run_experiments.py            # Experiment harness entry point
├── pyproject.toml                # Tooling configuration (ruff, black, pytest)
└── requirements.txt              # Dependencies
```

## 5.2 Key Source Modules

### `src/document_intelligence_engine/domain/`

**Responsibility**: Typed data contracts shared across all modules.

- `contracts.py` — Runtime contracts: `BoundingBox`, `OCRToken`, `OCRResult`, `ModelPrediction`, `ConstraintResult`, `ValidatedFile`, `DocumentProcessingResult`
- `experiment_models.py` — Experiment contracts: `ProcessedDocument`, `ExtractionOutput`, `PipelineBase` protocol

**Design pattern**: Pydantic models as both **data containers** and **runtime validators**. Every inter-stage communication goes through these models, catching type mismatches early.

### `src/document_intelligence_engine/pipelines/`

**Responsibility**: Three extraction implementations for controlled benchmarking.

| Pipeline | File | Approach |
|---|---|---|
| **DRISE** | `drise.py` | Full pipeline: OCR → LayoutLMv3 → post-processing → constraints |
| **LLM-only** | `llm_only.py` | LLM with schema-constrained prompting on full OCR text |
| **RAG+LLM** | `rag_llm.py` | Per-field retrieval using Sentence-BERT, then per-field LLM extraction |
| **Base** | `base.py` | Abstract `BasePipeline` class defining the `run(document) -> ExtractionOutput` interface |

**Design pattern**: **Strategy pattern** — all three pipelines implement the same interface (`BasePipeline`), making them interchangeable in the `ExperimentRunner`.

### `src/document_intelligence_engine/services/`

**Responsibility**: Production service orchestration.

- `pipeline.py` — `DocumentPipeline`: orchestrates the full end-to-end flow for the API
- `document_parser.py` — `DocumentParserService`: shared by both API and experiments; calls ingestion → OCR → model → postprocessing
- `model_runtime.py` — `LayoutAwareModelService`: loads LayoutLMv3 at startup, manages heuristic fallback, exposes `predict()` method

### `src/document_intelligence_engine/multimodal/`

**Responsibility**: LayoutLMv3 inference and fine-tuning.

- `layoutlmv3.py` — `LayoutLMv3InferenceService`: loads a checkpoint, runs token classification with BIO labels, returns per-token predictions with softmax confidences
- `cord_dataset.py` — `CORDDataset`: PyTorch dataset wrapping the CORD receipt dataset for fine-tuning; maps CORD labels to the project's 5-class BIO schema
- `training.py` — Fine-tuning entry point

### `src/document_intelligence_engine/evaluation/`

**Responsibility**: Benchmarking and metrics.

- `runner.py` — `ExperimentRunner`: runs multiple pipelines across a dataset, handles resume/caching, enforces cost caps
- `evaluator.py` — `Evaluator`: computes per-document metrics
- `metrics.py` — Field-level F1, exact match, schema validity, hallucination rate computation
- `report.py` — Generates experiment summary reports (CSV, JSON, visualizations)
- `stats.py` — Statistical tests (McNemar's exact test for pairwise comparison)

### `src/postprocessing/` (legacy, still actively used)

**Responsibility**: Three-stage deterministic post-processing.

- `pipeline.py` — `postprocess_predictions()`: runs all three stages in sequence
- `entity_grouping.py` — Groups sequential KEY/VALUE tokens into field spans
- `normalization.py` — Converts dates to ISO 8601, currencies to floats, corrects OCR artifacts (`O → 0`, `l → 1`, `S → 5`)
- `validation.py` — Regex pattern matching on fields, checks for required fields
- `constraints.py` — Cross-field consistency (e.g., `Σ(line_items) ≈ total_amount` within tolerance)
- `confidence.py` — Confidence thresholding, low-confidence field dropping

### `src/ingestion/`

**Responsibility**: File validation and PDF rasterization.

- `pipeline.py` — `process_document_with_metadata()`: validates → loads → preprocesses → OCRs → aligns tokens
- `file_validator.py` — Extension, MIME type, magic byte, size validation
- `pdf_loader.py` — PDF → page images via `pdf2image` or `PyMuPDF`

### `src/ocr/`

**Responsibility**: OCR engine abstraction.

- `service.py` — `OCRService`: dispatches to configured backend
- `ocr_engine.py` — PaddleOCR integration with batch token extraction
- `bbox_alignment.py` — Aligns OCR tokens with preprocessed image coordinates

### `src/document_intelligence_engine/llm/`

**Responsibility**: LLM client with caching and JSON parsing.

- `client.py` — `LLMClient`: OpenAI/NVIDIA API client with on-disk caching (`experiments/cache/llm/`), retry logic, cost tracking, mock backend for testing
- `prompts.py` — Prompt templates for extraction and per-field retrieval

### `src/document_intelligence_engine/retrieval/`

**Responsibility**: Semantic retrieval for the RAG pipeline.

- `embedder.py` — `Embedder`: Sentence-BERT wrapper
- `retriever.py` — `DocumentRetriever`: cosine-similarity retrieval of top-k chunks per query

### `src/document_intelligence_engine/core/`

**Responsibility**: Cross-cutting concerns.

- `config.py` — `get_settings()`: loads YAML config, applies `DIE_` env var overrides, resolves paths, caches via `@lru_cache`
- `logging.py` — Structured JSON logging via `structlog`
- `errors.py` — Custom exception hierarchy (`ConfigurationError`, `OCRProcessingError`, `ModelInferenceError`, etc.)

### `src/api/`

**Responsibility**: FastAPI application entry point.

- `main.py` — `create_app()`: FastAPI factory with middleware, routes, exception handlers, lifespan management
- `routes/` — `/health`, `/v1/documents/parse`, `/v1/documents/batch`
- `schemas/` — Pydantic request/response models
- `middleware.py` — CORS, request ID injection, rate limiting

## 5.3 How Modules Are Connected

```
API (FastAPI)
    ↓
DocumentPipeline (service)
    ↓
DocumentParserService (service)
    ├── ingestion/pipeline.py (validate → load → preprocess → OCR → align)
    ├── model_runtime.py (model.predict())
    │   └── multimodal/layoutlmv3.py (LayoutLMv3InferenceService)
    └── postprocessing/pipeline.py (group → normalize → validate → constrain → confidence)
        ├── postprocessing/entity_grouping.py
        ├── postprocessing/normalization.py
        ├── postprocessing/validation.py
        ├── postprocessing/constraints.py
        └── postprocessing/confidence.py
```

For **experiments** (not API):
```
run_experiments.py
    ↓
ExperimentRunner
    ↓
Pipelines (DRISEPipeline | LLMOnlyPipeline | RAGLLMPipeline)
    ├── DRISEPipeline → model_runtime → postprocessing
    ├── LLMOnlyPipeline → llm/client (full-text prompt)
    └── RAGLLMPipeline → retrieval (embedder + retriever) → llm/client (per-field prompts)
    ↓
Evaluator → metrics.py
    ↓
generate_experiment_report → experiments/results/
```

---

# 6. Core Workflows

## 6.1 Document Parsing (API Path)

A client uploads a PDF. The following steps execute:

1. **Validation** (`src/ingestion/file_validator.py`):
   - Extension check: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
   - MIME type check: `application/pdf`, `image/png`, etc.
   - Magic byte verification
   - Size check: ≤25MB
   - SHA256 computation for `document_id`

2. **Rasterization** (`src/ingestion/pdf_loader.py`):
   - If PDF: convert each page to a PIL Image at configured DPI (default: 200)
   - If image: open directly with PIL

3. **Preprocessing** (`src/preprocessing/image_preprocessing.py`):
   - Resize to max 1600×1600 while preserving aspect ratio
   - Normalize pixel values to [0, 1]
   - Optional noise reduction (median blur)
   - Optional deskewing

4. **OCR** (`src/ocr/ocr_engine.py`, PaddleOCR):
   - Extract text tokens with bounding boxes
   - Each token: `{text: str, bbox: [x0, y0, x1, y1], confidence: float, page_number: int}`
   - PaddleOCR returns ~50-500 tokens per document depending on complexity

5. **Model Inference** (`src/document_intelligence_engine/multimodal/layoutlmv3.py`):
   - LayoutLMv3Processor tokenizes OCR words, encodes bounding boxes normalized to 0-1000
   - Model runs token classification with 5 labels: `O`, `B-KEY`, `I-KEY`, `B-VALUE`, `I-VALUE`
   - Softmax on logits → confidence scores
   - Subword-level predictions aggregated back to word level via `word_ids`

6. **Post-processing** (`src/postprocessing/pipeline.py`, three stages):

   **Stage 1 — Entity Grouping**: Convert BIO sequences into grouped entities:
   ```
   Input: [B-KEY, I-KEY, B-VALUE, I-VALUE, O, O, B-VALUE, I-VALUE]
   Output: {"invoice_number": ["INV-1023"], "vendor": ["ABC Corp"], "total_amount": ["1200.50"]}
   ```

   **Stage 2 — Normalization**:
   - Dates → ISO 8601 (`"Jan 12, 2025"` → `"2025-01-12"`)
   - Currency strings → floats (`"$1,200.50"` → `1200.50`)
   - OCR artifact correction: `O → 0`, `l → 1`, `S → 5` (applied contextually based on numeric proximity)

   **Stage 3 — Validation**:
   - Regex pattern matching (e.g., `invoice_number` must match `^[A-Za-z0-9][A-Za-z0-9_/-]{1,63}$`)
   - Required fields check
   - Type checking (date must be ISO format, amount must be numeric)

   **Stage 4 — Constraints**:
   - Cross-field consistency: compute `Σ(line_items)` and compare to `total_amount` within tolerance (default: 1%)
   - If mismatch: append `line_items_sum_mismatch` flag to `_constraint_flags`
   - Constraints are **diagnostic**, not corrective — downstream consumers decide how to handle discrepancies

   **Stage 5 — Confidence Policy**:
   - If field confidence < `min_field_confidence` (default: 0.60) → set `valid: false`
   - Optionally drop fields below threshold

7. **Response**:
   ```json
   {
     "document": {
       "invoice_number": {"value": "INV-1023", "confidence": 0.924, "valid": true},
       "date": {"value": "2025-01-12", "confidence": 0.911, "valid": true},
       "vendor": {"value": "ABC Pvt Ltd", "confidence": 0.887, "valid": true},
       "total_amount": {"value": 1200.50, "confidence": 0.883, "valid": true},
       "line_items": {"value": [{"description": "Product A", "quantity": 2, "unit_price": 400.00}], "valid": true},
       "_constraint_flags": [],
       "_errors": []
     },
     "metadata": {"filename": "invoice.pdf", "page_count": 1, "request_id": "req_..."}
   }
   ```

## 6.2 Experiment Execution (Benchmarking Path)

1. **Configuration** (`run_experiments.py`, `configs/experiments.yaml`):
   - Load experiment config: dataset path, systems to run, output directory, cost cap
   - Set random seeds (42) for reproducibility across Python, NumPy, PyTorch

2. **Dataset Loading** (`src/document_intelligence_engine/data/annotation_loader.py`):
   - Load `data/annotations/test.jsonl` — each line is a JSON object with:
     - `doc_id`: unique document identifier
     - `image_path`: path to source PDF/image
     - `ground_truth`: the expected extraction output
     - `ocr_text`: pre-computed OCR text (optional; if absent, OCR is re-run)

3. **System Instantiation**:
   - `DRISEPipeline(config)`: builds `DocumentParserService` + `LayoutAwareModelService`
   - `LLMOnlyPipeline(config)`: builds `LLMClient` with configured backend/model
   - `RAGLLMPipeline(config)`: builds `LLMClient` + `DocumentRetriever` (Embedder + cosine similarity)

4. **Execution Loop** (`ExperimentRunner.run()`):
   - For each document in dataset:
     - Call `pipeline.run(document)` → `ExtractionOutput`
     - Run `evaluator.evaluate(output, ground_truth, ocr_text)` → metrics dict
     - Write incremental results to `experiments/results/{pipeline_name}.json`
     - Enforce cost cap (`max_total_cost_usd`) — abort if exceeded

5. **Evaluation Metrics** (`src/document_intelligence_engine/evaluation/metrics.py`):
   - `compute_field_level_f1()`: micro-averaged F1 across all fields
   - `compute_exact_match()`: document-level exact match (all fields must match exactly after normalization)
   - `compute_schema_validity()`: JSON Schema validation pass/fail
   - `compute_hallucination_rate()`: fraction of extracted values not grounded in source text
   - Per-field F1: `compute_field_f1_scores()` for each field individually

6. **Report Generation** (`src/document_intelligence_engine/evaluation/report.py`):
   - Aggregate metrics across all documents per system
   - Compute statistical significance via McNemar's exact test on exact-match binary outcomes
   - Generate CSV, JSON, and visualization outputs

## 6.3 Real-World Example

**Input**: A scanned invoice PDF with multi-column layout, "TOTAL AMOUNT: $1,2OO.5O" (OCR misread zeros as letter O).

**Processing steps**:
1. **Validation**: PDF passes magic byte check, size 2.3MB
2. **Rasterization**: 1 page → 1 PIL Image (200 DPI)
3. **Preprocessing**: Resized to 1600×2200, normalized pixels
4. **OCR**: PaddleOCR extracts 127 tokens with bounding boxes
5. **Model**: LayoutLMv3 classifies each token as KEY/VALUE/O with confidences (e.g., "TOTAL AMOUNT" → B-KEY at 0.94, "$1,2OO.5O" → B-VALUE at 0.88)
6. **Post-processing**:
   - **Entity grouping**: `{"total_amount": ["$1,2OO.5O"]}`
   - **Normalization**: currency parser sees "1,2OO.5O" → applies OCR artifact map → converts to `1200.50` (float)
   - **Validation**: regex `^-?\d+(\.\d+)?$` passes
   - **Constraints**: computes `Σ(line_items)` = 1200.50, compares to `total_amount` = 1200.50 → within 1% tolerance → no flag
   - **Confidence**: 0.88 ≥ 0.60 threshold → `valid: true`

**Output**:
```json
{"total_amount": {"value": 1200.50, "confidence": 0.88, "valid": true}}
```

**Key insight**: Without the deterministic post-processing, the raw model output "$1,2OO.5O" would pass through to the client, causing downstream parsing failures. The normalization layer catches and corrects this.

---

# 7. Data Layer & State Management

## 7.1 Configuration Data

Configuration is loaded from `configs/config.yaml` with environment variable overrides:

- YAML file defines all system settings (API, OCR, model, post-processing rules, etc.)
- `DIE_` prefix + double underscore for nesting: `DIE_OCR__MIN_CONFIDENCE=0.6`
- Settings are cached via `@lru_cache` to avoid repeated YAML parsing
- Path resolution: relative paths are resolved against the project root

**Key config sections**:
- `paths`: data directories, experiment directories
- `api`: host, port, CORS, rate limits, batch settings
- `ingestion`: max file size, DPI, supported extensions
- `ocr`: backend (paddleocr/tesseract), confidence threshold, GPU settings
- `model`: LayoutLMv3 model name, device (cpu/cuda), checkpoint path, heuristic fallback
- `postprocessing.field_aliases`: mapping from natural language field names to canonical names
- `postprocessing.normalization`: date fields, currency fields, OCR artifact maps
- `postprocessing.validation`: regex patterns for each field, required fields
- `postprocessing.constraints`: amount tolerance (0.01 = 1%), date handling

## 7.2 Runtime Data

### In-Memory State
- **Model loaded once** at startup in `LayoutAwareModelService` — not reloaded per request
- **DocumentPipeline** instantiated per request (or pooled) — holds references to model service
- **LLMClient** maintains cumulative cost tracking and on-disk cache

### Persistent Data
- **File uploads**: Written to `data/processed/uploads/` with sanitized filenames (SHA256-based)
- **Experiment results**: Written to `experiments/results/{system}.json` — JSONL format with one record per document
- **LLM cache**: `experiments/cache/llm/{sha256}.json` — cached API responses for reproducibility
- **Retrieval cache**: `experiments/cache/retrieval/` — cached embeddings for RAG pipeline
- **Model checkpoints**: `experiments/artifacts/cord_finetuned/` — fine-tuned model weights

## 7.3 Data Flow in the Post-Processing Pipeline

```
Raw model predictions (list[dict])
    │
    ▼
entity_grouping.py ──────────────► Group sequential KEY/VALUE tokens into field spans
    │  Input: [{"text": "Invoice", "label": "B-KEY"}, {"text": "#", "label": "I-KEY"}, {"text": "1023", "label": "B-VALUE"}]
    │  Output: {"invoice_number": [{"text": "1023", "label": "B-VALUE", "confidence": 0.91}]}
    ▼
normalization.py ───────────────► Normalize values (dates → ISO, currencies → float, OCR artifacts corrected)
    │  Input: {"date": [{"text": "Jan 12, 2025"}], "total_amount": [{"text": "$1,2OO.5O"}]}
    │  Output: {"date": "2025-01-12", "total_amount": 1200.50}
    ▼
validation.py ───────────────────► Regex validation, required field check, type checking
    │  Input: {"invoice_number": "1023", "total_amount": 1200.50}
    │  Output: {"invoice_number": {"value": "1023", "valid": true}, "total_amount": {"value": 1200.50, "valid": true}}
    ▼
constraints.py ──────────────────► Cross-field consistency checks (line items sum vs. total)
    │  Input: {"line_items": [...], "total_amount": 1200.50}
    │  Output: document + _constraint_flags: [] (or ["line_items_sum_mismatch"])
    ▼
confidence.py ──────────────────► Drop or flag low-confidence fields
    │  Input: {"invoice_number": {"value": "1023", "confidence": 0.55}}
    │  Output: {"invoice_number": {"value": "1023", "confidence": 0.55, "valid": false}}  // below 0.60 threshold
    ▼
Final structured document dict
```

---

# 8. Key Design Patterns Used

## 8.1 Pipeline Pattern

**Location**: `src/ingestion/pipeline.py`, `src/postprocessing/pipeline.py`, `src/document_intelligence_engine/services/document_parser.py`

Each stage transforms input to output, passing the result to the next stage. This enables:
- Easy insertion of new stages
- Easy disabling or swapping of stages (e.g., `apply_constraints=False` for ablation)
- Independent testing of each stage

## 8.2 Strategy Pattern

**Location**: `src/document_intelligence_engine/pipelines/` — `DRISEPipeline`, `LLMOnlyPipeline`, `RAGLLMPipeline`

All implement `BasePipeline` protocol with `run(document) -> ExtractionOutput`. The `ExperimentRunner` iterates over a list of systems without needing to know which implementation is being used. This makes adding new extraction methods trivial.

## 8.3 Factory Pattern

**Location**: `src/document_intelligence_engine/core/config.py` — `get_settings()`

Configuration is constructed from YAML + environment variables, with sensible defaults and type coercion. The single `get_settings()` function is the factory for the entire application's configuration.

Also used in:
- `LayoutAwareModelService.load()` — resolves checkpoint path, falls back to published model or heuristic
- `DocumentParserService` — builds ingestion → model → postprocessing chain
- `OCRService` — instantiates the configured OCR backend

## 8.4 Template Method Pattern

**Location**: `src/document_intelligence_engine/multimodal/layoutlmv3.py` — `LayoutLMv3InferenceService.predict()`

The inference workflow is fixed: extract words/boxes → normalize boxes → encode → forward → softmax → aggregate. Subclass overrides or configuration flags modify specific steps (e.g., `use_layout=False` for ablation disables bounding box encoding).

## 8.5 Dependency Injection

**Location**: `src/document_intelligence_engine/services/` — services receive dependencies in constructor

- `DocumentParserService(settings, model_service)` — dependencies injected, not created internally
- Allows testing with mock model services
- Allows swapping implementations without changing calling code

## 8.6 Data Contracts (Pydantic)

**Location**: `src/document_intelligence_engine/domain/contracts.py`

Every inter-module communication uses Pydantic models. This is not a formal pattern but serves as **implicit interface enforcement** — the type system catches contract violations at runtime.

## 8.7 Caching with Disk Persistence

**Location**: `src/document_intelligence_engine/llm/client.py` — LLM responses cached to `experiments/cache/llm/`

- SHA256 of prompt + config → JSON file
- Enables resume of interrupted experiments
- Enables exact reproduction of results without API calls
- Cost tracking persists across runs

---

# 9. Performance & Scalability Considerations

## 9.1 Bottlenecks in the Current System

| Bottleneck | Location | Severity | Impact |
|---|---|---|---|
| **LayoutLMv3 inference on CPU** | `src/document_intelligence_engine/multimodal/layoutlmv3.py` | High | ~300ms per document on CPU; GPU inference would reduce to ~50ms |
| **PaddleOCR** | `src/ocr/ocr_engine.py` | Medium | ~200-400ms per document; dominated by model inference |
| **Sequential page processing** | `src/ingestion/pipeline.py` | Medium | Pages processed one at a time; batch processing is enabled but not parallelized |
| **LLM API calls (RAG pipeline)** | `src/document_intelligence_engine/pipelines/rag_llm.py` | High | Per-field LLM call → 5 calls per document → ~2 seconds per document + API latency + cost |
| **Sentence-BERT embeddings** | `src/document_intelligence_engine/retrieval/embedder.py` | Low-Medium | ~50ms per document for embedding; cached after first run |

## 9.2 How It Scales (or Doesn't)

- **Horizontally (API)**: FastAPI workers can be increased (`workers: 4` in config), but each worker loads its own model → memory scaling issue
- **Vertically (GPU)**: Switching `device: cuda` in config would dramatically improve inference throughput, but requires CUDA-enabled environment
- **Batch processing**: API supports `/parse-batch` with `max_batch_files: 10`, but internally processes sequentially
- **LLM pipelines**: Do not scale — each document triggers API calls with per-field cost; cache helps but doesn't reduce latency

## 9.3 Suggestions for Improvement

1. **GPU deployment**: Add `device: cuda` in production config; use ONNX or TensorRT for 2-3× inference speedup
2. **Async batch processing**: Use `asyncio` for concurrent document processing within a batch
3. **Model distillation**: Distill LayoutLMv3 to a smaller model (e.g., MobileBERT) for CPU deployment
4. **OCR caching**: Cache OCR results by file SHA256 — same document reprocessed in <10ms
5. **LLM batching**: For RAG pipeline, batch per-field prompts into a single LLM call if the model supports it
6. **Streaming response**: Use FastAPI's `StreamingResponse` to return results as they're computed, not after the full pipeline

---

# 10. Weaknesses & Limitations

## 10.1 Design Flaws or Risks

1. **OCR ceiling is absolute**: If OCR produces garbage (severely degraded scans, <100 DPI, extreme noise), no amount of model sophistication can recover. The system has no denoising or image enhancement pre-processing.

2. **Single-page assumption**: Multi-page documents are processed page-by-page, but cross-page field references (e.g., "total on page 2 references line items on page 1") are not resolved. This is a known gap in the roadmap.

3. **Table structure not reconstructed**: Table cells are extracted individually, but row/column/spans are lost in the output. A "Purchased Items" table becomes a flat list.

4. **Label schema mismatch**: The model is trained on CORD (receipts) which annotates field *values* not field *names*. The project maps everything to KEY/VALUE, but there's no actual KEY detection in the model — it's inferred from token position (text before value = KEY). This is brittle for non-receipt documents.

5. **Hallucination detection is heuristic**: The hallucination metric (`compute_hallucination_rate()`) uses fuzzy string matching against the OCR text. It flags formatting differences (e.g., `"1200.50"` vs `"1,200.50"`) as hallucination, which inflates the rate. Manual spot-checks suggest true fabrication rate is <2%, but the metric reports ~6.8%.

6. **Configuration drift**: The system loads config at startup and caches it. Hot-reloading config changes requires service restart. This is problematic in containerized deployments with configmap updates.

7. **Security: path traversal risk**: File validation checks extension/MIME but relies on `secure_filename()` for path sanitization. The pattern `[^A-Za-z0-9._-]` is applied, but the implementation path should be audited for edge cases.

## 10.2 Technical Debt Areas

1. **Dual module structure**: The codebase has both `src/document_intelligence_engine/` (newer, typed) and `src/ingestion/`, `src/ocr/`, `src/preprocessing/`, `src/postprocessing/` (legacy but actively used). The migration is incomplete — some imports reference legacy modules directly in services.

2. **Inconsistent error handling**: Some stages raise custom exceptions (`InvalidFileError`, `OCRProcessingError`), others raise generic `Exception`. The error hierarchy in `core/errors.py` exists but isn't uniformly applied.

3. **Test coverage**: Unit tests exist in `tests/unit/` but integration and load tests are minimal. The smoke test (`smoke_test.py`) exercises the end-to-end flow with a mock LLM but doesn't validate all pipeline stages.

4. **Logging inconsistency**: Some modules use `get_logger()` (structured logging), others use `print()` or no logging. A unified logging strategy isn't enforced.

5. **Hardcoded thresholds**: Many magic numbers exist without config exposure:
   - `min_field_confidence: 0.60` (in code, not config)
   - `amount_tolerance: 0.01` (in config, but others are hardcoded)
   - Ablation flags (`use_layout=True`, `apply_constraints=True`) passed as method args

## 10.3 What Would Break Under Scale or Edge Cases

| Scenario | What Breaks | Why |
|---|---|---|
| **1000-page PDF** | Memory, page processing | No streaming; entire PDF rasterized into memory |
| **Non-English documents** | Extraction quality | Model trained on English receipts; OCR language set to "en" in config |
| ** Extremely noisy image** | OCR + model output | OCR confidence drops below threshold → tokens filtered → model gets empty input |
| **Concurrent API requests (50+)** | Latency, memory | CPU-bound model inference serialized; each request blocks the model |
| **LLM API rate limiting** | RAG pipeline | No backoff beyond retry; could be rate-limited by NVIDIA/OpenAI |
| **Corrupted PDF** | Ingestion pipeline | `pdf2image` or `PyMuPDF` raises unhandled exception → 500 error |
| **Disk full** | Caching, uploads | No graceful handling; would fail silently or crash |

---

# 11. How to Improve This System

## 11.1 Concrete, Actionable Improvements

1. **Add image enhancement preprocessing**:
   - Insert a denoising stage (bilateral filter or DnCNN) before OCR
   - This addresses the OCR ceiling for degraded scans

2. **Implement cross-page field resolution**:
   - Add a "page joining" stage that tracks field references across pages
   - Simple heuristic: collect all `total_amount` mentions, use the last one; collect all `line_items`, concatenate across pages

3. **Extract and preserve table structure**:
   - Use the bounding box coordinates to infer row/column relationships
   - Implement table detection (detect grids of aligned boxes) and output nested structures

4. **Calibrate hallucination metric**:
   - Use the current metric as a "flagging" mechanism but not a direct score
   - Apply secondary filtering: exclude values that differ only in formatting (commas, currency symbols)

5. **Add health endpoint with model readiness**:
   - The `/health` endpoint currently checks service liveness but not model loaded state
   - Add a `/health/ready` endpoint that verifies `LayoutAwareModelService.loaded == True`

6. **Expose more thresholds to config**:
   - Move `min_field_confidence` from code to `config.yaml`
   - Add environment variable override support: `DIE_POSTPROCESSING__CONFIDENCE__MIN_FIELD_CONFIDENCE=0.7`

7. **Add request ID propagation**:
   - Currently, request IDs are generated but not propagated through all stages
   - Add a `request_id` context variable (using `contextvars`) that's logged at every stage for end-to-end tracing

8. **Implement circuit breaker for LLM calls**:
   - For RAG pipeline: if LLM is slow/unavailable, fall back to heuristic extraction rather than failing the entire request

## 11.2 Better Architectural Alternatives

1. **Fine-tuned domain-specific model**: Instead of using `jinhybr/OCR-LayoutLMv3-Invoice`, fine-tune on the specific document distribution (invoices from the target domain). This would likely improve F1 from 0.58 to the aspirational 0.82 target.

2. **Hybrid retrieval + extraction**: Combine RAG-style retrieval with extraction — retrieve similar documents from a labeled corpus, use their annotations as soft labels for the current document. This could improve generalization on unseen layouts.

3. **Active learning loop**: Route low-confidence extractions to human review, collect corrections, and use as additional training data. This is in the roadmap but not implemented.

4. **Modular model serving**: Instead of loading the model in each worker process, use a separate model server (TorchServe, Triton Inference Server) that handles batching and GPU sharing. This would dramatically improve throughput.

## 11.3 Refactoring Suggestions

1. **Complete the module migration**: Move all code from `src/ingestion/`, `src/ocr/`, `src/preprocessing/`, `src/postprocessing/` into `src/document_intelligence_engine/` with proper subdirectories. Delete legacy modules.

2. **Consolidate error handling**: Define a strict exception hierarchy in `core/errors.py` and use it consistently across all stages. Map each exception type to a single HTTP status code in the API.

3. **Add structured logging everywhere**: Replace `print()` statements and ad-hoc logging with the `get_logger()` pattern everywhere. Ensure every log entry includes `request_id`, `stage`, and `duration_ms`.

4. **Separate experiment code from production code**: Currently, `pipelines/drise.py` is used by both the API (`DocumentParserService`) and experiments (`ExperimentRunner`). Extract the production path into a dedicated service class and keep experiment-specific logic isolated.

---

# 12. Learning Notes (For a Developer)

## 12.1 Key Concepts to Study from This Project

### Document Intelligence Fundamentals
- **Token classification for document extraction**: Understanding BIO tagging schemes, sequence labeling with seqeval, and per-token vs. per-entity metrics
- **Layout-aware multimodal models**: How LayoutLMv3 jointly encodes text, bounding boxes, and image patches; why this matters for multi-column and table layouts
- **OCR fundamentals**: The difference between text extraction (what the computer "sees") and semantic understanding (what the document "means")

### System Design Patterns
- **Pipeline architecture**: Decomposing a complex task into independent, testable stages
- **Deterministic post-processing on top of probabilistic ML**: How to guarantee reproducibility from a non-deterministic model
- **Experiment framework design**: How to build a rigorous, reproducible benchmarking system with multiple baselines, ablation studies, and statistical testing

### Production Engineering
- **Configuration management**: YAML + environment variables + Pydantic for type-safe, overridable configuration
- **API design**: FastAPI with typed request/response models, structured error handling, and async capabilities
- **Caching strategies**: Disk-based caching for LLM responses, embedding caching, incremental result writing for long-running experiments

## 12.2 What Skills This Project Demonstrates

| Skill | Where It's Demonstrated |
|---|---|
| **End-to-end ML system design** | From PDF upload to structured JSON output — full pipeline |
| **Multimodal deep learning** | LayoutLMv3 inference, fine-tuning on CORD |
| **API development** | FastAPI with proper error handling, validation, middleware |
| **Experiment design and execution** | Three baseline comparison, ablation studies, McNemar statistical tests |
| **Configuration management** | YAML + env vars + Pydantic typing |
| **Performance optimization** | Caching, batch processing, memory management |
| **Code organization** | Modular package structure, clear separation of concerns |

## 12.3 How to Replicate or Build Something Similar

**For a learner building their first document extraction system:**

1. **Start with the pipeline concept**: Don't try to build everything at once. Start with one stage (e.g., OCR only), verify it works, then add the next stage.

2. **Use pre-trained models first**: Don't fine-tune from scratch. Start with `microsoft/layoutlmv3-base` or `jinhybr/OCR-LayoutLMv3-Invoice`. Fine-tuning comes later after you've validated the pipeline.

3. **Make post-processing explicit**: Every decision about how to interpret model output should be a configurable rule, not hidden in the model. This is what makes DRISE "deterministic."

4. **Measure everything**: Add timing logging at every stage. You can't optimize what you can't measure.

5. **Build the experiment framework early**: Before you optimize, establish a baseline. DRISE's greatest strength is that it can rigorously compare three extraction approaches on the same dataset.

6. **Use typed contracts everywhere**: Pydantic models aren't just for the API — they're the connective tissue between every stage. Invest in defining them well upfront.

**For a senior engineer building a production version:**

1. **Add GPU support first**: The CPU baseline is useful for development, but production requires GPU for acceptable latency.

2. **Separate model serving**: Don't embed the model in the API worker. Use a dedicated inference service (TorchServe, Triton) with batching and GPU sharing.

3. **Add observability**: Distributed tracing (OpenTelemetry), metrics (Prometheus), structured logging with correlation IDs.

4. **Implement idempotent processing**: Use content hashing (SHA256) to skip reprocessing of identical documents.

5. **Plan for multi-tenancy**: The current system is single-tenant. Add tenant isolation, per-tenant model versioning, and quota management.

---

# Appendix: Quick Reference

## Key Files

| Purpose | Path |
|---|---|
| Main config | `configs/config.yaml` |
| Experiment config | `configs/experiments.yaml` |
| API entrypoint | `src/api/main.py` |
| Experiment runner | `run_experiments.py` |
| Data contracts | `src/document_intelligence_engine/domain/contracts.py` |
| LayoutLMv3 inference | `src/document_intelligence_engine/multimodal/layoutlmv3.py` |
| Post-processing pipeline | `src/postprocessing/pipeline.py` |
| LLM client | `src/document_intelligence_engine/llm/client.py` |
| Evaluation metrics | `src/document_intelligence_engine/evaluation/metrics.py` |

## Key Config Values

| Parameter | Default | Meaning |
|---|---|---|
| `model.device` | `cpu` | Inference device |
| `ocr.backend` | `paddleocr` | OCR engine |
| `postprocessing.confidence.min_field_confidence` | `0.60` | Minimum confidence for `valid: true` |
| `postprocessing.constraints.amount_tolerance` | `0.01` | 1% tolerance for line item sum vs. total |
| `performance.page_batch_size` | `4` | Pages processed per batch |

## Key Commands

```bash
# Run the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run experiments
python run_experiments.py --config configs/experiments.yaml

# Parse a single document via CLI
python -m document_intelligence_engine entrypoint path/to/document.pdf

# Fine-tune on CORD
python -m document_intelligence_engine.multimodal.training

# Run tests
pytest tests/
```

---

*Generated for deep architectural study of the DRISE document intelligence system.*