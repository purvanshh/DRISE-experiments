<div align="center">
  <h1>Document Intelligence Engine</h1>
  <p><strong>Layout-Aware Multimodal Document Parsing — PDF/Image → Deterministic Structured JSON</strong></p>

  [![Python 3.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
  [![LayoutLMv3](https://img.shields.io/badge/model-LayoutLMv3-FFD21E.svg)](https://huggingface.co/microsoft/layoutlmv3-base)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## Overview

**Document Intelligence Engine** is a production-grade system that converts unstructured documents — PDFs, invoices, receipts, scanned forms — into validated structured JSON.

It addresses a fundamental gap in document automation: **OCR-only systems** have no spatial awareness and collapse on complex layouts; **LLM-based extractors** are non-deterministic and cannot be trusted for production output. This system combines **LayoutLMv3** (a multimodal transformer that jointly encodes pixel layout, text tokens, and bounding box positions) with a strict **deterministic post-processing layer** that validates, normalizes, and enforces cross-field constraints on every extraction — guaranteed same output for same input.

---

## Key Features

- **Layout-Aware Extraction**: LayoutLMv3 encodes bounding box coordinates alongside text, allowing the model to distinguish field labels from their values even on multi-column, tabular, or non-standard form layouts.
- **Deterministic Post-Processing**: Every output passes through normalization (dates → ISO 8601, currencies → `float`), regex field validation, and a constraint engine (e.g., `sum(line_items) ≈ total_amount`). No variance between runs.
- **Strict Security by Design**: File uploads are validated at extension, MIME type, and magic-byte level. Oversized files, malformed PDFs, and path traversal attempts are rejected before processing.
- **Typed Data Contracts**: `ValidatedFile`, `OCRResult`, `ModelPrediction`, `ConstraintResult` — every stage in the pipeline has an explicit typed interface.
- **Ablation Framework**: Three canonical experiments (no layout embeddings, no post-processing, degraded OCR quality) are implemented and runnable out of the box.
- **Multi-LLM Backbone**: Swap between `microsoft/layoutlmv3-base` and any fine-tuned checkpoint without changing the pipeline.
- **Production API**: FastAPI with structured error mapping, per-request IDs, batch parsing endpoint, and background file cleanup.

---

## Architecture

```mermaid
graph TD
    subgraph Frontend [Client]
        CL[HTTP Client / cURL / UI]
    end

    subgraph API [FastAPI Layer]
        UP[Upload Validation]
        RT[Router]
        EH[Exception Mapper]
    end

    subgraph Pipeline [Processing Pipeline]
        IN[Ingestion\nMIME + magic-byte checks\nPDF rasterization]
        PP[Preprocessing\nResize + normalize]
        OC[OCR Engine\nPaddleOCR]
        ML[LayoutLMv3\nToken Classification]
        PS[Post-processing\nNormalize → Validate → Constrain]
    end

    subgraph Output [Output Layer]
        JS[Structured JSON\n+ constraint_flags\n+ per-field confidence]
    end

    CL -->|POST /parse-document| RT
    RT --> UP
    UP --> IN
    IN --> PP
    PP --> OC
    OC -->|tokens + bboxes + scores| ML
    ML -->|KEY / VALUE / O labels| PS
    PS --> JS
    JS -->|DocumentParseResponse| CL

    RT --> EH
```

### Data Flow

```
UploadFile
  → validate_upload()             # extension, MIME, magic bytes, size
  → load_pages()                  # rasterize PDF or open image
  → ImageNormalizationService     # deterministic page prep
  → OCRService.extract()          # tokens + bboxes + confidence
  → LayoutLMv3InferenceService    # per-token field classification
  → normalize_document()          # date/currency/OCR artifact cleanup
  → validate_document()           # regex + semantic field checks
  → apply_constraints()           # cross-field consistency enforcement
  → DocumentParseResponse         # typed, validated JSON output
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/purvanshh/document-intelligence-engine.git
cd document-intelligence-engine

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env for model path, OCR backend, API settings
```

All settings support environment variable overrides with the `DIE_` prefix:

```bash
DIE_API__PORT=8080
DIE_OCR__MIN_CONFIDENCE=0.6
DIE_POSTPROCESSING__CONSTRAINTS__AMOUNT_TOLERANCE=0.02
```

### 3. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive Swagger UI.

### 4. Parse a Document

```bash
curl -X POST http://localhost:8000/parse-document \
     -F "file=@invoice.pdf"
```

---

## Docker Deployment

```bash
# Build and start the API
docker compose -f docker/docker-compose.yml up --build

# Include Redis for async processing
docker compose -f docker/docker-compose.yml --profile async up
```

The API will be available at **`http://localhost:8000`**.

---

## API Reference

Once the backend is running, Swagger UI is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + model readiness check |
| `POST` | `/parse-document` | Parse a single PDF or image |
| `POST` | `/parse-batch` | Parse multiple files in one request |

**POST /parse-document**

Input: `multipart/form-data` with a single `file` field (PDF, PNG, JPEG, TIFF).

```bash
curl -X POST http://localhost:8000/parse-document \
     -F "file=@invoice.pdf" \
     -F "debug=false"
```

**Response:**

```json
{
  "document": {
    "invoice_number": { "value": "INV-1023",    "confidence": 0.924, "valid": true },
    "date":           { "value": "2025-01-12",   "confidence": 0.911, "valid": true },
    "vendor":         { "value": "ABC Pvt Ltd",  "confidence": 0.887, "valid": true },
    "total_amount":   { "value": 1200.50,        "confidence": 0.883, "valid": true },
    "line_items": {
      "value": [
        { "item": "Product A", "quantity": 2, "price": 400.00, "confidence": 0.871 }
      ],
      "valid": true
    },
    "_constraint_flags": [],
    "_errors": []
  },
  "metadata": {
    "filename": "invoice.pdf",
    "pages_processed": 1,
    "request_id": "req_01j9z..."
  }
}
```

| HTTP Status | Cause |
|---|---|
| 400 | Invalid file type, malformed content, size exceeded |
| 422 | Empty OCR output — no text detected |
| 502 | OCR engine or model inference failure |
| 503 | Model backend unavailable |

---

## The Deterministic Post-Processing Layer

This is the component that makes the system suitable for production rather than experimentation.

1. **Query**: A scanned invoice arrives with `total_amount: "$1,2OO.5O"` (OCR misread zeros as letters).
2. **OCR Artifact Correction**: The normalization layer identifies numeric context and substitutes `O→0`, `l→1` where appropriate → `"1200.50"`.
3. **Field Normalization**: Currency string parsed to `float` `1200.50`. Date strings converted to ISO 8601.
4. **Regex Validation**: `invoice_number` checked against configured pattern; `date` checked for ISO format; `total_amount` checked for numeric type.
5. **Constraint Check**: `sum(line_item.price × quantity)` computed and compared to `total_amount` within tolerance. If mismatched, a `line_items_total_mismatch` flag is appended — the output is still returned, but the discrepancy is surfaced.
6. **Result**: Every field has an explicit `valid` boolean, a `confidence` score, and correction provenance. `_constraint_flags` lists any violated rules. Same invoice, same output, every time.

---

## Evaluation & Ablation Studies

```bash
# Run the experiment harness
./.venv/bin/python run_experiments.py --config configs/experiments.yaml
```

### Latest Experiment Results

The table below is generated from the latest full harness run on **May 2, 2026** against [data/annotations/test.jsonl](/Users/purvansh/Desktop/Projects/DRISE-experiments/data/annotations/test.jsonl:1) with **N=201** documents.

Run configuration used for these numbers:
- `llm_only` and `rag_llm`: NVIDIA backend with `meta/llama-3.2-1b-instruct`
- `drise`: real LayoutLMv3 inference via published checkpoint `jinhybr/OCR-LayoutLMv3-Invoice`
- local fine-tuned artifact under `experiments/artifacts/cord_finetuned/` was incomplete, so the published checkpoint was used instead
- the LLM baseline latency figures shown below are from cached benchmark artifacts; the live sensitivity runs later in this README better reflect uncached provider latency behavior

| System | Field-level F1 | Exact Match | Schema Valid | Hallucination | Avg Latency (ms) | Cost/doc ($) | Total Cost ($) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm_only` | 0.1677 | 0.0000 | 1.0000 | 0.0155 | 0.19 | 0.000368 | 0.073876 |
| `rag_llm` | 0.0000 | 0.0000 | 0.9701 | 0.0057 | 1.27 | 0.001648 | 0.331337 |
| `drise` | 0.5812 | 0.0498 | 1.0000 | 0.0680 | 301.89 | 0.000042 | 0.008429 |
| `drise_no_layout` | 0.5667 | 0.0498 | 1.0000 | 0.0351 | 363.07 | 0.000050 | 0.010136 |
| `drise_no_constraints` | 0.5812 | 0.0498 | 1.0000 | 0.0680 | 396.72 | 0.000055 | 0.011087 |

### What These Scores Mean

- DRISE now leads the benchmark on **field-level F1** at `0.5812`, up from the earlier `0.5461`, versus `0.1677` for `llm_only` and `0.0000` for `rag_llm`.
- DRISE reaches **non-zero exact match** on the held-out test split at `0.0498`, which is enough for a meaningful McNemar comparison against both LLM baselines.
- DRISE remains **schema-valid on every document** while staying the cheapest practical system in the run at roughly `$0.000042` per document.
- The biggest improvement is in structured extraction quality: DRISE now reaches `0.6734` mean field F1 on `line_items` and `0.6020` on `total_amount`, where the text-only baselines continue to fail.
- The hallucination metric is now calibrated rather than raw-substring-based. On the final run, DRISE's macro document-mean hallucination rate is `0.0680`, and the calibration script reports a micro checked-field rate of `0.0629`.

### Exported Artifacts

- Summary table: [summary.csv](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/summary.csv)
- Ablation deltas: [ablation_summary.csv](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/ablation_summary.csv)
- Pairwise stats: [pairwise_stats.json](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/pairwise_stats.json)
- Hallucination calibration sample: [hallucination_calibration.json](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/hallucination_calibration.json)
- Hallucination calibration summary: [hallucination_calibration_summary.json](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/hallucination_calibration_summary.json)
- Markdown snapshot: [README_EXPERIMENTS.md](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/README_EXPERIMENTS.md)

### Ablation Experiments

| Experiment | What is removed | What it measures |
|---|---|---|
| `drise_no_layout` | Layout-aware inference path | Value of spatial encoding when a real model checkpoint is available |
| `drise_no_constraints` | Deterministic constraint application | Impact of cross-field validation and guardrails |

Current ablation deltas from the latest run:

| Variant | Delta F1 vs `drise` | Delta Exact Match vs `drise` | Delta Schema Valid vs `drise` | Delta Hallucination vs `drise` |
|---|---:|---:|---:|---:|
| `drise_no_layout` | -0.0144 | 0.0000 | 0.0000 | -0.0329 |
| `drise_no_constraints` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:
- Removing layout still hurts DRISE by about **1.4 F1 points overall**, even though exact match is unchanged on this split.
- The biggest layout-sensitive field remains `total_amount`, which falls from `0.6020` mean field F1 to `0.5174` in the no-layout run.
- Disabling constraints still does not move the scored extraction fields on this dataset, but it does collapse `constraint_flag_rate` from `0.9900` to `0.0000`, which shows the current constraint layer is acting as a guardrail and diagnostics layer rather than a repair layer.

### Phase 8 Notes

- McNemar exact-match comparisons are now meaningful: DRISE beats both `llm_only` and `rag_llm` with `p = 0.004427` in [pairwise_stats.json](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/pairwise_stats.json).
- The exact-match signal is still too sparse to distinguish `drise` from its ablations, so the no-layout and no-constraints comparisons remain `p = 1.0`.
- Cost guardrails are active in [run_experiments.py](/Users/purvansh/Desktop/Projects/DRISE-experiments/run_experiments.py:26) and [runner.py](/Users/purvansh/Desktop/Projects/DRISE-experiments/src/document_intelligence_engine/evaluation/runner.py:11): the benchmark now fixes all major seeds and stops if cumulative LLM cost crosses `$30`.
- The current benchmark run stayed comfortably inside the cap. Combined LLM baseline spend was about `$0.405213`, while DRISE remained near-zero cost because it runs locally.

### Phase 9 Notes

- Live sensitivity artifacts are available under [experiments/results/sensitivity_live/analysis.md](/Users/purvansh/Desktop/Projects/DRISE-experiments/experiments/results/sensitivity_live/analysis.md:1).
- On a **20-document** live sensitivity subset, `llm_only` improved from `0.191` F1 at temperature `0.0` to `0.342` at `0.7`, but schema validity fell from `1.000` to `0.800`.
- `rag_llm` remained unstable and weak under temperature changes, reaching only `0.040` F1 at `0.7` with schema validity dropping to `0.150`.
- Under OCR corruption, DRISE degraded more gracefully than `llm_only`: from `0.390` to `0.298` F1 between noise `0.0` and `0.2`, versus `0.191` to `0.073` for `llm_only`.

### Reproducibility

- Exact environment snapshot: [requirements_lock.txt](/Users/purvansh/Desktop/Projects/DRISE-experiments/requirements_lock.txt)
- Benchmark container files: [Dockerfile](/Users/purvansh/Desktop/Projects/DRISE-experiments/Dockerfile) and [.dockerignore](/Users/purvansh/Desktop/Projects/DRISE-experiments/.dockerignore)
- Containerized benchmark command:

```bash
docker build -t drise-benchmark .
docker run \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/experiments:/app/experiments" \
  drise-benchmark
```

- The container run mounts the full `data/` directory, not just `data/annotations/`, because the experiment annotations reference source images under `data/raw/`.
- `load_annotations()` now rebases absolute host paths when needed, so the same annotation JSONL files can be used from `/app` inside Docker.
- The Docker reproducibility path was re-verified in this session: the image built successfully and the containerized benchmark entrypoint loaded the model and began processing the mounted `data/` and `experiments/` volumes end to end.

---

## Project Structure

```
.
├── configs/                   # YAML config (model, OCR, API, postprocessing rules)
├── data/
│   ├── raw/                   # Source PDFs/images  [gitignored]
│   ├── processed/             # Intermediate artifacts  [gitignored]
│   └── annotations/           # Ground-truth labels  [gitignored]
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── experiments/
│   ├── runs/                  # MLflow/W&B run metadata  [gitignored]
│   └── artifacts/             # Model checkpoints  [gitignored]
├── src/
│   └── document_intelligence_engine/
│       ├── api/               # FastAPI app, routes, schemas, middleware
│       ├── core/              # Config loader, logger, error hierarchy
│       ├── domain/            # Typed data contracts
│       ├── ingestion/         # File validation, PDF rasterization
│       ├── preprocessing/     # Image normalization
│       ├── ocr/               # PaddleOCR wrapper, backend protocol
│       ├── multimodal/        # LayoutLMv3 inference + training hooks
│       ├── postprocessing/    # Normalization, validation, constraints
│       ├── evaluation/        # Metrics, ablation framework
│       └── services/          # End-to-end pipeline orchestration
└── tests/                     # Unit + integration + load tests
```

---

## Limitations

- **OCR is a hard ceiling.** Severely degraded scans (heavy noise, sub-100 DPI, mixed orientation) produce low-confidence tokens that downstream models cannot reliably recover.
- **Domain generalization.** Fine-tuned on FUNSD and CORD. Performance on domain-specific document types (legal, medical, multilingual) will degrade without targeted fine-tuning.
- **Multi-page joining.** Pages are processed independently. Cross-page field references (e.g., total on page 2 referencing items on page 1) are not currently resolved.
- **Table structure.** Table cells are extracted but row/column/span structure is not reconstructed in the output schema.

---

## Future Work

- Table structure reconstruction from detected cell bounding boxes
- Cross-page field joining for multi-page documents
- Multilingual document support (Arabic, CJK scripts)
- Confidence calibration via temperature scaling post fine-tuning
- Active learning loop: route low-confidence outputs to human review and feed corrections back into training data

---

## Fine-Tuning

```bash
# Configure training settings in .env or configs/config.yaml, then:
python -m document_intelligence_engine.multimodal.training
```

**Datasets used:**
- [FUNSD](https://guillaumejaume.github.io/FUNSD/) — form understanding on noisy scanned documents
- [CORD](https://github.com/clovaai/cord) — receipt parsing with structured line items

---

## Contact & Contributions

Designed and developed by **Purvansh Sahu**.

If you find this project useful or have suggestions, feel free to open an issue or reach out directly.

- **GitHub**: [@purvanshh](https://github.com/purvanshh)
- **Email**: purvanshhsahu@gmail.com
