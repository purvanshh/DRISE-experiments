# Sensitivity Analysis

- Backend: `nvidia`
- Sample count: `5`

## Temperature Sweep

| System | Temperature | Field F1 | Exact Match | Schema Valid |
| --- | ---: | ---: | ---: | ---: |
| llm_only | 0.0 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.0 | 0.000 | 0.000 | 1.000 |
| llm_only | 0.3 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.3 | 0.000 | 0.000 | 1.000 |
| llm_only | 0.7 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.7 | 0.000 | 0.000 | 1.000 |

## OCR Noise Sweep

| System | Noise Factor | Field F1 | Exact Match | Schema Valid |
| --- | ---: | ---: | ---: | ---: |
| llm_only | 0.0 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.0 | 0.000 | 0.000 | 1.000 |
| drise | 0.0 | 0.678 | 0.000 | 1.000 |
| llm_only | 0.1 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.1 | 0.000 | 0.000 | 1.000 |
| drise | 0.1 | 0.533 | 0.000 | 1.000 |
| llm_only | 0.2 | 0.000 | 0.000 | 1.000 |
| rag_llm | 0.2 | 0.000 | 0.000 | 1.000 |
| drise | 0.2 | 0.471 | 0.000 | 1.000 |
