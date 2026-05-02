# Sensitivity Analysis

- Backend: `nvidia`
- Sample count: `20`

## Temperature Sweep

| System | Temperature | Field F1 | Exact Match | Schema Valid |
| --- | ---: | ---: | ---: | ---: |
| llm_only | 0.0 | 0.191 | 0.000 | 1.000 |
| rag_llm | 0.0 | 0.000 | 0.000 | 1.000 |
| llm_only | 0.3 | 0.301 | 0.000 | 1.000 |
| rag_llm | 0.3 | 0.000 | 0.000 | 1.000 |
| llm_only | 0.7 | 0.342 | 0.000 | 0.800 |
| rag_llm | 0.7 | 0.040 | 0.000 | 0.150 |

## OCR Noise Sweep

| System | Noise Factor | Field F1 | Exact Match | Schema Valid |
| --- | ---: | ---: | ---: | ---: |
| llm_only | 0.0 | 0.191 | 0.000 | 1.000 |
| rag_llm | 0.0 | 0.000 | 0.000 | 1.000 |
| drise | 0.0 | 0.390 | 0.000 | 1.000 |
| llm_only | 0.1 | 0.135 | 0.000 | 0.950 |
| rag_llm | 0.1 | 0.000 | 0.000 | 1.000 |
| drise | 0.1 | 0.336 | 0.000 | 1.000 |
| llm_only | 0.2 | 0.073 | 0.000 | 1.000 |
| rag_llm | 0.2 | 0.000 | 0.000 | 1.000 |
| drise | 0.2 | 0.298 | 0.000 | 1.000 |
