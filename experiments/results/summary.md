## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| llm_only | 0.486 | 0.239 | 1.000 | 0.050 | $0.0002 |
| rag_llm | 0.503 | 0.090 | 0.861 | 0.032 | $0.0005 |
| llm_only_strong | 0.486 | 0.239 | 1.000 | 0.050 | $0.0002 |
| rag_llm_strong | 0.503 | 0.090 | 0.861 | 0.032 | $0.0005 |
| drise | 0.625 | 0.249 | 1.000 | 0.180 | $0.0000 |
| drise_no_layout | 0.625 | 0.249 | 1.000 | 0.180 | $0.0001 |
| drise_no_constraints | 0.625 | 0.249 | 1.000 | 0.180 | $0.0001 |

### Significance

- llm_only_vs_rag_llm: p=0.000008, statistic=20.023810
- llm_only_vs_llm_only_strong: p=1.000000, statistic=0.000000
- llm_only_vs_rag_llm_strong: p=0.000008, statistic=20.023810
- llm_only_vs_drise: p=0.882783, statistic=0.021739
- llm_only_vs_drise_no_layout: p=0.885234, statistic=0.020833
- llm_only_vs_drise_no_constraints: p=0.882783, statistic=0.021739
- rag_llm_vs_llm_only_strong: p=0.000008, statistic=20.023810
- rag_llm_vs_rag_llm_strong: p=1.000000, statistic=0.000000
- rag_llm_vs_drise: p=0.000003, statistic=21.840909
- rag_llm_vs_drise_no_layout: p=0.000003, statistic=21.840909
- rag_llm_vs_drise_no_constraints: p=0.000003, statistic=21.840909
- llm_only_strong_vs_rag_llm_strong: p=0.000008, statistic=20.023810
- llm_only_strong_vs_drise: p=0.882783, statistic=0.021739
- llm_only_strong_vs_drise_no_layout: p=0.885234, statistic=0.020833
- llm_only_strong_vs_drise_no_constraints: p=0.882783, statistic=0.021739
- rag_llm_strong_vs_drise: p=0.000003, statistic=21.840909
- rag_llm_strong_vs_drise_no_layout: p=0.000003, statistic=21.840909
- rag_llm_strong_vs_drise_no_constraints: p=0.000003, statistic=21.840909
- drise_vs_drise_no_layout: p=0.479500, statistic=0.500000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=0.479500, statistic=0.500000

### Conditional Field F1 (only docs where the field exists)

| System | Field | F1 | Docs |
| --- | --- | --- | --- |
| llm_only | date | 0.222 | 3 |
| llm_only | line_items | 0.585 | 166 |
| llm_only | total_amount | 0.386 | 197 |
| rag_llm | date | 0.333 | 3 |
| rag_llm | line_items | 0.591 | 166 |
| rag_llm | total_amount | 0.680 | 197 |
| llm_only_strong | date | 0.222 | 3 |
| llm_only_strong | line_items | 0.585 | 166 |
| llm_only_strong | total_amount | 0.386 | 197 |
| rag_llm_strong | date | 0.333 | 3 |
| rag_llm_strong | line_items | 0.591 | 166 |
| rag_llm_strong | total_amount | 0.680 | 197 |
| drise | date | 0.222 | 3 |
| drise | line_items | 0.737 | 166 |
| drise | total_amount | 0.589 | 197 |
| drise_no_layout | date | 0.222 | 3 |
| drise_no_layout | line_items | 0.737 | 166 |
| drise_no_layout | total_amount | 0.589 | 197 |
| drise_no_constraints | date | 0.222 | 3 |
| drise_no_constraints | line_items | 0.738 | 166 |
| drise_no_constraints | total_amount | 0.589 | 197 |
