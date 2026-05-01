## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| llm_only | 1.000 | 1.000 | 1.000 | 0.100 | $0.0000 |
| rag_llm | 1.000 | 1.000 | 1.000 | 0.100 | $0.0000 |
| drise | 0.311 | 0.000 | 0.000 | 0.000 | $0.0000 |
| drise_no_layout | 0.311 | 0.000 | 0.000 | 0.000 | $0.0000 |
| drise_no_constraints | 0.311 | 0.000 | 0.000 | 0.000 | $0.0000 |

### Significance

- llm_only_vs_rag_llm: p=1.000000, statistic=0.000000
- llm_only_vs_drise: p=0.479500, statistic=0.500000
- llm_only_vs_drise_no_layout: p=0.479500, statistic=0.500000
- llm_only_vs_drise_no_constraints: p=0.479500, statistic=0.500000
- rag_llm_vs_drise: p=0.479500, statistic=0.500000
- rag_llm_vs_drise_no_layout: p=0.479500, statistic=0.500000
- rag_llm_vs_drise_no_constraints: p=0.479500, statistic=0.500000
- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=1.000000, statistic=0.000000
