## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| llm_only | 0.000 | 0.000 | 1.000 | 0.000 | $0.0000 |
| rag_llm | 0.000 | 0.000 | 1.000 | 0.000 | $0.0000 |
| drise | 0.533 | 0.000 | 0.000 | 0.895 | $0.0000 |
| drise_no_layout | 0.482 | 0.000 | 0.000 | 0.932 | $0.0000 |
| drise_no_constraints | 0.533 | 0.000 | 0.000 | 0.895 | $0.0001 |

### Significance

- llm_only_vs_rag_llm: p=1.000000, statistic=0.000000
- llm_only_vs_drise: p=1.000000, statistic=0.000000
- llm_only_vs_drise_no_layout: p=1.000000, statistic=0.000000
- llm_only_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- rag_llm_vs_drise: p=1.000000, statistic=0.000000
- rag_llm_vs_drise_no_layout: p=1.000000, statistic=0.000000
- rag_llm_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=1.000000, statistic=0.000000
