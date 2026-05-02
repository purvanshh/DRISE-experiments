## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| llm_only | 0.168 | 0.000 | 1.000 | 0.016 | $0.0004 |
| rag_llm | 0.000 | 0.000 | 0.970 | 0.006 | $0.0016 |
| drise | 0.581 | 0.050 | 1.000 | 0.068 | $0.0000 |
| drise_no_layout | 0.567 | 0.050 | 1.000 | 0.035 | $0.0001 |
| drise_no_constraints | 0.581 | 0.050 | 1.000 | 0.068 | $0.0001 |

### Significance

- llm_only_vs_rag_llm: p=1.000000, statistic=0.000000
- llm_only_vs_drise: p=0.004427, statistic=8.100000
- llm_only_vs_drise_no_layout: p=0.004427, statistic=8.100000
- llm_only_vs_drise_no_constraints: p=0.004427, statistic=8.100000
- rag_llm_vs_drise: p=0.004427, statistic=8.100000
- rag_llm_vs_drise_no_layout: p=0.004427, statistic=8.100000
- rag_llm_vs_drise_no_constraints: p=0.004427, statistic=8.100000
- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=1.000000, statistic=0.000000
