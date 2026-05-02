## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| drise | 0.546 | 0.015 | 1.000 | 0.112 | $0.0000 |
| drise_no_constraints | 0.546 | 0.015 | 1.000 | 0.112 | $0.0001 |
| drise_no_layout | 0.531 | 0.015 | 1.000 | 0.116 | $0.0000 |
| llm_only | 0.168 | 0.000 | 1.000 | 0.017 | $0.0004 |
| rag_llm | 0.000 | 0.000 | 0.970 | 0.006 | $0.0016 |

### Significance

- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_llm_only: p=0.248213, statistic=1.333333
- drise_vs_rag_llm: p=0.248213, statistic=1.333333
- drise_no_constraints_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_no_constraints_vs_llm_only: p=0.248213, statistic=1.333333
- drise_no_constraints_vs_rag_llm: p=0.248213, statistic=1.333333
- drise_no_layout_vs_llm_only: p=0.248213, statistic=1.333333
- drise_no_layout_vs_rag_llm: p=0.248213, statistic=1.333333
- llm_only_vs_rag_llm: p=1.000000, statistic=0.000000
