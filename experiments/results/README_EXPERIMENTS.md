## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| llm_only | 0.460 | 0.144 | 1.000 | 0.050 | $0.0002 |
| rag_llm | 0.485 | 0.070 | 0.861 | 0.032 | $0.0005 |
| llm_only_strong | 0.397 | 0.065 | 1.000 | 0.037 | $0.0003 |
| rag_llm_strong | 0.493 | 0.075 | 0.891 | 0.039 | $0.0011 |
| drise | 0.581 | 0.050 | 1.000 | 0.068 | $0.0000 |
| drise_no_layout | 0.567 | 0.050 | 1.000 | 0.035 | $0.0001 |
| drise_no_constraints | 0.581 | 0.050 | 1.000 | 0.068 | $0.0001 |

### Significance

- llm_only_vs_rag_llm: p=0.007054, statistic=7.259259
- llm_only_vs_llm_only_strong: p=0.000796, statistic=11.250000
- llm_only_vs_rag_llm_strong: p=0.005578, statistic=7.681818
- llm_only_vs_drise: p=0.003085, statistic=8.756757
- llm_only_vs_drise_no_layout: p=0.003085, statistic=8.756757
- llm_only_vs_drise_no_constraints: p=0.003085, statistic=8.756757
- rag_llm_vs_llm_only_strong: p=1.000000, statistic=0.000000
- rag_llm_vs_rag_llm_strong: p=1.000000, statistic=0.000000
- rag_llm_vs_drise: p=0.522431, statistic=0.409091
- rag_llm_vs_drise_no_layout: p=0.522431, statistic=0.409091
- rag_llm_vs_drise_no_constraints: p=0.522431, statistic=0.409091
- llm_only_strong_vs_rag_llm_strong: p=0.813664, statistic=0.055556
- llm_only_strong_vs_drise: p=0.676657, statistic=0.173913
- llm_only_strong_vs_drise_no_layout: p=0.676657, statistic=0.173913
- llm_only_strong_vs_drise_no_constraints: p=0.676657, statistic=0.173913
- rag_llm_strong_vs_drise: p=0.404248, statistic=0.695652
- rag_llm_strong_vs_drise_no_layout: p=0.404248, statistic=0.695652
- rag_llm_strong_vs_drise_no_constraints: p=0.404248, statistic=0.695652
- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=1.000000, statistic=0.000000
