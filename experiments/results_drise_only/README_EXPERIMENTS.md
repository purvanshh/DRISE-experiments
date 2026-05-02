## Experimental Results

| System | Field F1 | Exact Match | Schema Valid | Hallucination | Cost/doc |
| --- | --- | --- | --- | --- | --- |
| drise | 0.546 | 0.015 | 1.000 | 0.112 | $0.0000 |
| drise_no_layout | 0.531 | 0.015 | 1.000 | 0.116 | $0.0000 |
| drise_no_constraints | 0.546 | 0.015 | 1.000 | 0.112 | $0.0001 |

### Significance

- drise_vs_drise_no_layout: p=1.000000, statistic=0.000000
- drise_vs_drise_no_constraints: p=1.000000, statistic=0.000000
- drise_no_layout_vs_drise_no_constraints: p=1.000000, statistic=0.000000
