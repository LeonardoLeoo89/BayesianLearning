# Structural Accuracy Report

This table compares the learned networks against the true ground-truth network structures that generated the data.

- **SHD**: Structural Hamming Distance (lower is better, 0 is perfect). The number of edge additions, deletions, or reversals needed to match the true graph.
- **TP**: True Positives (correct edges)
- **FP**: False Positives (extra edges)
- **FN**: False Negatives (missing edges)
- **F1**: F1 Score (higher is better, 1.0 is perfect)
- **KL_Div**: Kullback-Leibler Divergence (lower is better, 0 is perfect). Measures the statistical information lost due to structural errors by applying parameter learning to both the true and inferred structures.

| Algorithm_Type   | Dataset         | Algorithm                   |   SHD |   TP |   FP |   FN |       F1 |      KL_Div |   Hellinger |   Bhattacharyya |   Jensen_Shannon |
|:-----------------|:----------------|:----------------------------|------:|-----:|-----:|-----:|---------:|------------:|------------:|----------------:|-----------------:|
| Categorical      | allergy         | FCI_(Tetrad)                |     0 |    7 |    0 |    0 | 1        |   0.0124647 |   0.0647939 |      0.00210131 |       0.00301943 |
| Categorical      | allergy         | Genetic_K2                  |     9 |    7 |    3 |    0 | 0.823529 |   0.0655659 |   0.118195  |      0.0070095  |       0.00944882 |
| Categorical      | allergy         | Hill_Climbing_(Agrum)       |     3 |    7 |    1 |    0 | 0.933333 |   0.0199188 |   0.0701452 |      0.00246318 |       0.00339852 |
| Categorical      | allergy         | PC_(Tetrad)                 |     0 |    7 |    0 |    0 | 1        |   0.0124647 |   0.0647939 |      0.00210131 |       0.00301943 |
| Categorical      | allergy         | RFCI_(Tetrad)               |     0 |    7 |    0 |    0 | 1        |   0.0124647 |   0.0647939 |      0.00210131 |       0.00301943 |
| Categorical      | allergy         | Structural_EM               |     9 |    7 |    5 |    0 | 0.736842 |   0.0674113 |   0.0914554 |      0.00419079 |       0.00507489 |
| Categorical      | train_delay     | FCI_Tetrad                  |     6 |    8 |    0 |    5 | 0.761905 |   0.529515  |   0.435105  |      0.0994425  |       0.125976   |
| Categorical      | train_delay     | Genetic_K2                  |    12 |    8 |    2 |    5 | 0.695652 |   0.348009  |   0.341124  |      0.0599443  |       0.0787439  |
| Categorical      | train_delay     | Hill_Climbing_Agrum         |    11 |    8 |    2 |    5 | 0.695652 |   0.214811  |   0.261629  |      0.0348242  |       0.047872   |
| Categorical      | train_delay     | PC_Tetrad                   |     6 |    8 |    0 |    5 | 0.761905 |   0.529515  |   0.435105  |      0.0994425  |       0.125976   |
| Categorical      | train_delay     | RFCI_Tetrad                 |     6 |    8 |    0 |    5 | 0.761905 |   0.549595  |   0.440207  |      0.101912   |       0.129025   |
| Categorical      | train_delay     | Structural_EM               |    17 |    5 |    7 |    8 | 0.4      |   1.18568   |   0.600496  |      0.198814   |       0.233415   |
| Categorical      | tsunami         | FCI_(Tetrad)                |     0 |    4 |    0 |    0 | 1        |   0.0563319 |   0.0942228 |      0.00444882 |       0.00558391 |
| Categorical      | tsunami         | Genetic_K2                  |     4 |    3 |    1 |    1 | 0.75     |   0.0661091 |   0.0950345 |      0.00452598 |       0.00564523 |
| Categorical      | tsunami         | Hill_Climbing_(Agrum)       |     5 |    3 |    1 |    1 | 0.75     |   0.066109  |   0.0950342 |      0.00452595 |       0.00564519 |
| Categorical      | tsunami         | PC_(Tetrad)                 |     0 |    4 |    0 |    0 | 1        |   0.0563319 |   0.0942228 |      0.00444882 |       0.00558391 |
| Categorical      | tsunami         | RFCI_(Tetrad)               |     0 |    4 |    0 |    0 | 1        |   0.0563319 |   0.0942228 |      0.00444882 |       0.00558391 |
| Categorical      | tsunami         | Structural_EM               |     5 |    2 |    1 |    2 | 0.571429 |   0.0594789 |   0.136205  |      0.00931922 |       0.0120327  |
| SEM              | allergy_sem     | subset_500_std.csv_DAGMA    |     7 |    5 |    3 |    2 | 0.666667 | nan         | nan         |    nan          |     nan          |
| SEM              | allergy_sem     | subset_500_std.csv_DAG_GNN  |     9 |    4 |    5 |    3 | 0.5      | nan         | nan         |    nan          |     nan          |
| SEM              | allergy_sem     | subset_500_std.csv_GraN-DAG |    11 |    6 |    9 |    1 | 0.545455 | nan         | nan         |    nan          |     nan          |
| SEM              | train_delay_sem | subset_500_std.csv_DAGMA    |    16 |    8 |    4 |    5 | 0.64     | nan         | nan         |    nan          |     nan          |
| SEM              | train_delay_sem | subset_500_std.csv_DAG_GNN  |    26 |   13 |   15 |    0 | 0.634146 | nan         | nan         |    nan          |     nan          |
| SEM              | train_delay_sem | subset_500_std.csv_GraN-DAG |    42 |   13 |   37 |    0 | 0.412698 | nan         | nan         |    nan          |     nan          |
| SEM              | tsunami_sem     | subset_500_std.csv_DAGMA    |     5 |    4 |    1 |    0 | 0.888889 | nan         | nan         |    nan          |     nan          |
| SEM              | tsunami_sem     | subset_500_std.csv_DAG_GNN  |     4 |    4 |    2 |    0 | 0.8      | nan         | nan         |    nan          |     nan          |
| SEM              | tsunami_sem     | subset_500_std.csv_GraN-DAG |     2 |    4 |    2 |    0 | 0.8      | nan         | nan         |    nan          |     nan          |