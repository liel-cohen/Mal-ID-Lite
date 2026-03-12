# Model 1 Training Results

**Timestamp**: 20260312_100308
**Model**: lasso_cv
**Parameters**: {'gene_locus': 'TCR', 'n_pcs': 15}

---

## Overall Performance

### Global Metrics (Following Paper Methodology)

**Accuracy**:
- **Global (concatenated predictions)**: 0.702
- Per-fold average: 0.702 ± 0.026

### Probability-Based Metrics (Per-Fold Then Averaged)

**Primary Metrics**:
- **AUROC (OvO, weighted)**: **0.943 ± 0.007**
- **AUPRC (OvO, weighted)**: **0.810 ± 0.025**

**Other**:
- Log loss: 0.739 ± 0.040

### Per-Class AUROC (OvR) - Individual Disease Performance

| Disease | AUROC (OvR) | Std Dev | Performance |
|---------|-------------|---------|-------------|
| Covid19 | 0.951 | ±0.008 | Excellent |
| HIV | 0.967 | ±0.004 | Excellent |
| Healthy/Background | 0.925 | ±0.008 | Very Good |
| Influenza | 0.971 | ±0.019 | Excellent |
| Lupus | 0.921 | ±0.013 | Very Good |
| T1D | 0.938 | ±0.028 | Very Good |

### Per-Fold Results

| Fold | Accuracy | AUROC (OvO) | AUPRC (OvO) | Log Loss |
|------|----------|-------------|-------------|----------|
| 0 | 0.674 | 0.935 | 0.781 | 0.772 |
| 1 | 0.726 | 0.946 | 0.829 | 0.752 |
| 2 | 0.706 | 0.948 | 0.821 | 0.695 |

## Aggregated Confusion Matrix (All Folds)

```
True Label → Predicted Label

                               Covid19                 HIV  Healthy/Background           Influenza               Lupus                 T1D  │ Total
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Covid19                             35                   0                   4                   0                   8                  11  │   58
HIV                                  2                  83                   4                   7                   2                   0  │   98
Healthy/Background                   7                  29                 133                   9                  14                   5  │  197
Influenza                            0                   2                   5                  30                   0                   0  │   37
Lupus                                5                   0                   4                   2                  46                   7  │   64
T1D                                 12                   2                   1                   3                  19                  59  │   96
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Total                               61                 116                 151                  51                  89                  82  │  550

Diagonal (correct): 386 / 550 = 70.2% accuracy
```

### Per-Class Accuracy

| Disease | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Covid19 | 35 | 58 | 60.3% |
| HIV | 83 | 98 | 84.7% |
| Healthy/Background | 133 | 197 | 67.5% |
| Influenza | 30 | 37 | 81.1% |
| Lupus | 46 | 64 | 71.9% |
| T1D | 59 | 96 | 61.5% |

## Individual Fold Confusion Matrices

### Fold 0

```
True Label → Predicted Label

                               Covid19                 HIV  Healthy/Background           Influenza               Lupus                 T1D  │ Total
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Covid19                             10                   0                   3                   0                   3                   3  │   19
HIV                                  1                  28                   2                   2                   0                   0  │   33
Healthy/Background                   4                   9                  43                   4                   4                   2  │   66
Influenza                            0                   1                   2                   9                   0                   0  │   12
Lupus                                0                   0                   0                   1                  18                   3  │   22
T1D                                  4                   1                   0                   0                  11                  16  │   32
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Total                               19                  39                  50                  16                  36                  24  │  184

Diagonal (correct): 124 / 184 = 67.4% accuracy
```

**Per-class AUROC (OvR) - Fold 0**:

- Covid19: 0.942
- HIV: 0.966
- Healthy/Background: 0.921
- Influenza: 0.950
- Lupus: 0.922
- T1D: 0.926

### Fold 1

```
True Label → Predicted Label

                               Covid19                 HIV  Healthy/Background           Influenza               Lupus                 T1D  │ Total
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Covid19                             14                   0                   0                   0                   2                   4  │   20
HIV                                  1                  28                   1                   3                   0                   0  │   33
Healthy/Background                   1                  11                  48                   3                   3                   0  │   66
Influenza                            0                   0                   1                  11                   0                   0  │   12
Lupus                                2                   0                   2                   1                  14                   2  │   21
T1D                                  6                   1                   1                   2                   4                  20  │   34
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Total                               24                  40                  53                  20                  23                  26  │  186

Diagonal (correct): 135 / 186 = 72.6% accuracy
```

**Per-class AUROC (OvR) - Fold 1**:

- Covid19: 0.958
- HIV: 0.963
- Healthy/Background: 0.935
- Influenza: 0.985
- Lupus: 0.933
- T1D: 0.918

### Fold 2

```
True Label → Predicted Label

                               Covid19                 HIV  Healthy/Background           Influenza               Lupus                 T1D  │ Total
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Covid19                             11                   0                   1                   0                   3                   4  │   19
HIV                                  0                  27                   1                   2                   2                   0  │   32
Healthy/Background                   2                   9                  42                   2                   7                   3  │   65
Influenza                            0                   1                   2                  10                   0                   0  │   13
Lupus                                3                   0                   2                   0                  14                   2  │   21
T1D                                  2                   0                   0                   1                   4                  23  │   30
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────
Total                               18                  37                  48                  15                  30                  32  │  180

Diagonal (correct): 127 / 180 = 70.6% accuracy
```

**Per-class AUROC (OvR) - Fold 2**:

- Covid19: 0.955
- HIV: 0.971
- Healthy/Background: 0.921
- Influenza: 0.979
- Lupus: 0.907
- T1D: 0.970

---

*Generated by Model 1 training script*
