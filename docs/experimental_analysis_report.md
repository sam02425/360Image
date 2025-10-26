# Multi-View Object Detection Experiment Analysis

## Experimental Data Summary
- Total experiments completed: 4
- Architectures tested: yolov11n, yolov8n
- View configurations tested: dual

## Performance Results

### Yolov11N Dual

| Metric | Mean ± SD | 95% CI | N |
|--------|-----------|--------|---|
| map50 | 0.932 ± 0.004 | [0.900, 0.965] | 2 |
| map50_95 | 0.913 ± 0.003 | [0.883, 0.942] | 2 |
| precision | 0.837 ± 0.003 | [0.808, 0.866] | 2 |
| recall | 0.857 ± 0.003 | [0.826, 0.888] | 2 |
| f1_score | 0.847 ± 0.003 | [0.817, 0.877] | 2 |

### Yolov8N Dual

| Metric | Mean ± SD | 95% CI | N |
|--------|-----------|--------|---|
| map50 | 0.921 ± 0.000 | [0.919, 0.923] | 2 |
| map50_95 | 0.897 ± 0.000 | [0.893, 0.901] | 2 |
| precision | 0.840 ± 0.034 | [0.534, 1.147] | 2 |
| recall | 0.855 ± 0.018 | [0.694, 1.016] | 2 |
| f1_score | 0.847 ± 0.009 | [0.770, 0.924] | 2 |

## Statistical Significance Tests

### MAP50

| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |
|------------|-----------|---------|-------------|-----------|-------------|
| yolov11n dual vs yolov8n dual | 0.012 | 0.0461 | 0.0461 | 4.497 | **Yes** |

### MAP50_95

| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |
|------------|-----------|---------|-------------|-----------|-------------|
| yolov11n dual vs yolov8n dual | 0.015 | 0.0216 | 0.0216 | 6.690 | **Yes** |

### PRECISION

| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |
|------------|-----------|---------|-------------|-----------|-------------|
| yolov11n dual vs yolov8n dual | -0.003 | 0.9128 | 0.9128 | -0.124 | No |

### RECALL

| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |
|------------|-----------|---------|-------------|-----------|-------------|
| yolov11n dual vs yolov8n dual | 0.002 | 0.8783 | 0.8783 | 0.173 | No |

### F1_SCORE

| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |
|------------|-----------|---------|-------------|-----------|-------------|
| yolov11n dual vs yolov8n dual | -0.000 | 0.9966 | 0.9966 | -0.005 | No |
