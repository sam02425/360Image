# Rigorous Multi-View Object Detection: Statistical Analysis Report

## Experimental Design

- **Architectures Tested**: yolov8n, yolov11n
- **View Configurations**: dual, quad, octal, full
- **Independent Runs**: 5
- **Cross-Validation**: No
- **Total Experiments**: 40
- **Random Seeds**: [np.int64(7271), np.int64(861), np.int64(5391), np.int64(5192), np.int64(5735)]
- **Statistical Significance Level**: 0.05
- **Multiple Comparison Correction**: bonferroni

## Descriptive Statistics

### Yolov11N Dual

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.458 ± 0.020 | [0.433, 0.483] | 0.449 | 0.447 | 0.494 | 5 |
| map50_95 | 0.433 ± 0.018 | [0.410, 0.455] | 0.424 | 0.423 | 0.465 | 5 |
| precision | 0.695 ± 0.060 | [0.620, 0.769] | 0.712 | 0.595 | 0.749 | 5 |
| recall | 0.418 ± 0.025 | [0.388, 0.449] | 0.411 | 0.396 | 0.446 | 5 |
| f1_score | 0.520 ± 0.016 | [0.500, 0.540] | 0.514 | 0.508 | 0.548 | 5 |
| training_time_seconds | 4434.479 ± 17.085 | [4413.265, 4455.693] | 4441.792 | 4404.851 | 4447.170 | 5 |

### Yolov11N Full

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.776 ± 0.010 | [0.764, 0.789] | 0.775 | 0.764 | 0.792 | 5 |
| map50_95 | 0.746 ± 0.011 | [0.732, 0.760] | 0.745 | 0.732 | 0.763 | 5 |
| precision | 0.806 ± 0.033 | [0.765, 0.847] | 0.798 | 0.771 | 0.852 | 5 |
| recall | 0.717 ± 0.014 | [0.699, 0.735] | 0.709 | 0.705 | 0.740 | 5 |
| f1_score | 0.759 ± 0.014 | [0.741, 0.776] | 0.762 | 0.739 | 0.774 | 5 |
| training_time_seconds | 4425.329 ± 18.035 | [4402.936, 4447.722] | 4417.015 | 4409.921 | 4452.012 | 5 |

### Yolov11N Octal

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.924 ± 0.006 | [0.917, 0.932] | 0.923 | 0.918 | 0.934 | 5 |
| map50_95 | 0.887 ± 0.008 | [0.878, 0.897] | 0.886 | 0.879 | 0.899 | 5 |
| precision | 0.840 ± 0.033 | [0.799, 0.881] | 0.820 | 0.810 | 0.877 | 5 |
| recall | 0.874 ± 0.013 | [0.858, 0.891] | 0.876 | 0.859 | 0.892 | 5 |
| f1_score | 0.857 ± 0.018 | [0.835, 0.878] | 0.855 | 0.838 | 0.878 | 5 |
| training_time_seconds | 4416.200 ± 15.236 | [4397.282, 4435.117] | 4416.232 | 4400.878 | 4439.832 | 5 |

### Yolov11N Quad

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.900 ± 0.012 | [0.885, 0.914] | 0.900 | 0.883 | 0.916 | 5 |
| map50_95 | 0.861 ± 0.013 | [0.845, 0.877] | 0.861 | 0.843 | 0.879 | 5 |
| precision | 0.798 ± 0.030 | [0.761, 0.835] | 0.809 | 0.746 | 0.821 | 5 |
| recall | 0.843 ± 0.012 | [0.828, 0.858] | 0.844 | 0.827 | 0.857 | 5 |
| f1_score | 0.820 ± 0.021 | [0.794, 0.845] | 0.830 | 0.784 | 0.832 | 5 |
| training_time_seconds | 4422.353 ± 20.843 | [4396.474, 4448.233] | 4427.764 | 4396.147 | 4447.480 | 5 |

### Yolov8N Dual

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.448 ± 0.015 | [0.430, 0.466] | 0.445 | 0.434 | 0.465 | 5 |
| map50_95 | 0.422 ± 0.014 | [0.405, 0.438] | 0.415 | 0.410 | 0.436 | 5 |
| precision | 0.707 ± 0.034 | [0.666, 0.749] | 0.710 | 0.659 | 0.738 | 5 |
| recall | 0.395 ± 0.016 | [0.376, 0.415] | 0.391 | 0.380 | 0.414 | 5 |
| f1_score | 0.507 ± 0.013 | [0.491, 0.522] | 0.503 | 0.491 | 0.523 | 5 |
| training_time_seconds | 4263.742 ± 11.570 | [4249.376, 4278.107] | 4265.654 | 4251.173 | 4278.136 | 5 |

### Yolov8N Full

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.779 ± 0.004 | [0.774, 0.784] | 0.779 | 0.773 | 0.784 | 5 |
| map50_95 | 0.741 ± 0.004 | [0.735, 0.746] | 0.742 | 0.734 | 0.745 | 5 |
| precision | 0.798 ± 0.028 | [0.763, 0.833] | 0.795 | 0.771 | 0.844 | 5 |
| recall | 0.720 ± 0.009 | [0.710, 0.731] | 0.720 | 0.708 | 0.731 | 5 |
| f1_score | 0.757 ± 0.010 | [0.745, 0.769] | 0.755 | 0.747 | 0.770 | 5 |
| training_time_seconds | 4396.616 ± 6.510 | [4388.533, 4404.699] | 4399.245 | 4385.182 | 4400.598 | 5 |

### Yolov8N Octal

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.912 ± 0.004 | [0.907, 0.917] | 0.914 | 0.907 | 0.917 | 5 |
| map50_95 | 0.870 ± 0.004 | [0.866, 0.875] | 0.872 | 0.866 | 0.874 | 5 |
| precision | 0.814 ± 0.008 | [0.805, 0.824] | 0.812 | 0.807 | 0.828 | 5 |
| recall | 0.868 ± 0.016 | [0.849, 0.888] | 0.878 | 0.848 | 0.881 | 5 |
| f1_score | 0.840 ± 0.006 | [0.833, 0.848] | 0.842 | 0.830 | 0.845 | 5 |
| training_time_seconds | 4372.493 ± 8.506 | [4361.932, 4383.054] | 4371.510 | 4363.681 | 4384.659 | 5 |

### Yolov8N Quad

| Metric | Mean ± SD | 95% CI | Median | Min | Max | N |
|--------|-----------|--------|--------|-----|-----|---|
| map50 | 0.883 ± 0.010 | [0.871, 0.895] | 0.886 | 0.872 | 0.892 | 5 |
| map50_95 | 0.839 ± 0.008 | [0.829, 0.849] | 0.843 | 0.827 | 0.847 | 5 |
| precision | 0.787 ± 0.009 | [0.775, 0.798] | 0.785 | 0.775 | 0.801 | 5 |
| recall | 0.814 ± 0.017 | [0.793, 0.835] | 0.817 | 0.789 | 0.829 | 5 |
| f1_score | 0.800 ± 0.008 | [0.789, 0.810] | 0.801 | 0.787 | 0.809 | 5 |
| training_time_seconds | 4349.569 ± 42.844 | [4296.371, 4402.767] | 4363.446 | 4274.460 | 4379.988 | 5 |

## Statistical Significance Tests

Pairwise comparisons with bonferroni correction for multiple testing.

### MAP50

| Comparison | Mean Diff | Test | Statistic | p-value | p-corrected | Cohen's d | Effect | Significant |
|------------|-----------|------|-----------|---------|-------------|-----------|--------|-------------|
| yolov11n dual vs yolov11n full | -0.318 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -19.907 | large | No |
| yolov11n dual vs yolov11n octal | -0.466 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -31.210 | large | No |
| yolov11n dual vs yolov11n quad | -0.442 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -26.802 | large | No |
| yolov11n dual vs yolov8n dual | 0.010 | mannwhitneyu | 17.000 | 0.4206 | 1.0000 | 0.589 | medium | No |
| yolov11n dual vs yolov8n full | -0.321 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -21.998 | large | No |
| yolov11n dual vs yolov8n octal | -0.454 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -31.165 | large | No |
| yolov11n dual vs yolov8n quad | -0.425 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -26.895 | large | No |
| yolov11n full vs yolov11n octal | -0.148 | ttest | -27.946 | 0.0000 | 0.0000 | -17.675 | large | **Yes** |
| yolov11n full vs yolov11n quad | -0.123 | ttest | -17.932 | 0.0000 | 0.0000 | -11.341 | large | **Yes** |
| yolov11n full vs yolov8n dual | 0.328 | ttest | 41.407 | 0.0000 | 0.0000 | 26.188 | large | **Yes** |
| yolov11n full vs yolov8n full | -0.003 | ttest | -0.535 | 0.6075 | 1.0000 | -0.338 | small | No |
| yolov11n full vs yolov8n octal | -0.136 | ttest | -27.902 | 0.0000 | 0.0000 | -17.647 | large | **Yes** |
| yolov11n full vs yolov8n quad | -0.107 | ttest | -17.168 | 0.0000 | 0.0000 | -10.858 | large | **Yes** |
| yolov11n octal vs yolov11n quad | 0.024 | ttest | 4.131 | 0.0033 | 0.0922 | 2.613 | large | No |
| yolov11n octal vs yolov8n dual | 0.476 | ttest | 67.413 | 0.0000 | 0.0000 | 42.636 | large | **Yes** |
| yolov11n octal vs yolov8n full | 0.145 | ttest | 44.125 | 0.0000 | 0.0000 | 27.907 | large | **Yes** |
| yolov11n octal vs yolov8n octal | 0.012 | ttest | 3.588 | 0.0071 | 0.1989 | 2.269 | large | No |
| yolov11n octal vs yolov8n quad | 0.041 | ttest | 8.101 | 0.0000 | 0.0011 | 5.124 | large | **Yes** |
| yolov11n quad vs yolov8n dual | 0.452 | ttest | 54.263 | 0.0000 | 0.0000 | 34.319 | large | **Yes** |
| yolov11n quad vs yolov8n full | 0.121 | ttest | 21.959 | 0.0000 | 0.0000 | 13.888 | large | **Yes** |
| yolov11n quad vs yolov8n octal | -0.012 | ttest | -2.272 | 0.0527 | 1.0000 | -1.437 | large | No |
| yolov11n quad vs yolov8n quad | 0.017 | ttest | 2.497 | 0.0371 | 1.0000 | 1.579 | large | No |
| yolov8n dual vs yolov8n full | -0.331 | ttest | -48.914 | 0.0000 | 0.0000 | -30.936 | large | **Yes** |
| yolov8n dual vs yolov8n octal | -0.464 | ttest | -68.698 | 0.0000 | 0.0000 | -43.448 | large | **Yes** |
| yolov8n dual vs yolov8n quad | -0.435 | ttest | -55.907 | 0.0000 | 0.0000 | -35.359 | large | **Yes** |
| yolov8n full vs yolov8n octal | -0.133 | ttest | -51.816 | 0.0000 | 0.0000 | -32.772 | large | **Yes** |
| yolov8n full vs yolov8n quad | -0.104 | ttest | -22.444 | 0.0000 | 0.0000 | -14.195 | large | **Yes** |
| yolov8n octal vs yolov8n quad | 0.029 | ttest | 6.322 | 0.0002 | 0.0064 | 3.998 | large | **Yes** |

### MAP50_95

| Comparison | Mean Diff | Test | Statistic | p-value | p-corrected | Cohen's d | Effect | Significant |
|------------|-----------|------|-----------|---------|-------------|-----------|--------|-------------|
| yolov11n dual vs yolov11n full | -0.313 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -20.838 | large | No |
| yolov11n dual vs yolov11n octal | -0.455 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -32.444 | large | No |
| yolov11n dual vs yolov11n quad | -0.428 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -27.217 | large | No |
| yolov11n dual vs yolov8n dual | 0.011 | mannwhitneyu | 17.000 | 0.4206 | 1.0000 | 0.696 | medium | No |
| yolov11n dual vs yolov8n full | -0.308 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -23.270 | large | No |
| yolov11n dual vs yolov8n octal | -0.438 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -33.300 | large | No |
| yolov11n dual vs yolov8n quad | -0.407 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -28.931 | large | No |
| yolov11n full vs yolov11n octal | -0.141 | ttest | -23.438 | 0.0000 | 0.0000 | -14.824 | large | **Yes** |
| yolov11n full vs yolov11n quad | -0.115 | ttest | -15.259 | 0.0000 | 0.0000 | -9.651 | large | **Yes** |
| yolov11n full vs yolov8n dual | 0.325 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 26.291 | large | No |
| yolov11n full vs yolov8n full | 0.005 | ttest | 1.031 | 0.3328 | 1.0000 | 0.652 | medium | No |
| yolov11n full vs yolov8n octal | -0.124 | ttest | -23.982 | 0.0000 | 0.0000 | -15.168 | large | **Yes** |
| yolov11n full vs yolov8n quad | -0.093 | ttest | -15.375 | 0.0000 | 0.0000 | -9.724 | large | **Yes** |
| yolov11n octal vs yolov11n quad | 0.026 | ttest | 3.890 | 0.0046 | 0.1290 | 2.460 | large | No |
| yolov11n octal vs yolov8n dual | 0.466 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 42.076 | large | No |
| yolov11n octal vs yolov8n full | 0.147 | ttest | 36.805 | 0.0000 | 0.0000 | 23.277 | large | **Yes** |
| yolov11n octal vs yolov8n octal | 0.017 | ttest | 4.363 | 0.0024 | 0.0673 | 2.760 | large | No |
| yolov11n octal vs yolov8n quad | 0.048 | ttest | 9.618 | 0.0000 | 0.0003 | 6.083 | large | **Yes** |
| yolov11n quad vs yolov8n dual | 0.440 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 33.324 | large | No |
| yolov11n quad vs yolov8n full | 0.121 | ttest | 19.956 | 0.0000 | 0.0000 | 12.621 | large | **Yes** |
| yolov11n quad vs yolov8n octal | -0.009 | ttest | -1.547 | 0.1604 | 1.0000 | -0.979 | large | No |
| yolov11n quad vs yolov8n quad | 0.022 | ttest | 3.241 | 0.0119 | 0.3320 | 2.050 | large | No |
| yolov8n dual vs yolov8n full | -0.319 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -31.699 | large | No |
| yolov8n dual vs yolov8n octal | -0.449 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -45.115 | large | No |
| yolov8n dual vs yolov8n quad | -0.418 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -37.555 | large | No |
| yolov8n full vs yolov8n octal | -0.130 | ttest | -51.054 | 0.0000 | 0.0000 | -32.290 | large | **Yes** |
| yolov8n full vs yolov8n quad | -0.099 | ttest | -24.393 | 0.0000 | 0.0000 | -15.427 | large | **Yes** |
| yolov8n octal vs yolov8n quad | 0.031 | ttest | 7.912 | 0.0000 | 0.0013 | 5.004 | large | **Yes** |

### PRECISION

| Comparison | Mean Diff | Test | Statistic | p-value | p-corrected | Cohen's d | Effect | Significant |
|------------|-----------|------|-----------|---------|-------------|-----------|--------|-------------|
| yolov11n dual vs yolov11n full | -0.111 | ttest | -3.616 | 0.0068 | 0.1910 | -2.287 | large | No |
| yolov11n dual vs yolov11n octal | -0.146 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -2.992 | large | No |
| yolov11n dual vs yolov11n quad | -0.104 | mannwhitneyu | 1.000 | 0.0159 | 0.4444 | -2.176 | large | No |
| yolov11n dual vs yolov8n dual | -0.013 | ttest | -0.412 | 0.6911 | 1.0000 | -0.261 | small | No |
| yolov11n dual vs yolov8n full | -0.103 | ttest | -3.471 | 0.0084 | 0.2360 | -2.195 | large | No |
| yolov11n dual vs yolov8n octal | -0.120 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -2.784 | large | No |
| yolov11n dual vs yolov8n quad | -0.092 | ttest | -3.369 | 0.0098 | 0.2745 | -2.131 | large | No |
| yolov11n full vs yolov11n octal | -0.034 | mannwhitneyu | 6.000 | 0.2222 | 1.0000 | -1.029 | large | No |
| yolov11n full vs yolov11n quad | 0.008 | mannwhitneyu | 13.000 | 1.0000 | 1.0000 | 0.249 | small | No |
| yolov11n full vs yolov8n dual | 0.099 | ttest | 4.666 | 0.0016 | 0.0451 | 2.951 | large | **Yes** |
| yolov11n full vs yolov8n full | 0.008 | ttest | 0.414 | 0.6900 | 1.0000 | 0.262 | small | No |
| yolov11n full vs yolov8n octal | -0.008 | mannwhitneyu | 9.000 | 0.5476 | 1.0000 | -0.344 | small | No |
| yolov11n full vs yolov8n quad | 0.019 | ttest | 1.256 | 0.2445 | 1.0000 | 0.795 | medium | No |
| yolov11n octal vs yolov11n quad | 0.042 | mannwhitneyu | 21.000 | 0.0952 | 1.0000 | 1.334 | large | No |
| yolov11n octal vs yolov8n dual | 0.133 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 3.980 | large | No |
| yolov11n octal vs yolov8n full | 0.042 | mannwhitneyu | 22.000 | 0.0556 | 1.0000 | 1.373 | large | No |
| yolov11n octal vs yolov8n octal | 0.026 | mannwhitneyu | 19.000 | 0.2222 | 1.0000 | 1.074 | large | No |
| yolov11n octal vs yolov8n quad | 0.054 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 2.199 | large | No |
| yolov11n quad vs yolov8n dual | 0.091 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 2.858 | large | No |
| yolov11n quad vs yolov8n full | 0.000 | mannwhitneyu | 16.000 | 0.5476 | 1.0000 | 0.007 | negligible | No |
| yolov11n quad vs yolov8n octal | -0.016 | mannwhitneyu | 6.000 | 0.2222 | 1.0000 | -0.742 | medium | No |
| yolov11n quad vs yolov8n quad | 0.012 | mannwhitneyu | 20.000 | 0.1508 | 1.0000 | 0.523 | medium | No |
| yolov8n dual vs yolov8n full | -0.091 | ttest | -4.621 | 0.0017 | 0.0478 | -2.923 | large | **Yes** |
| yolov8n dual vs yolov8n octal | -0.107 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -4.388 | large | No |
| yolov8n dual vs yolov8n quad | -0.079 | ttest | -5.080 | 0.0010 | 0.0267 | -3.213 | large | **Yes** |
| yolov8n full vs yolov8n octal | -0.016 | mannwhitneyu | 5.000 | 0.1508 | 1.0000 | -0.792 | medium | No |
| yolov8n full vs yolov8n quad | 0.011 | ttest | 0.856 | 0.4171 | 1.0000 | 0.541 | medium | No |
| yolov8n octal vs yolov8n quad | 0.028 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 3.193 | large | No |

### RECALL

| Comparison | Mean Diff | Test | Statistic | p-value | p-corrected | Cohen's d | Effect | Significant |
|------------|-----------|------|-----------|---------|-------------|-----------|--------|-------------|
| yolov11n dual vs yolov11n full | -0.299 | ttest | -23.373 | 0.0000 | 0.0000 | -14.782 | large | **Yes** |
| yolov11n dual vs yolov11n octal | -0.456 | ttest | -36.637 | 0.0000 | 0.0000 | -23.171 | large | **Yes** |
| yolov11n dual vs yolov11n quad | -0.425 | ttest | -34.699 | 0.0000 | 0.0000 | -21.946 | large | **Yes** |
| yolov11n dual vs yolov8n dual | 0.023 | ttest | 1.738 | 0.1204 | 1.0000 | 1.099 | large | No |
| yolov11n dual vs yolov8n full | -0.302 | ttest | -25.905 | 0.0000 | 0.0000 | -16.384 | large | **Yes** |
| yolov11n dual vs yolov8n octal | -0.450 | ttest | -34.522 | 0.0000 | 0.0000 | -21.833 | large | **Yes** |
| yolov11n dual vs yolov8n quad | -0.395 | ttest | -29.663 | 0.0000 | 0.0000 | -18.761 | large | **Yes** |
| yolov11n full vs yolov11n octal | -0.157 | ttest | -18.101 | 0.0000 | 0.0000 | -11.448 | large | **Yes** |
| yolov11n full vs yolov11n quad | -0.126 | ttest | -15.005 | 0.0000 | 0.0000 | -9.490 | large | **Yes** |
| yolov11n full vs yolov8n dual | 0.322 | ttest | 33.370 | 0.0000 | 0.0000 | 21.105 | large | **Yes** |
| yolov11n full vs yolov8n full | -0.003 | ttest | -0.426 | 0.6811 | 1.0000 | -0.270 | small | No |
| yolov11n full vs yolov8n octal | -0.151 | ttest | -15.899 | 0.0000 | 0.0000 | -10.056 | large | **Yes** |
| yolov11n full vs yolov8n quad | -0.097 | ttest | -9.744 | 0.0000 | 0.0003 | -6.163 | large | **Yes** |
| yolov11n octal vs yolov11n quad | 0.032 | ttest | 4.012 | 0.0039 | 0.1088 | 2.537 | large | No |
| yolov11n octal vs yolov8n dual | 0.479 | ttest | 52.129 | 0.0000 | 0.0000 | 32.969 | large | **Yes** |
| yolov11n octal vs yolov8n full | 0.154 | ttest | 22.252 | 0.0000 | 0.0000 | 14.073 | large | **Yes** |
| yolov11n octal vs yolov8n octal | 0.006 | ttest | 0.665 | 0.5248 | 1.0000 | 0.420 | small | No |
| yolov11n octal vs yolov8n quad | 0.061 | ttest | 6.410 | 0.0002 | 0.0058 | 4.054 | large | **Yes** |
| yolov11n quad vs yolov8n dual | 0.447 | ttest | 50.289 | 0.0000 | 0.0000 | 31.805 | large | **Yes** |
| yolov11n quad vs yolov8n full | 0.123 | ttest | 18.758 | 0.0000 | 0.0000 | 11.863 | large | **Yes** |
| yolov11n quad vs yolov8n octal | -0.025 | ttest | -2.911 | 0.0196 | 0.5478 | -1.841 | large | No |
| yolov11n quad vs yolov8n quad | 0.029 | ttest | 3.177 | 0.0131 | 0.3657 | 2.009 | large | No |
| yolov8n dual vs yolov8n full | -0.325 | ttest | -40.174 | 0.0000 | 0.0000 | -25.408 | large | **Yes** |
| yolov8n dual vs yolov8n octal | -0.473 | ttest | -47.432 | 0.0000 | 0.0000 | -29.999 | large | **Yes** |
| yolov8n dual vs yolov8n quad | -0.418 | ttest | -40.410 | 0.0000 | 0.0000 | -25.558 | large | **Yes** |
| yolov8n full vs yolov8n octal | -0.148 | ttest | -18.660 | 0.0000 | 0.0000 | -11.802 | large | **Yes** |
| yolov8n full vs yolov8n quad | -0.093 | ttest | -11.106 | 0.0000 | 0.0001 | -7.024 | large | **Yes** |
| yolov8n octal vs yolov8n quad | 0.055 | ttest | 5.344 | 0.0007 | 0.0193 | 3.380 | large | **Yes** |

### F1_SCORE

| Comparison | Mean Diff | Test | Statistic | p-value | p-corrected | Cohen's d | Effect | Significant |
|------------|-----------|------|-----------|---------|-------------|-----------|--------|-------------|
| yolov11n dual vs yolov11n full | -0.238 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -15.875 | large | No |
| yolov11n dual vs yolov11n octal | -0.336 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -20.012 | large | No |
| yolov11n dual vs yolov11n quad | -0.299 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -16.231 | large | No |
| yolov11n dual vs yolov8n dual | 0.014 | mannwhitneyu | 18.000 | 0.3095 | 1.0000 | 0.942 | large | No |
| yolov11n dual vs yolov8n full | -0.237 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -17.832 | large | No |
| yolov11n dual vs yolov8n octal | -0.320 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -26.221 | large | No |
| yolov11n dual vs yolov8n quad | -0.279 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -21.819 | large | No |
| yolov11n full vs yolov11n octal | -0.098 | ttest | -9.809 | 0.0000 | 0.0003 | -6.204 | large | **Yes** |
| yolov11n full vs yolov11n quad | -0.061 | mannwhitneyu | 0.000 | 0.0079 | 0.2222 | -3.487 | large | No |
| yolov11n full vs yolov8n dual | 0.252 | ttest | 30.124 | 0.0000 | 0.0000 | 19.052 | large | **Yes** |
| yolov11n full vs yolov8n full | 0.002 | ttest | 0.226 | 0.8266 | 1.0000 | 0.143 | negligible | No |
| yolov11n full vs yolov8n octal | -0.082 | ttest | -11.987 | 0.0000 | 0.0001 | -7.581 | large | **Yes** |
| yolov11n full vs yolov8n quad | -0.041 | ttest | -5.681 | 0.0005 | 0.0130 | -3.593 | large | **Yes** |
| yolov11n octal vs yolov11n quad | 0.037 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 1.931 | large | No |
| yolov11n octal vs yolov8n dual | 0.350 | ttest | 36.329 | 0.0000 | 0.0000 | 22.977 | large | **Yes** |
| yolov11n octal vs yolov8n full | 0.100 | ttest | 11.140 | 0.0000 | 0.0001 | 7.045 | large | **Yes** |
| yolov11n octal vs yolov8n octal | 0.016 | ttest | 1.952 | 0.0867 | 1.0000 | 1.235 | large | No |
| yolov11n octal vs yolov8n quad | 0.057 | ttest | 6.546 | 0.0002 | 0.0050 | 4.140 | large | **Yes** |
| yolov11n quad vs yolov8n dual | 0.313 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 18.387 | large | No |
| yolov11n quad vs yolov8n full | 0.063 | mannwhitneyu | 25.000 | 0.0079 | 0.2222 | 3.912 | large | No |
| yolov11n quad vs yolov8n octal | -0.021 | mannwhitneyu | 3.000 | 0.0556 | 1.0000 | -1.355 | large | No |
| yolov11n quad vs yolov8n quad | 0.020 | mannwhitneyu | 20.000 | 0.1508 | 1.0000 | 1.272 | large | No |
| yolov8n dual vs yolov8n full | -0.250 | ttest | -35.313 | 0.0000 | 0.0000 | -22.334 | large | **Yes** |
| yolov8n dual vs yolov8n octal | -0.334 | ttest | -53.170 | 0.0000 | 0.0000 | -33.628 | large | **Yes** |
| yolov8n dual vs yolov8n quad | -0.293 | ttest | -43.491 | 0.0000 | 0.0000 | -27.506 | large | **Yes** |
| yolov8n full vs yolov8n octal | -0.084 | ttest | -16.119 | 0.0000 | 0.0000 | -10.194 | large | **Yes** |
| yolov8n full vs yolov8n quad | -0.043 | ttest | -7.485 | 0.0001 | 0.0020 | -4.734 | large | **Yes** |
| yolov8n octal vs yolov8n quad | 0.041 | ttest | 8.641 | 0.0000 | 0.0007 | 5.465 | large | **Yes** |

## Data Quality Assessment

### Training Quality Indicators

- **Data Leakage Detection**: 0/40 runs (0.0%)
- **Training Convergence**: 0/40 runs (0.0%)
- **Overfitting Detection**: 0/40 runs (0.0%)

### Quality Flags

- ⚠️ LOW_CONVERGENCE_RATE

## Statistical Power Analysis

| Configuration | Sample Size | Statistical Power | Adequate (≥0.8) |
|---------------|-------------|-------------------|------------------|
| yolov11n dual | 5 | 0.064 | ❌ No |
| yolov11n full | 5 | 0.064 | ❌ No |
| yolov11n octal | 5 | 0.064 | ❌ No |
| yolov11n quad | 5 | 0.064 | ❌ No |
| yolov8n dual | 5 | 0.064 | ❌ No |
| yolov8n full | 5 | 0.064 | ❌ No |
| yolov8n octal | 5 | 0.064 | ❌ No |
| yolov8n quad | 5 | 0.064 | ❌ No |

