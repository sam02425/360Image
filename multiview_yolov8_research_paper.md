# Multi-View YOLOv8 Object Detection: A Comprehensive Academic Study

## Abstract
This paper presents an in-depth analysis of multi-view YOLOv8 object detection using four dataset configurations (Dual, Quad, Octal, Full) on an NVIDIA RTX 3070. We evaluate detection accuracy, computational efficiency, and real-world deployment scenarios, providing actionable insights for both industry and academia.

## 1. Introduction
Object detection is a cornerstone of computer vision, with applications ranging from retail analytics to autonomous vehicles. Multi-view detection, where objects are observed from multiple angles, promises improved accuracy but increases computational demands. This study systematically benchmarks YOLOv8 performance across four multi-view configurations.

## 2. Methodology
### 2.1 Hardware & Environment
- **GPU:** NVIDIA RTX 3070 (8GB VRAM)
- **CPU:** 8-core
- **RAM:** 32GB
- **OS:** Ubuntu 20.04
- **Frameworks:** PyTorch, Ultralytics YOLOv8

### 2.2 Dataset Configurations
- **Dual:** 2 views per object (minimal, fast)
- **Quad:** 4 views (balanced)
- **Octal:** 8 views (high accuracy)
- **Full:** 24 views (maximum coverage)

Each dataset contains 414 classes, with identical train/val/test splits and label structure.

### 2.3 Training Protocol
- **Epochs:** 100
- **Batch Size:** 16
- **Mixed Precision:** Enabled
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (cosine schedule)
- **Augmentation:** Standard YOLOv8 pipeline

### 2.4 Evaluation Metrics
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision, Recall, F1**
- **Training Time**
- **Resource Utilization:** GPU/CPU/RAM

# Methodology

## Dataset Preparation
- Four dataset variants were used: dual, quad, octal, and full, each representing different camera/view configurations and data volumes.
- All datasets were split into train, val, and test sets. For evaluation, a unified test set was created to ensure direct comparability across models.
- Data integrity was verified, and missing labels or images were fixed to avoid mAP calculation errors.

## Model Training
- YOLOv8 and YOLOv11 models were trained separately on each dataset variant using the Ultralytics framework.
- Training parameters included: batch size 16, image size 640, 100 epochs, early stopping (patience 20), mixed precision (AMP), and caching for speed.
- Training was performed on an NVIDIA RTX 3070 GPU with CUDA 12.8 and PyTorch 2.8.0+cu128.
- Model weights were saved in `data/models/` for each variant.

## Evaluation
- All trained models were evaluated on the unified test set for fair comparison.
- Evaluation metrics included mAP@0.5, mAP@0.5:0.95, precision, recall, F1 score, and per-class AP.
- Confusion matrices and precision-recall curves were generated for each model/dataset combination.
- Results were saved in `data/results/` as JSON files and images.

## Analytics Workflow
- Custom Python scripts automated training, evaluation, and analytics extraction for both YOLOv8 and YOLOv11.
- Scripts ensured correct data.yaml paths and handled missing directories/files.
- All results were aggregated for direct comparison and reporting.
- Markdown-based reporting was used for research paper generation, with instructions for including all relevant images and metrics.

## Reproducibility
- All code, configuration files, and results are stored in the workspace for full reproducibility.
- Experiment logs and resource usage statistics are included in the results for transparency.

## 3. Results
### 3.1 Quantitative Results
| Config | Views | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1   | Time    |
|--------|-------|---------|-------------|-----------|--------|------|---------|
| Dual   | 2     | 0.476   | 0.456       | 0.657     | 0.441  | 0.528| 1:15:19 |
| Quad   | 4     | 0.957   | 0.924       | 0.813     | 0.889  | 0.849| 1:15:21 |
| Octal  | 8     | 0.973   | 0.948       | 0.941     | 0.957  | 0.949| 1:15:23 |
| Full   | 24    | 0.808   | 0.785       | 0.877     | 0.753  | 0.810| 1:15:29 |

### 3.2 Resource Utilization
- **GPU Memory:** 3-4GB (peak), higher for octal/full
- **CPU Usage:** 7-10% during training
- **RAM Usage:** 8-10GB

### 3.3 Analysis
- **Accuracy:** Octal achieved the highest mAP and F1. Quad is optimal for most real-world deployments.
- **Computation:** Training time is similar across configs, but full view requires more memory and storage.
- **Recall:** Quad and octal offer the best recall, critical for safety and quality control applications.

## 4. Discussion
### 4.1 Trade-offs
- **Dual:** Fast, low-resource, but lower accuracy. Best for edge/real-time.
- **Quad:** Best balance for industry (retail, security).
- **Octal:** For high-accuracy needs (medical, QC).
- **Full:** For research, benchmarking, or maximum coverage.

### 4.2 Real-World Deployment
- **Dual:** Robotics, mobile, IoT.
- **Quad:** Retail analytics, smart cameras.
- **Octal:** Automated inspection, medical imaging.
- **Full:** Academic research, safety-critical systems.

## 5. Conclusion
Multi-view YOLOv8 detection scales well on RTX 3070. Quad and octal configurations are optimal for most industry use. Full view is best for research or when every angle matters. This study provides a reproducible benchmark and practical guidance for deploying multi-view detection in real-world scenarios.

## 6. References
- Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection.
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.

# YOLOv8 vs YOLOv11 Unified Test Set Analysis

## Overview
This section presents a comprehensive comparison between YOLOv8 and YOLOv11 models, each trained on four dataset variants (dual, quad, octal, full) and evaluated on a unified test set. All metrics, results, and analysis are provided for journal-grade reporting.

## Metrics Compared
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score

## Results Table
| Model   | mAP@0.5 (v8) | mAP@0.5 (v11) | mAP@0.5:0.95 (v8) | mAP@0.5:0.95 (v11) | Precision (v8) | Precision (v11) | Recall (v8) | Recall (v11) | F1 (v8) | F1 (v11) |
|---------|--------------|--------------|-------------------|--------------------|---------------|----------------|-------------|-------------|------------|----------|
| Dual    | 0.476        | 0.708        | 0.456             | 0.674              | 0.657         | 0.704          | 0.441       | 0.649       | 0.528      | 0.676    |
| Quad    | 0.957        | 0.815        | 0.924             | 0.780              | 0.813         | 0.793          | 0.889       | 0.729       | 0.849      | 0.760    |
| Octal   | 0.973        | 0.831        | 0.948             | 0.804              | 0.941         | 0.813          | 0.957       | 0.794       | 0.949      | 0.803    |
| Full    | 0.808        | 0.844        | 0.785             | 0.816              | 0.877         | 0.801          | 0.753       | 0.815       | 0.810      | 0.808    |

## Analysis
- **YOLOv8** achieves highest mAP on quad and octal datasets, with octal showing best overall performance.
- **YOLOv11** shows improved recall and F1 on dual and full datasets, indicating better generalization in those cases.
- Precision is consistently high for both models, but YOLOv8 outperforms YOLOv11 on quad and octal, while YOLOv11 is better on dual and full.
- The unified test set provides a fair benchmark for all models, ensuring direct comparability.

## Visualizations
- For each model/dataset, include:
  - **Precision-Recall Curve:** `data/results/[model]_model/BoxP_curve.png`
  - **Confusion Matrix:** `data/results/[model]_model/confusion_matrix_normalized.png`
- Example:
  - ![Dual YOLOv8 PR Curve](data/results/dual_model/BoxP_curve.png)
  - ![Dual YOLOv8 Confusion Matrix](data/results/dual_model/confusion_matrix_normalized.png)
  - ![Dual YOLOv11 PR Curve](data/results/dual_model/BoxP_curve.png) <!-- Replace with YOLOv11 path if available -->

## Recommendations
- Use octal or quad dataset for highest accuracy with YOLOv8.
- For applications requiring higher recall, YOLOv11 on full or dual dataset is preferable.
- Both models are suitable for real-world deployment; selection depends on specific metric priorities.

## Further Work
- Include additional metrics (per-class AP, resource utilization) as needed.
- Add more visualizations from `data/results/[model]_model/` for deeper insight.
- Automate report generation for future experiments.

---
*For supplementary code, datasets, and results, see the project repository.*
