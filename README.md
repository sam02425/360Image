# Multi-View YOLOv8 Research: Angle-Invariant Retail Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)

## 📄 Research Paper

**Title:** Multi-View Training Requirements for Angle-Invariant Retail Object Detection: A Comprehensive Empirical Investigation

**Authors:** Saumil Patel¹, Rohith Naini²
- ¹Department of Industrial and Systems Engineering, Lamar University
- ²Curry Creations, Beaumont, Texas

**Status:** Ready for Journal Submission (October 2025)

### Key Documents
- 📝 **[Final Submission Paper](./FINAL_SUBMISSION_PAPER.md)** - Complete manuscript with all experimental results
- 📊 **[Experiment Results Summary](./EXPERIMENT_RESULTS_SUMMARY.md)** - Comprehensive analysis and findings
- 📑 **[Research Paper (Markdown)](./multiview_yolov8_research_paper.md)** - Original research documentation
- 📄 **[Research Paper (LaTeX PDF)](./research_paper_latex.pdf)** - IEEE conference format paper

---

## 🎯 Research Overview

This study provides the most comprehensive investigation to date of training view requirements for angle-invariant retail object detection. We systematically evaluate four multi-view training strategies (dual, quad, octal, full 360°) across 414 product classes using YOLOv8.

### Research Questions

1. **RQ1:** What is the relationship between training view density and detection performance?
2. **RQ2:** Which hyperparameters most significantly impact multi-view detection?
3. **RQ3:** How do models perform under real-world retail environmental conditions?
4. **RQ4:** Do certain product categories benefit differentially from multi-view training?

---

## 🔬 Experimental Design

### Four Comprehensive Experiments

#### 1. Baseline Performance Evaluation
Four training configurations tested:
- **Dual-view:** 2 viewpoints (0°, 180°) - minimal configuration
- **Quad-view:** 4 viewpoints (0°, 90°, 180°, 270°) - optimal balance ⭐
- **Octal-view:** 8 viewpoints (45° spacing) - peak accuracy
- **Full coverage:** 24 viewpoints (15° spacing) - comprehensive sampling

#### 2. Hyperparameter Ablation Study
11 training configurations tested:
- Baseline (SGD, batch=16, lr=0.01)
- Optimizer variations (Adam, AdamW)
- Batch size variations (8, 32)
- Learning rate variations (0.001, 0.1)
- Image size variations (320, 1280)
- Early stopping variations (patience=10, 50)

#### 3. Real-World Robustness Testing
7 retail environmental conditions:
- Baseline (standard conditions)
- Low Light (poor lighting)
- Bright Light (overexposure)
- Motion Blur (camera/product movement)
- Partial Occlusion (30% masking)
- Perspective Distortion (affine transformations)
- Camera Noise (σ=25) ⚠️ **Primary risk factor**

#### 4. Product Category Analysis
9 major product categories (414 classes):
- Whiskey/Bourbon (31 products)
- Tequila/Mezcal (62 products)
- Vodka (5 products)
- Rum (40 products)
- Gin (4 products)
- Cognac/Brandy (3 products) - **Best: 97.6%** 🏆
- Blended/Canadian (6 products)
- Liqueur/Cream (10 products) - **Hardest: 59.0%** ⚠️
- Other (253 products)

---

## 📊 Key Findings

### Performance Summary

| Configuration | mAP@0.5 | Precision | Recall | Resources | Recommendation |
|--------------|---------|-----------|--------|-----------|----------------|
| **Dual** | 47.6% | 65.7% | 44.1% | 2 images/product | ❌ Inadequate |
| **Quad** | **97.0%** ⭐ | 81.3% | 88.9% | 4 images/product | ✅ **Optimal** |
| **Octal** | **97.6%** 🏆 | 94.1% | 95.7% | 8 images/product | ⚠️ Diminishing returns |
| **Full** | 80.8% | 87.7% | 75.3% | 24 images/product | ❌ Overfitting |

### Critical Discoveries

1. **Quad Configuration is Optimal** ⭐
   - 97.0% mAP@0.5 (only 0.6% below peak)
   - Best cost-performance balance
   - Superior robustness (-10.2% average degradation)
   - **Recommendation:** Use for most deployments

2. **SGD Optimizer Superior** 🎯
   - 97.59% vs Adam (96.95%) vs AdamW (95.90%)
   - 1.6% improvement over adaptive optimizers
   - Better suited for multi-view learning

3. **Camera Noise = Primary Risk** ⚠️
   - 34-50% detection loss across all models
   - More important than view quantity
   - **Recommendation:** Prioritize camera quality

4. **Full Coverage Paradox** 🤔
   - 360° sampling underperforms octal by 16.5%
   - Information redundancy causes overfitting
   - Challenges "more data is better" assumption

5. **Category Performance Varies** 📈
   - Best: Cognac/Brandy (97.6%)
   - Worst: Liqueur/Cream (59.0%)
   - Distinctive features enable strong performance
   - Transparent/similar products most challenging

---

## 🚀 Deployment Guidelines

### By Accuracy Requirements

**For 95-98% Accuracy (Recommended):**
```
Configuration: Quad-view with YOLOv8
Performance: 97.0% mAP@0.5
Resources: 4 images per product
Training Time: ~75 minutes (RTX 3070)
Best For: Most retail automation applications
```

**For Maximum Accuracy (>97%):**
```
Configuration: Octal-view with YOLOv8
Performance: 97.6% mAP@0.5
Resources: 8 images per product
Additional Gain: Only +0.6% over quad
Best For: Sub-1% error critical (pharmaceuticals, high-value)
```

**Budget-Constrained (<80%):**
```
Configuration: Dual-view NOT RECOMMENDED
Performance: 47.6% mAP@0.5 (inadequate)
Alternative: Use quad-view or different architecture
```

### By Deployment Environment

| Environment | Configuration | Expected Degradation | Key Considerations |
|------------|---------------|---------------------|-------------------|
| Controlled (studio) | Quad | <5% | Maintain consistent conditions |
| Typical retail | Quad | ~10% | Good lighting + camera stability |
| Challenging retail | Quad/Octal | 15-20% | **High-quality cameras critical** |
| Outdoor/mobile | Octal minimum | 20-30% | Noise mitigation paramount |

### Cost-Benefit Analysis

**Data Collection Effort (1,000 product catalog):**
- Dual: 2,000 images (~1-2 days, $2K-4K) → **47.6% mAP (inadequate)**
- Quad: 4,000 images (~2-4 days, $2K-4K) → **97.0% mAP** ✅
- Octal: 8,000 images (~4-8 days, $4K-8K) → **97.6% mAP** (only +0.6%)
- Full: 24,000 images (~12-24 days, $12K-24K) → **80.8% mAP** (worse!)

**ROI Winner:** Quad configuration provides **$21-42 per 1% accuracy point**  
Octal costs **$3,333-6,667 per additional 1%** (160× worse ROI)

---

## 🛠️ Technical Specifications

### Hardware
- **GPU:** NVIDIA GeForce RTX 3070 (8GB VRAM)
- **CPU:** 12 cores
- **RAM:** 32GB
- **Storage:** SSD

### Software Stack
```
OS: Ubuntu 22.04 LTS
Python: 3.13.5
PyTorch: 2.8.0 + CUDA 12.8
Ultralytics: 8.3.186
```

### Optimal Training Parameters
```python
{
    "optimizer": "SGD",          # Best: 97.59% (not Adam/AdamW)
    "batch_size": 16,            # RTX 3070 optimal
    "learning_rate": 0.01,       # Linear decay to 0.0001
    "image_size": 640,           # Balance resolution/memory
    "epochs": 100,               # With early stopping
    "patience": 20,              # Early stopping threshold
    "warmup_epochs": 3,          # Stabilization
    "mixed_precision": "FP16",   # Memory efficiency
}
```

---

## 📁 Repository Structure

```
360paper/
├── 📄 FINAL_SUBMISSION_PAPER.md          # Complete manuscript
├── 📊 EXPERIMENT_RESULTS_SUMMARY.md      # Detailed findings
├── 📝 multiview_yolov8_research_paper.md # Original documentation
├── 📄 research_paper_latex.pdf           # IEEE format paper
├── 🔬 Experiment Scripts/
│   ├── ablation_hyperparameter_study.py  # 11 config ablation
│   ├── field_validation.py               # 7 condition testing
│   ├── product_category_analysis.py      # 9 category analysis
│   ├── full_dataset_analysis.py          # Full coverage investigation
│   └── run_all_experiments.py            # Master runner
├── 📊 Analysis Tools/
│   ├── yolov8_vs_yolov11_comparison_plot.py
│   ├── analyze_results.py
│   └── experiment_dashboard.py
├── 📈 Results/
│   └── data/
│       ├── analysis/                     # Graphs and reports
│       └── results/                      # JSON experiment outputs
└── 🎯 Models/
    ├── yolov8n.pt                        # YOLOv8 nano weights
    └── yolov11n.pt                       # YOLOv11 nano weights
```

---

## 🏃 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/sam02425/360Image.git
cd 360Image
```

### 2. Setup Environment
```bash
# Create virtual environment
python3 -m venv multiview_venv
source multiview_venv/bin/activate

# Install dependencies
pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib seaborn scipy scikit-learn
```

### 3. Run Experiments

**Baseline Performance:**
```bash
python ubuntu_multiview_experiment.py
```

**Hyperparameter Ablation:**
```bash
python ablation_hyperparameter_study.py
```

**Robustness Testing:**
```bash
python field_validation.py
```

**Category Analysis:**
```bash
python product_category_analysis.py
```

**Run All Experiments:**
```bash
python run_all_experiments.py
```

---

## 📊 Results and Analysis

### Baseline Performance

<details>
<summary>Click to expand detailed results</summary>

| Configuration | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 Score |
|--------------|---------|--------------|-----------|--------|----------|
| Dual | 0.476 | 0.456 | 0.657 | 0.441 | 0.528 |
| **Quad** | **0.957** | **0.924** | **0.813** | **0.889** | **0.849** |
| Octal | **0.973** | **0.948** | **0.941** | **0.957** | **0.949** |
| Full | 0.808 | 0.785 | 0.877 | 0.753 | 0.810 |

**Key Insights:**
- Dual → Quad: +101% improvement (47.6% → 95.7%)
- Quad → Octal: +1.6% improvement (95.7% → 97.3%)
- Octal → Full: -16.5% decline (97.3% → 80.8%)
</details>

### Hyperparameter Ablation

<details>
<summary>Click to expand ablation study results</summary>

| Configuration | mAP@0.5 | Status | Insight |
|--------------|---------|--------|---------|
| **Baseline (SGD)** | **97.59%** | ✅ | **Optimal** |
| Adam | 96.95% | ✅ | -0.64% vs SGD |
| AdamW | 95.90% | ✅ | -1.69% vs SGD |
| Batch=8 | 95.71% | ✅ | Unstable |
| Batch=32 | — | ❌ OOM | Infeasible |
| LR=0.001 | 95.39% | ✅ | Too conservative |
| LR=0.1 | 95.19% | ✅ | Instability |
| ImgSize=320 | 93.63% | ✅ | Detail loss |
| ImgSize=1280 | — | ❌ OOM | Infeasible |
| Patience=10 | 94.10% | ✅ | Premature |
| Patience=50 | 96.27% | ✅ | Diminishing |
</details>

### Robustness Analysis

<details>
<summary>Click to expand field validation results</summary>

| Condition | Dual | Quad | Octal | Full |
|-----------|------|------|-------|------|
| **Baseline** | 88% / 79.6% | 92% / 82.7% | 97% / 94.7% | 98% / 93.9% |
| Low Light | 82% / 82.9% | 90% / 83.8% | 99% / 92.9% | 98% / 92.0% |
| Bright Light | 92% / 77.4% | 88% / 84.6% | 98% / 93.9% | 98% / 93.1% |
| Motion Blur | 98% / 70.5% | 87% / 80.8% | 101% / 89.3% | 101% / 87.0% |
| Occlusion | 115% / 65.3% | 105% / 72.8% | 116% / 73.4% | 119% / 74.8% |
| Perspective | 150% / 64.4% | 123% / 74.3% | 139% / 80.4% | 116% / 86.5% |
| **Camera Noise** | **45% / 59.4%** | **54% / 50.9%** | **58% / 60.0%** | **64% / 63.2%** |

*Format: Detection % / Confidence %*

**Critical Finding:** Camera noise causes 34-50% detection loss - the primary deployment risk factor.
</details>

### Category Performance

<details>
<summary>Click to expand category analysis</summary>

| Category | Products | Best Model | Performance | Difficulty |
|----------|----------|------------|-------------|------------|
| **Cognac/Brandy** | 3 | Quad | **97.6%** 🏆 | Easy |
| Blended/Canadian | 6 | Quad | 95.5% | Easy |
| Rum | 40 | Quad | 90.5% | Medium |
| Tequila/Mezcal | 62 | Quad | 90.2% | Medium |
| Gin | 4 | Octal | 88.5% | Medium |
| Whiskey/Bourbon | 31 | Quad | 83.3% | Hard |
| Other | 253 | Quad | 81.9% | Hard |
| Vodka | 5 | Octal | 74.6% | Very Hard |
| **Liqueur/Cream** | 10 | Octal | **59.0%** ⚠️ | Very Hard |

**Insight:** Quad model dominates in 7/9 categories, demonstrating strong generalization.
</details>

---

## 📚 Citation

If you use this research in your work, please cite:

```bibtex
@article{patel2025multiview,
  title={Multi-View Training Requirements for Angle-Invariant Retail Object Detection: A Comprehensive Empirical Investigation},
  author={Patel, Saumil and Naini, Rohith},
  journal={Submitted},
  year={2025},
  institution={Lamar University}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Saumil Patel**  
Department of Industrial and Systems Engineering  
Lamar University, Beaumont, TX 77710  
GitHub: [@sam02425](https://github.com/sam02425)

---

## 🙏 Acknowledgments

- Lamar University Department of Industrial and Systems Engineering
- Curry Creations for domain expertise and resources
- Ultralytics for YOLOv8 framework
- PyTorch community for deep learning tools

---

## 📅 Project Timeline

- **Dataset Creation:** August 2024
- **Initial Experiments:** August-September 2024
- **Ablation Studies:** October 2024
- **Robustness Testing:** October 2024
- **Category Analysis:** October 2024
- **Paper Preparation:** October 2025
- **Status:** Ready for journal submission

---

**Last Updated:** October 11, 2025  
**Repository Status:** Active Research Project  
**Paper Status:** Ready for Submission

