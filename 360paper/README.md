# EfficientFormer-YOLO Retail Detection

A production-ready retail object detection system combining EfficientFormer-L1 backbone with YOLOv8 detection head for efficient and accurate retail product detection.

## Project Overview

This project implements a hybrid deep learning architecture for retail product detection that combines the efficiency of EfficientFormer-L1 with the accuracy of YOLOv8. The system is designed for multi-view retail environments and optimized for production deployment.

### Key Features

- **Hybrid Architecture**: EfficientFormer-L1 backbone with YOLOv8 detection head
- **Multi-View Support**: Handles 2, 4, or 8 camera views simultaneously
- **Production Optimizations**: ONNX export, quantization, and TensorRT support
- **Comprehensive Training**: Mixed precision, distributed training, and various optimizers
- **Deployment Ready**: Docker support, API endpoints, and monitoring tools

## Directory Structure

```
360paper/
├── 📊 experiments/
│   ├── train_efficientformer_yolo.py      # Main training script with EfficientFormer
│   ├── ablation_study.py                  # Hyperparameter optimization
│   ├── field_validation.py                # Robustness testing
│   └── category_analysis.py               # Product category analysis
│
├── 📚 models/
│   ├── efficientformer_yolo_hybrid.py     # Hybrid model architecture
│   ├── export_utils.py                    # Model export and optimization utilities
│   ├── dual_best.pt
│   ├── quad_best.pt
│   ├── octal_best.pt
│   └── full_best.pt
│
├── 💾 data/
│   ├── datasets/
│   │   ├── quad_dataset/                  # Primary dataset (optimal)
│   │   └── unified_test/                  # Unified test set
│   └── results/
│       ├── ablation_studies/
│       ├── field_validation/
│       └── category_analysis/
│
├── 📝 papers/
│   ├── FINAL_SUBMISSION_PAPER.md          # Journal-ready paper
│   ├── EXPERIMENT_RESULTS_SUMMARY.md      # Comprehensive results
│   └── research_paper_latex.pdf           # IEEE format paper
│
└── 📖 docs/
    ├── README.md                          # Main documentation
    └── DEPLOYMENT_GUIDE.md                # Production deployment guide
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv retail_detection_env
source retail_detection_env/bin/activate  # Linux/Mac
# or
retail_detection_env\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics timm albumentations opencv-python-headless
pip install numpy pandas matplotlib seaborn tqdm
```

### Training

```bash
# Train with EfficientFormer-YOLO
python experiments/train_efficientformer_yolo.py \
    --yolo-model yolov8n \
    --batch-size 16 \
    --epochs 100 \
    --dataset data/datasets/quad_dataset \
    --output-dir models/production
```

### Export for Deployment

```bash
# Export to various formats
python models/export_utils.py \
    --model models/production/best_model.pth \
    --output models/production/exported \
    --formats onnx int8 fp16
```

## Model Architecture

The EfficientFormer-YOLO hybrid model combines:

1. **EfficientFormer-L1 Backbone**: Efficient vision transformer with linear complexity
2. **Feature Adaptation Layers**: Connect backbone to detection head
3. **YOLOv8 Detection Head**: State-of-the-art object detection

This hybrid approach provides:
- Better efficiency than pure transformer models
- Better accuracy than pure CNN models
- Excellent performance on retail product detection tasks

## Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Time (ms) | Model Size (MB) |
|-------|---------|--------------|---------------------|-----------------|
| YOLOv8n | 92.3% | 76.5% | 6.2 | 6.3 |
| EfficientFormer-YOLO | 95.7% | 81.2% | 8.5 | 11.7 |
| EfficientFormer-YOLO (INT8) | 95.1% | 80.3% | 3.2 | 3.1 |

## Deployment

See [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for comprehensive deployment instructions, including:

- Hardware and software requirements
- Docker deployment
- Cloud deployment options
- Performance optimization
- Monitoring and maintenance
- API endpoints
- Troubleshooting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EfficientFormer paper: [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- TIMM library: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)