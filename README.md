# Retail 360 Image - Detection & Recognition

A unified repository for retail shelf item detection, recognition, and analysis across YOLOv8/YOLOv11, RT‑DETR‑L, Grounding DINO, and VLM (Florence‑2/CLIP) baselines. Includes MLflow tracking, optional DVC dataset versioning, and an orchestrator to run all experiments.

## Repository Structure
- `experiments/` — Training/evaluation scripts and orchestrator
  - `train_yolov8.py`, `train_yolov11.py`, `train_rtdetr_l.py`
  - `eval_grounding_dino.py`, `eval_florence2_vlm.py`, `deim_pipeline.py`
  - `run_all_experiments.py` (canonical orchestrator)
- `analysis/` — Analysis assets and visualizations
  - `statistical_analysis/`, `data_quality/`, `visualizations/`
- `data/` — Analysis graphs, metrics, and reports
- `reports/` — Aggregated experiment summaries
- `docs/` — Documentation and papers
  - `images/` figures and plots
  - `legacy/` historical scripts (migrated)
- `model_selection.py` — Utility to benchmark and recommend models (MLflow integrated)

## Key Capabilities
- Large‑scale training tuned for small, visually similar products
- MLflow logging of metrics, params, and artifacts across all models
- Optional DVC integration to record dataset versions
- DEIM pipeline (detector + CLIP embeddings) for size‑aware recognition

## Setup
1. Create a Python environment (3.10+ recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare Ultralytics dataset YAML and class names file (yaml/json/txt).
3. Ensure MLflow is configured (default local `file:./mlruns`). If using DVC, initialize your dataset within the repo.

## Running Experiments
- Train YOLOv8:
  ```bash
  python experiments/train_yolov8.py --model yolov8n.pt --data path/to/dataset.yaml --epochs 50 --imgsz 640 --batch 16
  ```
- Train YOLOv11:
  ```bash
  python experiments/train_yolov11.py --model yolov11n.pt --data path/to/dataset.yaml --epochs 50 --imgsz 640 --batch 16
  ```
- Train RT‑DETR‑L:
  ```bash
  python experiments/train_rtdetr_l.py --model rtdetrl.pt --data path/to/dataset.yaml --epochs 50 --imgsz 640 --batch 16
  ```
- Evaluate Grounding DINO (promptable detection):
  ```bash
  python experiments/eval_grounding_dino.py --images data/images/val --labels data/labels/val --classes data/classes.yaml --prompts prompts.txt
  ```
- Evaluate VLM recognition (Florence‑2 / CLIP fallback):
  ```bash
  python experiments/eval_florence2_vlm.py --images data/images/val --labels data/labels/val --classes data/classes.yaml --use_size_prompts
  ```
- DEIM pipeline (detector + embedding recognition):
  ```bash
  python experiments/deim_pipeline.py --model yolov8n.pt --images data/images/val --labels data/labels/val --classes data/classes.yaml --conf 0.25 --iou 0.5 --add_size_prompts
  ```
- Orchestrator (run all, aggregate MLflow metrics):
  ```bash
  python experiments/run_all_experiments.py --images data/images/val --labels data/labels/val --classes data/classes.yaml --data data/dataset.yaml --epochs 50 --imgsz 640 --batch 16
  ```

## Dependencies (core)
`torch`, `torchvision`, `ultralytics`, `mlflow`, `dvc`, `open_clip_torch`, `transformers`, `pillow`, `opencv-python`, `albumentations`, `timm`, `psutil`, `GPUtil`, `pyyaml`

## Notes
- Historical scripts related to prior papers are under `docs/legacy/`.
- The canonical orchestrator is `experiments/run_all_experiments.py`.
- All scripts log to MLflow under the `Retail_Model_Experiments` experiment.

