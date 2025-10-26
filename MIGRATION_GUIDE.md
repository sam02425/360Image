# Migration Guide

This guide documents repository structural changes and how to update paths and workflows accordingly. It consolidates academic-era materials, standardizes documentation, and establishes canonical entry points for experiments and pipelines.

## Overview
- Documentation reorganized under `docs/` with images in `docs/images/`
- Analysis utilities consolidated under `analysis/` and `analysis/visualizations/`
- Canonical experiment entry points established under `experiments/`
- Legacy scripts archived under `docs/legacy/`
- Environment standardized via `requirements.txt`
- CLI scripts verified to expose `--help` reliably

## Moves and Renames

### Documentation
- `multiview_yolov8_research_paper.md` → `docs/multiview_yolov8_research_paper.md`
- `revised_retail_detection_paper.md` → `docs/revised_retail_detection_paper.md`
- `research_paper_latex.(pdf|tex)` → `docs/research_paper_latex.(pdf|tex)`
- `experimental_results_analysis.png` → `docs/images/experimental_results_analysis.png`
- `training_time_analysis.png` → `docs/images/training_time_analysis.png`

### Analysis
- `yolov8_vs_yolov11_comparison_plot.py` → `analysis/visualizations/yolov8_vs_yolov11_comparison_plot.py`
- `yolov8_vs_yolov11_{f1,map50,map50_95,precision,recall}.png` → `analysis/visualizations/`

### Legacy Archive (new)
- `ablation_hyperparameter_study.py` → `docs/legacy/ablation_hyperparameter_study.py`
- `field_validation.py` → `docs/legacy/field_validation.py`
- `product_category_analysis.py` → `docs/legacy/product_category_analysis.py`
- `full_dataset_analysis.py` → `docs/legacy/full_dataset_analysis.py`
- `eval_octal_on_full.py` → `docs/legacy/eval_octal_on_full.py`
- `train_all_yolo11_on_unified_eval.py` → `docs/legacy/train_all_yolo11_on_unified_eval.py`
- `ubuntu_multiview_yolo11_sam_experiment.py` → `docs/legacy/ubuntu_multiview_yolo11_sam_experiment.py`

## Canonical Entry Points
- Training:
  - `experiments/train_yolov8.py`
  - `experiments/train_yolov11.py`
  - `experiments/train_rtdetr_l.py`
  - `experiments/train_efficientformer_yolo.py`
- Evaluation:
  - `experiments/eval_grounding_dino.py`
  - `experiments/eval_florence2_vlm.py`
- Pipeline:
  - `experiments/deim_pipeline.py`
- Orchestrator:
  - `experiments/run_all_experiments.py`

All scripts expose `--help`; heavy dependencies are lazy-loaded to keep help fast.

## Environment and Tools
- Install dependencies: `pip install -r requirements.txt`
- MLflow logging is enabled in training and pipelines when configured
- DVC optionally used for datasets and artifacts

## Updating Paths and Imports
- Update any hard-coded paths to new locations:
  - Documentation under `docs/` and images under `docs/images/`
  - Analysis figures and scripts under `analysis/` and `analysis/visualizations/`
  - Legacy references should point to `docs/legacy/`
- Prefer relative imports within packages where applicable

## Backwards Compatibility
- Legacy scripts are preserved in `docs/legacy/` for historical context and reproducibility
- Canonical scripts supersede older root-level scripts; prefer using `experiments/` entry points

## New Documentation
- `docs/GIT_WORKFLOW.md`: Git version compatibility, branching, commits, tags
- `docs/DOCUMENTATION_STANDARDS.md`: Markdown formatting and structure standards
- `docs/CODE_DOCSTRING_GUIDE.md`: Python docstring style and examples
- `docs/INDEX.md`: Navigation index for documentation

## Verification
- Run CLI smoke tests: `pytest -q`
- Manually verify help:
  - `python experiments/deim_pipeline.py --help`
  - `python experiments/train_yolov8.py --help`

## Notes
- Keep `CHANGELOG.md` updated for significant changes
- Prefer small, focused pull requests per `docs/GIT_WORKFLOW.md`