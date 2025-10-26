#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train YOLOv11 for Retail Item Detection with MLflow + DVC tracking.
- Uses Ultralytics YOLOv11 for training
- Logs metrics, params, artifacts to MLflow
- Records dataset version via DVC if available
"""

import os
from pathlib import Path
import argparse
import mlflow
import yaml

# Optional DVC integration
try:
    import dvc.api as dvc_api
except Exception:
    dvc_api = None

# Ultralytics
try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics not found. Please install with: pip install ultralytics")
    raise


def log_dvc_dataset_info(data_yaml: str):
    if dvc_api is None:
        mlflow.log_param("dvc_enabled", False)
        return
    try:
        mlflow.log_param("dvc_enabled", True)
        with open(data_yaml, 'r') as f:
            dy = yaml.safe_load(f)
        for key in ["train", "val", "test"]:
            if key in dy and isinstance(dy[key], str):
                try:
                    url = dvc_api.get_url(dy[key])
                    mlflow.log_param(f"dvc_{key}_url", url)
                except Exception:
                    mlflow.log_param(f"dvc_{key}_url", "not_dvc_tracked")
    except Exception as e:
        mlflow.log_param("dvc_error", str(e))


def parse_args():
    """Parse command-line arguments for YOLOv11 training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(description="Train YOLOv11 with MLflow+DVC")
    p.add_argument('--data', type=str, required=True, help='Dataset YAML file path')
    p.add_argument('--model', type=str, default='yolo11m.pt', help='Model weights or name')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', type=str, default='runs', help='Ultralytics project dir')
    p.add_argument('--name', type=str, default='yolo11_retail', help='Run name')
    p.add_argument('--device', type=str, default=None, help='cuda or cpu')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cache', action='store_true', help='Cache images for faster training')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exist_ok', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Retail_Model_Experiments")

    with mlflow.start_run(run_name=f"train_yolo11_{Path(args.model).stem}"):
        mlflow.log_param("framework", "ultralytics_yolo11")
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

        log_dvc_dataset_info(args.data)

        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            resume=args.resume,
            seed=args.seed,
            exist_ok=args.exist_ok
        )

        # Locate results
        try:
            save_dir = Path(model.trainer.save_dir)
        except Exception:
            save_dir = Path(args.project) / 'detect' / 'train'
        mlflow.log_param("ultralytics_save_dir", str(save_dir))

        # Log artifacts and metrics
        results_csv = save_dir / 'results.csv'
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv))
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                last = df.iloc[-1]
                for col in [
                    'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)',
                    'fitness']:
                    if col in df.columns:
                        mlflow.log_metric(col.replace('(B)', ''), float(last[col]))
            except Exception as e:
                mlflow.log_param("metrics_parse_error", str(e))

        weights_dir = save_dir / 'weights'
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt))
        if last_pt.exists():
            mlflow.log_artifact(str(last_pt))
        print(f"Training complete. Artifacts logged to MLflow. Save dir: {save_dir}")


if __name__ == '__main__':
    main()