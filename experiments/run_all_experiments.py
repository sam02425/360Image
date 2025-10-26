#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all model training/evaluation scripts sequentially and summarize results in MLflow.
Includes: YOLOv8, YOLOv11, RT-DETR-L, Grounding DINO eval, Florence-2 VLM eval, DEIM pipeline.

Usage example:
    python experiments/run_all_experiments.py \
        --images data/images/val --labels data/labels/val \
        --classes data/classes.yaml --data data/dataset.yaml \
        --yolov8_model yolov8n.pt --yolov11_model yolov11n.pt --rtdetr_model rtdetrl.pt

This orchestrator assumes the individual scripts are in experiments/ and accept the specified arguments.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# optional DVC
dvc_available = False
try:
    import dvc.api  # noqa
    dvc_available = True
except Exception:
    dvc_available = False


SCRIPTS = {
    "yolov8": "experiments/train_yolov8.py",
    "yolov11": "experiments/train_yolov11.py",
    "rtdetr": "experiments/train_rtdetr_l.py",
    "grounding_dino": "experiments/eval_grounding_dino.py",
    "florence2": "experiments/eval_florence2_vlm.py",
    "deim": "experiments/deim_pipeline.py",
}

RUN_NAMES = {
    "yolov8": "train_yolov8",
    "yolov11": "train_yolov11",
    "rtdetr": "train_rtdetr_l",
    "grounding_dino": "eval_grounding_dino",
    "florence2": "eval_vlm_recognition",
    "deim": "DEIM_pipeline",
}


def parse_args():
    """Parse command-line arguments for the experiment orchestrator.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(description="Run all retail experiments with MLflow logging")
    p.add_argument('--images', type=str, required=True)
    p.add_argument('--labels', type=str, required=True)
    p.add_argument('--classes', type=str, required=True)
    p.add_argument('--data', type=str, required=True, help='Dataset YAML for Ultralytics training scripts')
    p.add_argument('--yolov8_model', type=str, default='yolov8n.pt')
    p.add_argument('--yolov11_model', type=str, default='yolov11n.pt')
    p.add_argument('--rtdetr_model', type=str, default='rtdetrl.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--gdino_prompts', type=str, default=None, help='JSON or TXT of prompts for Grounding DINO')
    p.add_argument('--deim_conf', type=float, default=0.25)
    p.add_argument('--deim_iou', type=float, default=0.5)
    p.add_argument('--use_size_prompts', action='store_true')
    p.add_argument('--skip', type=str, default='', help='Comma list: yolov8,yolov11,rtdetr,grounding_dino,florence2,deim')
    return p.parse_args()


def run_script(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or str(Path.cwd()), stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"Script failed: {' '.join(cmd)}")
    return result.returncode


def aggregate_mlflow_metrics():
    client = MlflowClient()
    exp = client.get_experiment_by_name("Retail_Model_Experiments")
    if not exp:
        print("MLflow experiment not found; skipping aggregation.")
        return {}
    metrics = {}
    for key, run_name in RUN_NAMES.items():
        runs = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1, order_by=["attributes.start_time DESC"])
        if runs:
            r = runs[0]
            metrics[key] = r.data.metrics
    return metrics


if __name__ == '__main__':
    args = parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Retail_Model_Experiments")
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

    skip = set([s.strip() for s in args.skip.split(',') if s.strip()])

    with mlflow.start_run(run_name="orchestrator_all_models"):
        mlflow.log_param("images", args.images)
        mlflow.log_param("labels", args.labels)
        mlflow.log_param("classes", args.classes)
        mlflow.log_param("data_yaml", args.data)
        mlflow.log_param("yolov8_model", args.yolov8_model)
        mlflow.log_param("yolov11_model", args.yolov11_model)
        mlflow.log_param("rtdetr_model", args.rtdetr_model)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("imgsz", args.imgsz)
        mlflow.log_param("batch", args.batch)
        mlflow.log_param("device", args.device or '')
        mlflow.log_param("gdino_prompts", args.gdino_prompts or '')
        mlflow.log_param("deim_conf", args.deim_conf)
        mlflow.log_param("deim_iou", args.deim_iou)
        mlflow.log_param("use_size_prompts", args.use_size_prompts)
        mlflow.log_param("dvc_available", dvc_available)

        # Optional: log DVC dataset version info
        if dvc_available:
            try:
                import dvc.api as dvcapi
                dataset_url = dvcapi.get_url(args.images)
                mlflow.log_param("dvc_dataset_url", dataset_url)
            except Exception as e:
                mlflow.log_param("dvc_dataset_url_error", str(e))

        # YOLOv8
        if "yolov8" not in skip and Path(SCRIPTS["yolov8"]).exists():
            run_script([
                sys.executable, SCRIPTS["yolov8"],
                "--model", args.yolov8_model, "--data", args.data,
                "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
                "--batch", str(args.batch)
            ])
        # YOLOv11
        if "yolov11" not in skip and Path(SCRIPTS["yolov11"]).exists():
            run_script([
                sys.executable, SCRIPTS["yolov11"],
                "--model", args.yolov11_model, "--data", args.data,
                "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
                "--batch", str(args.batch)
            ])
        # RT-DETR-L
        if "rtdetr" not in skip and Path(SCRIPTS["rtdetr"]).exists():
            run_script([
                sys.executable, SCRIPTS["rtdetr"],
                "--model", args.rtdetr_model, "--data", args.data,
                "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
                "--batch", str(args.batch)
            ])
        # Grounding DINO eval
        if "grounding_dino" not in skip and Path(SCRIPTS["grounding_dino"]).exists():
            cmd = [
                sys.executable, SCRIPTS["grounding_dino"],
                "--images", args.images, "--labels", args.labels,
                "--classes", args.classes
            ]
            if args.gdino_prompts:
                cmd += ["--prompts", args.gdino_prompts]
            run_script(cmd)
        # Florence-2 / CLIP eval
        if "florence2" not in skip and Path(SCRIPTS["florence2"]).exists():
            cmd = [
                sys.executable, SCRIPTS["florence2"],
                "--images", args.images, "--labels", args.labels,
                "--classes", args.classes
            ]
            if args.use_size_prompts:
                cmd += ["--use_size_prompts"]
            run_script(cmd)
        # DEIM
        if "deim" not in skip and Path(SCRIPTS["deim"]).exists():
            cmd = [
                sys.executable, SCRIPTS["deim"],
                "--model", args.yolov8_model,
                "--images", args.images, "--labels", args.labels,
                "--classes", args.classes,
                "--conf", str(args.deim_conf), "--iou", str(args.deim_iou)
            ]
            if args.use_size_prompts:
                cmd += ["--add_size_prompts"]
            run_script(cmd)

        # Aggregate metrics and log summary
        metrics = aggregate_mlflow_metrics()
        for key, m in metrics.items():
            # log a few common metrics if present
            for mk, mv in m.items():
                mlflow.log_metric(f"{key}_{mk}", float(mv))
        print("Orchestration complete. Metrics aggregated to MLflow.")