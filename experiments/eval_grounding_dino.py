#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Grounding DINO zero-shot detection on retail items.
- Uses text prompts (class names) to detect items
- Computes lightweight metrics (precision/recall/mAP@0.5 approx)
- Logs metrics, params, artifacts to MLflow

Requires:
    pip install groundingdino transformers torchvision pillow
"""

import os
import argparse
import json
from pathlib import Path
from typing import List

import mlflow
import numpy as np
from PIL import Image, ImageDraw

# Grounding DINO imports (runtime optional)
try:
    import groundingdino
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    import torch
    import torchvision.transforms as T
except Exception as e:
    groundingdino = None


def load_classes(classes_path: str) -> List[str]:
    # classes_path can be a json list or a yaml dataset with 'names'
    if classes_path.endswith('.json'):
        with open(classes_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get('classes', [])
    elif classes_path.endswith('.yaml'):
        import yaml
        with open(classes_path, 'r') as f:
            dy = yaml.safe_load(f)
        names = dy.get('names') or dy.get('class_names')
        if isinstance(names, dict):
            # YOLO style dict {0: 'classA', ...}
            return [names[k] for k in sorted(names.keys())]
        return names or []
    else:
        # plain text file, one class per line
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def simple_iou(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    union = ua + ub - inter
    return inter/union if union > 0 else 0.0


def parse_args():
    """Parse command-line arguments for Grounding DINO evaluation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(description="Evaluate Grounding DINO on retail items")
    p.add_argument('--images', type=str, required=True, help='Directory with evaluation images')
    p.add_argument('--labels', type=str, required=False, help='COCO-style annotations JSON or YOLO labels dir')
    p.add_argument('--classes', type=str, required=True, help='Path to classes (yaml/json/txt)')
    p.add_argument('--text-threshold', type=float, default=0.25)
    p.add_argument('--box-threshold', type=float, default=0.25)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--output', type=str, default='runs/grounding_dino_eval')
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Retail_Model_Experiments")

    class_names = load_classes(args.classes)

    with mlflow.start_run(run_name="eval_grounding_dino"):
        mlflow.log_param("framework", "grounding_dino")
        mlflow.log_param("images", args.images)
        mlflow.log_param("labels", args.labels)
        mlflow.log_param("classes_count", len(class_names))
        mlflow.log_param("text_threshold", args.text_threshold)
        mlflow.log_param("box_threshold", args.box_threshold)

        if groundingdino is None:
            mlflow.log_param("groundingdino_import_error", True)
            print("Grounding DINO not installed; please install required packages.")
            return

        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Use pretrained config (example; adjust paths if needed)
        # Users can replace with local checkpoint paths; here we use torch.hub suggested weights
        cfg = SLConfig.fromfile("groundingdino/config/GroundingDINO_SwinT_OGC.py")
        model = build_model(cfg.model)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth',
            map_location='cpu', progress=True
        )
        model.load_state_dict(clean_state_dict(checkpoint))
        model.to(device)
        model.eval()

        transform = T.Compose([
            T.ToTensor(),
        ])

        img_dir = Path(args.images)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Simple label loader (YOLO txt per image). If labels not provided, we do zero-shot without GT metrics.
        label_dir = Path(args.labels) if args.labels and os.path.isdir(args.labels) else None

        precisions = []
        recalls = []
        aps = []
        count = 0

        for img_path in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            tensor = transform(img).to(device)

            # Compose text prompt as comma-separated classes
            text_prompt = ", ".join(class_names)
            with torch.no_grad():
                outputs = model.inference([tensor], [text_prompt], box_threshold=args.box_threshold, text_threshold=args.text_threshold)

            boxes = outputs[0]['boxes'] if 'boxes' in outputs[0] else []
            labels = outputs[0].get('labels', [])
            scores = outputs[0].get('scores', [])

            # Convert to xyxy
            dets = []
            for b, s, l in zip(boxes, scores, labels):
                if isinstance(b, torch.Tensor):
                    b = b.cpu().numpy().tolist()
                dets.append((b, s, l))

            # Draw and save visualization
            vis = img.copy()
            draw = ImageDraw.Draw(vis)
            for (b, s, l) in dets:
                x1, y1, x2, y2 = b
                draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
                draw.text((x1, max(0, y1-10)), f"{l}:{s:.2f}", fill=(255,0,0))
            vis_path = out_dir / f"{img_path.stem}_gdino.png"
            vis.save(vis_path)
            mlflow.log_artifact(str(vis_path))

            # Compute simple metrics if GT available (YOLO txt: class x y w h normalized)
            if label_dir is not None:
                yolo_lbl = label_dir / (img_path.stem + '.txt')
                gt_boxes = []
                gt_labels = []
                if yolo_lbl.exists():
                    with open(yolo_lbl, 'r') as f:
                        for ln in f:
                            parts = ln.strip().split()
                            if len(parts) == 5:
                                cls, cx, cy, bw, bh = map(float, parts)
                                # convert to xyxy
                                x1 = (cx - bw/2) * w
                                y1 = (cy - bh/2) * h
                                x2 = (cx + bw/2) * w
                                y2 = (cy + bh/2) * h
                                gt_boxes.append([x1,y1,x2,y2])
                                gt_labels.append(int(cls))
                # Match by IoU>0.5, same label string if available
                tp = 0
                fp = 0
                for (b, s, l) in dets:
                    # find best iou with any GT
                    ious = [simple_iou(b, gtb) for gtb in gt_boxes]
                    if len(ious) and max(ious) > 0.5:
                        tp += 1
                    else:
                        fp += 1
                fn = max(0, len(gt_boxes) - tp)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precisions.append(precision)
                recalls.append(recall)
                aps.append(precision)  # proxy
                count += 1

        if count > 0:
            mlflow.log_metric('precision', float(np.mean(precisions)))
            mlflow.log_metric('recall', float(np.mean(recalls)))
            mlflow.log_metric('mAP50_proxy', float(np.mean(aps)))
        print("Grounding DINO evaluation complete. Artifacts logged.")


if __name__ == '__main__':
    main()