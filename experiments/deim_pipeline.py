#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEIM (Detector + Embedding Identification Model) pipeline for retail items:
- Detect with YOLOv8/YOLOv11/RT-DETR-L (Ultralytics model path)
- Classify each crop with OpenCLIP embeddings against class prompts
- Compute detection metrics (precision/recall) and recognition accuracy
- Log params, metrics, and sample artifacts to MLflow

Requires:
    pip install ultralytics open_clip_torch mlflow pillow torch torchvision
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import mlflow
import numpy as np
from PIL import Image
# Lazy-import heavy deps (torch, ultralytics, open_clip) to keep --help fast

# Heavy libraries (Ultralytics, OpenCLIP) are imported lazily within functions


def load_classes(classes_path: str) -> List[str]:
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
            return [names[k] for k in sorted(names.keys())]
        return names or []
    else:
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def build_prompts(class_names: List[str], add_sizes: bool) -> List[str]:
    if not add_sizes:
        return class_names
    sizes = ["small", "medium", "large"]
    prompts = []
    for cn in class_names:
        for s in sizes:
            prompts.append(f"{s} size {cn}")
    return prompts


def clip_setup(device: str):
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.to(device)
    model.eval()
    return model, tokenizer, preprocess


def clip_classify_crops(model, tokenizer, preprocess, crops: List[Image.Image], prompts: List[str], device: str) -> List[int]:
    import torch
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        preds = []
        for crop in crops:
            img_t = preprocess(crop).unsqueeze(0).to(device)
            image_features = model.encode_image(img_t)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).squeeze(0)
            pred_idx = int(torch.argmax(logits).item())
            preds.append(pred_idx)
        return preds


def prompt_to_class_idx(prompts: List[str], class_names: List[str], idx: int, add_sizes: bool) -> int:
    prompt = prompts[idx]
    base = prompt.split('size')[-1].strip() if add_sizes else prompt
    try:
        return class_names.index(base)
    except ValueError:
        for i, cn in enumerate(class_names):
            if cn.lower() in base.lower():
                return i
        return 0


def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = areaA + areaB - inter + 1e-6
    return inter / denom


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    ann = []
    if not label_path.exists():
        return ann
    with open(label_path, 'r') as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            ann.append((int(cls), cx, cy, bw, bh))
    return ann


def yolo_norm_to_xyxy(w: int, h: int, cx: float, cy: float, bw: float, bh: float):
    x1 = (cx - bw/2) * w
    y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w
    y2 = (cy + bh/2) * h
    return [x1, y1, x2, y2]


def run_deim(model_path: str, images_dir: str, labels_dir: str, classes_path: str, conf: float, iou_thres: float, add_sizes: bool, device: str):
    import torch
    from ultralytics import YOLO

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Retail_Model_Experiments")

    class_names = load_classes(classes_path)
    prompts = build_prompts(class_names, add_sizes)

    with mlflow.start_run(run_name="DEIM_pipeline"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("images_dir", images_dir)
        mlflow.log_param("labels_dir", labels_dir)
        mlflow.log_param("classes_count", len(class_names))
        mlflow.log_param("add_size_prompts", add_sizes)
        mlflow.log_param("conf", conf)
        mlflow.log_param("iou_thres", iou_thres)

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        det_model = YOLO(model_path)
        clip_model, clip_tokenizer, clip_preprocess = clip_setup(device)

        img_paths = sorted(list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png')))
        tp = 0
        fp = 0
        fn = 0
        recog_total = 0
        recog_correct = 0

        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            # ground truth
            label_path = Path(labels_dir) / (Path(img_path).stem + '.txt')
            gts = read_yolo_labels(label_path)
            gt_boxes = [yolo_norm_to_xyxy(w, h, *gt[1:]) for gt in gts]
            gt_classes = [gt[0] for gt in gts]
            matched_gt = [False]*len(gt_boxes)

            # detection
            res = det_model.predict(source=str(img_path), conf=conf, iou=iou_thres, verbose=False)[0]
            det_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
            det_scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))

            crops = []
            for box in det_boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                crops.append(img.crop((x1, y1, x2, y2)))

            # classify crops
            if crops:
                pred_prompt_idxs = clip_classify_crops(clip_model, clip_tokenizer, clip_preprocess, crops, prompts, device)
            else:
                pred_prompt_idxs = []

            # evaluate matching
            for i, box in enumerate(det_boxes):
                # find best gt by IoU
                best_iou = 0.0
                best_j = -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched_gt[j]:
                        continue
                    iou_val = iou(box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j
                if best_j >= 0 and best_iou >= 0.5:
                    tp += 1
                    matched_gt[best_j] = True
                    recog_total += 1
                    pred_class_idx = prompt_to_class_idx(prompts, class_names, pred_prompt_idxs[i], add_sizes)
                    if pred_class_idx == gt_classes[best_j]:
                        recog_correct += 1
                else:
                    fp += 1
            fn += matched_gt.count(False)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        recog_acc = (recog_correct / recog_total) if recog_total > 0 else 0.0

        mlflow.log_metric("deim_precision", float(precision))
        mlflow.log_metric("deim_recall", float(recall))
        mlflow.log_metric("deim_f1", float(f1))
        mlflow.log_metric("deim_recognition_acc", float(recog_acc))

        print(f"DEIM complete. P={precision:.3f} R={recall:.3f} F1={f1:.3f} RecognAcc={recog_acc:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="DEIM pipeline: detector + embedding recognition")
    p.add_argument('--model', type=str, required=True, help='Ultralytics model path (YOLOv8/11/RT-DETR-L)')
    p.add_argument('--images', type=str, required=True, help='Directory of images')
    p.add_argument('--labels', type=str, required=True, help='Directory of YOLO labels')
    p.add_argument('--classes', type=str, required=True, help='Path to classes (yaml/json/txt)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--add_size_prompts', action='store_true')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_deim(args.model, args.images, args.labels, args.classes, args.conf, args.iou, args.add_size_prompts, args.device)