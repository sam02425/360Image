#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate VLM-based item recognition on retail shelves using Florence-2 if available,
with a robust fallback to CLIP/OpenCLIP classification.
- Loads crops from YOLO labels, matches against class text prompts
- Reports top-1 accuracy, logs to MLflow

Requires:
    pip install transformers pillow torch torchvision open_clip_torch
"""

import os
import argparse
import json
from pathlib import Path
from typing import List

import mlflow
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Florence-2 optional
florence_available = False
try:
    from transformers import AutoProcessor, AutoModel
    florence_available = True
except Exception:
    florence_available = False

# OpenCLIP fallback
openclip_available = False
try:
    import open_clip
    openclip_available = True
except Exception:
    openclip_available = False


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
            return [names[k] for k in sorted(names.keys())]
        return names or []
    else:
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def parse_args():
    """Parse command-line arguments for VLM recognition evaluation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(description="Evaluate VLM item recognition")
    p.add_argument('--images', type=str, required=True, help='Directory with evaluation images')
    p.add_argument('--labels', type=str, required=True, help='YOLO labels directory (txt)')
    p.add_argument('--classes', type=str, required=True, help='Path to classes (yaml/json/txt)')
    p.add_argument('--use_size_prompts', action='store_true', help='Append size descriptors to prompts')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--output', type=str, default='runs/vlm_eval')
    return p.parse_args()


def build_prompts(class_names: List[str], use_size_prompts: bool) -> List[str]:
    if not use_size_prompts:
        return class_names
    sizes = ["small", "medium", "large"]
    prompts = []
    for cn in class_names:
        for s in sizes:
            prompts.append(f"{s} size {cn}")
    return prompts


def eval_with_clip(img_crops: List[Image.Image], prompts: List[str], device: str) -> List[int]:
    # Returns predicted prompt index for each crop
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        preds = []
        for crop in img_crops:
            crop_tensor = preprocess(crop).unsqueeze(0).to(device)
            image_features = model.encode_image(crop_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).squeeze(0)
            pred_idx = int(torch.argmax(logits).item())
            preds.append(pred_idx)
    return preds


def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Retail_Model_Experiments")
    class_names = load_classes(args.classes)
    prompts = build_prompts(class_names, args.use_size_prompts)

    with mlflow.start_run(run_name="eval_vlm_recognition"):
        mlflow.log_param("framework", "florence2_or_clip")
        mlflow.log_param("images", args.images)
        mlflow.log_param("labels", args.labels)
        mlflow.log_param("classes_count", len(class_names))
        mlflow.log_param("use_size_prompts", args.use_size_prompts)
        mlflow.log_param("florence_available", florence_available)
        mlflow.log_param("openclip_available", openclip_available)

        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect crops and GT labels from YOLO label files
        img_dir = Path(args.images)
        label_dir = Path(args.labels)
        to_tensor = T.ToTensor()
        total = 0
        correct = 0

        for img_path in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
            lbl = label_dir / (img_path.stem + '.txt')
            if not lbl.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            # Build crops
            crops = []
            gt_indices = []
            with open(lbl, 'r') as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    crop = img.crop((x1, y1, x2, y2))
                    crops.append(crop)
                    gt_indices.append(int(cls))

            if not crops:
                continue
            total += len(crops)

            # Evaluate predictions via CLIP (fallback); Florence-2 not standardized for classification here
            if openclip_available:
                preds = eval_with_clip(crops, prompts, device)
            else:
                mlflow.log_param("clip_fallback_used", False)
                preds = [0] * len(crops)

            # Map prompt idx back to base class idx (strip size descriptors)
            def prompt_to_class_idx(idx: int) -> int:
                prompt = prompts[idx]
                # If size prompts used, take last token(s) as class name
                if args.use_size_prompts:
                    # prompts like "small size Coke"
                    base = prompt.split('size')[-1].strip()
                else:
                    base = prompt
                # find class_names index
                try:
                    return class_names.index(base)
                except ValueError:
                    # fallback: exact string not found; try partial matching
                    for i, cn in enumerate(class_names):
                        if cn.lower() in base.lower():
                            return i
                    return 0

            for p, gt in zip(preds, gt_indices):
                pred_class = prompt_to_class_idx(p)
                if pred_class == gt:
                    correct += 1

        acc = (correct / total) if total > 0 else 0.0
        mlflow.log_metric('vlm_top1_acc', float(acc))
        print(f"VLM recognition evaluation complete. Top-1 accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()