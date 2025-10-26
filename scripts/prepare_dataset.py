#!/usr/bin/env python3
"""
Prepare a YOLO dataset from either Roboflow or a local directory.

Options:
- Source 'roboflow': downloads the dataset using provided API details.
- Source 'local': validates a local dataset directory and optionally writes a data.yaml.

Usage examples:
  # Roboflow (env var or flag for API key)
  python scripts/prepare_dataset.py \
      --source roboflow \
      --workspace lamar-university-venef \
      --project liquor-data \
      --version 4 \
      --format yolov8

  # Local dataset (generate data.yaml)
  python scripts/prepare_dataset.py \
      --source local \
      --dataset-dir /path/to/dataset \
      --classes docs/classes.txt

Notes:
- Roboflow API key can be set via --api-key or ROBOFLOW_API_KEY env var.
- For local datasets, classes can be provided as .txt, .json, or .yaml.
- Prints the resolved data.yaml path to stdout for use with training scripts.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_classes(classes_path: Path) -> List[str]:
    ext = classes_path.suffix.lower()
    text = classes_path.read_text(encoding="utf-8")
    if ext == ".txt":
        return [line.strip() for line in text.splitlines() if line.strip()]
    if ext == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            if "names" in data and isinstance(data["names"], list):
                return [str(x) for x in data["names"]]
            if "classes" in data and isinstance(data["classes"], list):
                return [str(x) for x in data["classes"]]
        raise ValueError("JSON classes file must be a list or dict with 'names'/'classes'.")
    if ext in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            if "names" in data and isinstance(data["names"], list):
                return [str(x) for x in data["names"]]
        raise ValueError("YAML classes file must be a list or dict with 'names'.")
    raise ValueError(f"Unsupported classes file type: {ext}")


def write_data_yaml(dataset_dir: Path, classes: List[str], out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        out_path = dataset_dir / "data.yaml"
    data = {
        "train": str((dataset_dir / "train/images").resolve()),
        "val": str((dataset_dir / "val/images").resolve()),
        "test": str((dataset_dir / "test/images").resolve()),
        "names": classes,
    }
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def validate_local_structure(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train/images",
        dataset_dir / "train/labels",
        dataset_dir / "val/images",
        dataset_dir / "val/labels",
        dataset_dir / "test/images",
        dataset_dir / "test/labels",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset directories: " + ", ".join(str(p) for p in missing)
        )


def resolve_roboflow_dataset(args) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow is not installed. Install with: pip install roboflow", file=sys.stderr)
        sys.exit(2)

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Roboflow API key is required via --api-key or ROBOFLOW_API_KEY.", file=sys.stderr)
        sys.exit(2)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    # Download extracts the dataset locally; default to current working dir
    ds = version.download(args.format)
    # Attempt to locate data.yaml inside the downloaded directory
    base = Path(getattr(ds, "location", os.getcwd())).resolve()
    data_yaml = None
    # Common patterns
    candidates = list(base.rglob("data.yaml"))
    if candidates:
        data_yaml = candidates[0]
    else:
        # Fallback: search for *.yaml that looks like a YOLO config
        for f in base.rglob("*.yaml"):
            try:
                parsed = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(parsed, dict) and {"train", "val"}.issubset(parsed.keys()):
                    data_yaml = f
                    break
            except Exception:
                continue
    if not data_yaml:
        print(f"Could not find data.yaml under {base}", file=sys.stderr)
        sys.exit(2)
    return data_yaml


def resolve_local_dataset(args) -> Path:
    dataset_dir = Path(args.dataset_dir).resolve()
    validate_local_structure(dataset_dir)
    # If data.yaml exists, use it; else require classes and write one
    existing_yaml = dataset_dir / "data.yaml"
    if existing_yaml.exists():
        return existing_yaml
    if not args.classes:
        print("No data.yaml found. Provide --classes to generate one.", file=sys.stderr)
        sys.exit(2)
    classes_path = Path(args.classes).resolve()
    classes = read_classes(classes_path)
    return write_data_yaml(dataset_dir, classes)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset from Roboflow or local directory.")
    p.add_argument("--source", choices=["roboflow", "local"], required=True, help="Dataset source type")
    # Roboflow args
    p.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    p.add_argument("--workspace", help="Roboflow workspace slug", default=None)
    p.add_argument("--project", help="Roboflow project slug", default=None)
    p.add_argument("--version", type=int, help="Roboflow version number", default=None)
    p.add_argument("--format", default="yolov8", help="Download format (default: yolov8)")
    # Local args
    p.add_argument("--dataset-dir", help="Local dataset root (with train/val/test)")
    p.add_argument("--classes", help="Path to classes file (.txt/.json/.yaml) for local dataset")
    return p.parse_args()


def main():
    args = parse_args()
    if args.source == "roboflow":
        required = [args.workspace, args.project, args.version]
        if any(x is None for x in required):
            print("--workspace, --project, and --version are required for Roboflow source.", file=sys.stderr)
            sys.exit(2)
        data_yaml = resolve_roboflow_dataset(args)
    else:
        if not args.dataset_dir:
            print("--dataset-dir is required for local source.", file=sys.stderr)
            sys.exit(2)
        data_yaml = resolve_local_dataset(args)

    print(str(data_yaml.resolve()))


if __name__ == "__main__":
    main()