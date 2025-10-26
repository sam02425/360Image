#!/usr/bin/env python3
import os
from pathlib import Path

# Find available datasets
base_dir = Path('/home/saumil/Desktop/360paper')
datasets_dir = base_dir / 'datasets'

print("Looking for datasets...")
available_configs = []

for item in datasets_dir.iterdir():
    if item.is_dir():
        # Check if it has the required structure
        required_dirs = ['train/images', 'val/images', 'test/images']
        if all((item / d).exists() for d in required_dirs):
            available_configs.append(item.name)
            print(f"Found dataset: {item.name}")

if available_configs:
    # Use the first available dataset
    config_to_test = available_configs[0]
    print(f"Testing with: {config_to_test}")
    
    # Run experiment with found dataset
    os.system(f"python train_all_yolo11_on_unified_eval.py --base-dir {base_dir} --num-runs 1 --epochs 3 --view-configs {config_to_test}")
else:
    print("No valid datasets found!")
    print("Expected structure:")
    print("  datasets/")
    print("    your_dataset_name/")
    print("      train/images/")
    print("      train/labels/") 
    print("      val/images/")
    print("      val/labels/")
    print("      test/images/")
    print("      test/labels/")
    print("      data.yaml")