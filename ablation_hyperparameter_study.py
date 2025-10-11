"""
Ablation Study: Training Hyperparameters
Tests different hyperparameter configurations to understand their impact on model performance
"""

import json
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch

# Define hyperparameter configurations for ablation study
ABLATION_CONFIGS = {
    "baseline": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "batch_size_8": {
        "batch": 8,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "batch_size_32": {
        "batch": 32,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "image_size_320": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 320,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "image_size_1280": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 1280,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "learning_rate_low": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.001,
        "optimizer": "SGD"
    },
    "learning_rate_high": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.1,
        "optimizer": "SGD"
    },
    "adam_optimizer": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "Adam"
    },
    "adamw_optimizer": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 20,
        "lr0": 0.01,
        "optimizer": "AdamW"
    },
    "early_patience_10": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 10,
        "lr0": 0.01,
        "optimizer": "SGD"
    },
    "early_patience_50": {
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "patience": 50,
        "lr0": 0.01,
        "optimizer": "SGD"
    }
}

def run_ablation_experiment(config_name, config, dataset_path, results_dir):
    """Run a single ablation experiment"""
    print(f"\n{'='*80}")
    print(f"Running ablation: {config_name}")
    print(f"Config: {config}")
    print(f"{'='*80}\n")
    
    # Create results directory
    exp_dir = results_dir / config_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Record start time
    start_time = time.time()
    
    try:
        # Train model
        results = model.train(
            data=dataset_path,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            patience=config['patience'],
            lr0=config['lr0'],
            optimizer=config['optimizer'],
            device=0,
            workers=4,
            amp=True,
            cache=True,
            save_period=5,
            project=str(exp_dir),
            name='training',
            exist_ok=True
        )
        
        # Record training time
        training_time = time.time() - start_time
        
        # Validate on unified test set
        unified_test_path = "/home/saumil/Desktop/360paper/data/datasets/quad_dataset/data.yaml"
        val_results = model.val(data=unified_test_path)
        
        # Extract metrics
        metrics = {
            "config_name": config_name,
            "config": config,
            "training_time": training_time,
            "training_time_formatted": f"{int(training_time//3600)}:{int((training_time%3600)//60):02d}:{int(training_time%60):02d}",
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "f1": float(2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr + 1e-6)),
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in {config_name}: {str(e)}")
        metrics = {
            "config_name": config_name,
            "config": config,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    # Save results
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    """Run all ablation studies"""
    # Test on quad dataset (best performing from previous experiments)
    dataset_path = "/home/saumil/Desktop/360paper/data/datasets/quad_dataset/data.yaml"
    results_dir = Path("/home/saumil/Desktop/360paper/data/results/ablation_studies")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Run each ablation experiment
    for config_name, config in ABLATION_CONFIGS.items():
        metrics = run_ablation_experiment(config_name, config, dataset_path, results_dir)
        all_results[config_name] = metrics
        
        # Save intermediate results
        with open(results_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Generate comparative analysis
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    
    successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
    
    if successful_results:
        # Sort by mAP50
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['mAP50'], reverse=True)
        
        print("\nRanking by mAP@0.5:")
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            print(f"{rank}. {name:30s} | mAP50: {metrics['mAP50']:.4f} | "
                  f"mAP50-95: {metrics['mAP50_95']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Time: {metrics['training_time_formatted']}")
        
        # Find best configuration
        best_config = sorted_results[0]
        baseline = successful_results.get('baseline', {})
        
        print("\n" + "="*80)
        print(f"BEST CONFIGURATION: {best_config[0]}")
        print("="*80)
        print(json.dumps(best_config[1], indent=2))
        
        if baseline:
            print("\n" + "="*80)
            print("COMPARISON TO BASELINE:")
            print("="*80)
            for metric in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']:
                improvement = ((best_config[1][metric] - baseline[metric]) / baseline[metric]) * 100
                print(f"{metric:15s}: {baseline[metric]:.4f} -> {best_config[1][metric]:.4f} "
                      f"({improvement:+.2f}%)")
    
    print("\n" + "="*80)
    print("Ablation study complete! Results saved to:", results_dir)
    print("="*80)

if __name__ == "__main__":
    main()
