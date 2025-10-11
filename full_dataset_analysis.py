"""
Full Dataset Performance Analysis
Investigates why full coverage underperforms for YOLOv8
"""

import json
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_full_dataset_underperformance():
    """Comprehensive analysis of full dataset performance issues"""
    
    print("\n" + "="*80)
    print("ANALYZING FULL DATASET UNDERPERFORMANCE")
    print("="*80)
    
    results_dir = Path("/home/saumil/Desktop/360paper/data/results/full_dataset_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    models = {
        "dual": YOLO("/home/saumil/Desktop/360paper/data/models/dual_best.pt"),
        "quad": YOLO("/home/saumil/Desktop/360paper/data/models/quad_best.pt"),
        "octal": YOLO("/home/saumil/Desktop/360paper/data/models/octal_best.pt"),
        "full": YOLO("/home/saumil/Desktop/360paper/data/models/full_best.pt")
    }
    
    # Use quad dataset for validation (it has proper structure)
    data_yaml = "/home/saumil/Desktop/360paper/data/datasets/quad_dataset/data.yaml"
    
    analysis_results = {}
    
    # 1. Dataset statistics comparison
    print("\n1. DATASET STATISTICS COMPARISON")
    print("-" * 80)
    
    dataset_paths = {
        "dual": "/home/saumil/Desktop/360paper/data/datasets/dual_dataset",
        "quad": "/home/saumil/Desktop/360paper/data/datasets/quad_dataset",
        "octal": "/home/saumil/Desktop/360paper/data/datasets/octal_dataset",
        "full": "/home/saumil/Desktop/360paper/data/datasets/full_dataset"
    }
    
    dataset_stats = {}
    for name, path in dataset_paths.items():
        stats_file = Path(path) / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                dataset_stats[name] = json.load(f)
                print(f"\n{name.upper()}:")
                print(f"  Train images: {dataset_stats[name].get('train_images', 'N/A')}")
                print(f"  Val images: {dataset_stats[name].get('val_images', 'N/A')}")
                print(f"  Test images: {dataset_stats[name].get('test_images', 'N/A')}")
                print(f"  Total classes: {dataset_stats[name].get('num_classes', 'N/A')}")
    
    analysis_results['dataset_statistics'] = dataset_stats
    
    # 2. Per-class performance analysis
    print("\n2. PER-CLASS PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    per_class_performance = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        results = model.val(data=data_yaml)
        
        class_names = results.names
        per_class_ap = results.box.ap
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_idx, class_name in class_names.items():
            if len(per_class_ap) > class_idx:
                # Handle both 1D and 2D per_class_ap arrays
                if len(per_class_ap.shape) == 1:
                    # 1D array - just AP50
                    ap50 = float(per_class_ap[class_idx])
                    ap50_95 = ap50  # Use same value
                else:
                    # 2D array - multiple IoU thresholds
                    ap50 = float(per_class_ap[class_idx, 0])
                    ap50_95 = float(np.mean(per_class_ap[class_idx, :]))
                class_metrics[class_name] = {
                    "ap50": ap50,
                    "ap50_95": ap50_95
                }
        
        per_class_performance[model_name] = class_metrics
        
        # Find worst performing classes for this model
        sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['ap50'])
        print(f"\nWorst 10 classes for {model_name}:")
        for i, (class_name, metrics) in enumerate(sorted_classes[:10], 1):
            print(f"  {i}. {class_name:40s} AP50: {metrics['ap50']:.4f}")
    
    analysis_results['per_class_performance'] = per_class_performance
    
    # 3. Compare full vs other datasets
    print("\n3. FULL DATASET VS OTHERS COMPARISON")
    print("-" * 80)
    
    comparison = {}
    full_classes = set(per_class_performance['full'].keys())
    
    for other_model in ['dual', 'quad', 'octal']:
        other_classes = set(per_class_performance[other_model].keys())
        
        # Classes in full but not in other
        unique_to_full = full_classes - other_classes
        
        # Classes in both
        common_classes = full_classes & other_classes
        
        # Performance on common classes
        full_common_ap50 = [per_class_performance['full'][c]['ap50'] for c in common_classes if c in per_class_performance['full']]
        other_common_ap50 = [per_class_performance[other_model][c]['ap50'] for c in common_classes if c in per_class_performance[other_model]]
        
        comparison[other_model] = {
            "unique_to_full_count": len(unique_to_full),
            "common_classes_count": len(common_classes),
            "full_mean_ap50_on_common": float(np.mean(full_common_ap50)) if full_common_ap50 else 0.0,
            "other_mean_ap50_on_common": float(np.mean(other_common_ap50)) if other_common_ap50 else 0.0,
            "performance_gap": float(np.mean(other_common_ap50) - np.mean(full_common_ap50)) if full_common_ap50 and other_common_ap50 else 0.0
        }
        
        print(f"\nFull vs {other_model.upper()}:")
        print(f"  Common classes: {len(common_classes)}")
        print(f"  Unique to full: {len(unique_to_full)}")
        print(f"  Full mean AP50 (common): {comparison[other_model]['full_mean_ap50_on_common']:.4f}")
        print(f"  {other_model.capitalize()} mean AP50 (common): {comparison[other_model]['other_mean_ap50_on_common']:.4f}")
        print(f"  Performance gap: {comparison[other_model]['performance_gap']:.4f}")
    
    analysis_results['dataset_comparison'] = comparison
    
    # 4. Hypothesis testing
    print("\n4. HYPOTHESIS TESTING")
    print("-" * 80)
    
    hypotheses = {
        "class_imbalance": "Full dataset may have severe class imbalance",
        "data_quality": "Full dataset may have lower quality annotations or images",
        "overfitting": "Full model may be overfitting due to too much diverse data",
        "camera_angles": "Full coverage may include too many difficult angles",
        "training_convergence": "Full model may need different hyperparameters"
    }
    
    # Calculate class distribution variance
    if 'full' in dataset_stats:
        class_counts = dataset_stats['full'].get('class_distribution', {})
        if class_counts:
            counts = list(class_counts.values())
            class_variance = float(np.var(counts))
            class_imbalance_ratio = max(counts) / (min(counts) + 1e-6)
            
            print(f"\nClass Imbalance Analysis (Full Dataset):")
            print(f"  Variance: {class_variance:.2f}")
            print(f"  Max/Min ratio: {class_imbalance_ratio:.2f}")
            
            analysis_results['class_imbalance'] = {
                "variance": class_variance,
                "max_min_ratio": float(class_imbalance_ratio)
            }
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    # Check class imbalance
    if analysis_results.get('class_imbalance', {}).get('max_min_ratio', 0) > 10:
        recommendations.append({
            "issue": "Severe class imbalance detected",
            "recommendation": "Apply class-weighted loss or data augmentation for underrepresented classes"
        })
    
    # Check performance gap
    avg_gap = np.mean([v['performance_gap'] for v in comparison.values()])
    if avg_gap > 0.1:
        recommendations.append({
            "issue": "Significant performance gap compared to other datasets",
            "recommendation": "Consider filtering low-quality images or using a different training strategy"
        })
    
    # General recommendations
    recommendations.extend([
        {
            "issue": "Full dataset complexity",
            "recommendation": "Try longer training (more epochs) with lower learning rate"
        },
        {
            "issue": "Potential overfitting",
            "recommendation": "Increase regularization (dropout, weight decay) or use data augmentation"
        },
        {
            "issue": "Camera angle diversity",
            "recommendation": "Consider training separate models for different viewing angles, then ensemble"
        }
    ])
    
    analysis_results['recommendations'] = recommendations
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']}")
        print(f"   → {rec['recommendation']}")
    
    # Save complete analysis
    with open(results_dir / "full_dataset_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to:", results_dir)
    print("="*80)
    
    return analysis_results

if __name__ == "__main__":
    analyze_full_dataset_underperformance()
