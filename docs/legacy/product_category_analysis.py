"""
Product Category Analysis
Tests model performance across different product categories to identify strengths and weaknesses
"""

import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

def analyze_category_performance(model_path, data_yaml, output_dir):
    """Analyze model performance by product category"""
    
    print(f"\nAnalyzing category performance for model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=data_yaml)
    
    # Extract per-class metrics
    class_names = results.names
    per_class_ap = results.box.ap  # AP per class
    
    # Create category analysis
    category_metrics = {}
    
    for class_idx, class_name in class_names.items():
        # Extract AP50 and AP50-95 for this class
        if len(per_class_ap.shape) == 1:
            # 1D array - just AP50
            ap50 = float(per_class_ap[class_idx]) if len(per_class_ap) > class_idx else 0.0
            ap50_95 = ap50  # Use same value
        else:
            # 2D array - multiple IoU thresholds
            ap50 = float(per_class_ap[class_idx, 0]) if len(per_class_ap) > class_idx else 0.0
            ap50_95 = float(np.mean(per_class_ap[class_idx, :])) if len(per_class_ap) > class_idx else 0.0
        
        # Categorize products
        category = categorize_product(class_name)
        
        if category not in category_metrics:
            category_metrics[category] = {
                "products": [],
                "ap50_scores": [],
                "ap50_95_scores": []
            }
        
        category_metrics[category]["products"].append(class_name)
        category_metrics[category]["ap50_scores"].append(ap50)
        category_metrics[category]["ap50_95_scores"].append(ap50_95)
    
    # Calculate category statistics
    category_stats = {}
    for category, data in category_metrics.items():
        category_stats[category] = {
            "product_count": len(data["products"]),
            "products": data["products"],
            "mean_ap50": float(np.mean(data["ap50_scores"])),
            "std_ap50": float(np.std(data["ap50_scores"])),
            "min_ap50": float(np.min(data["ap50_scores"])),
            "max_ap50": float(np.max(data["ap50_scores"])),
            "mean_ap50_95": float(np.mean(data["ap50_95_scores"])),
            "std_ap50_95": float(np.std(data["ap50_95_scores"])),
            "min_ap50_95": float(np.min(data["ap50_95_scores"])),
            "max_ap50_95": float(np.max(data["ap50_95_scores"]))
        }
    
    # Save results
    output_path = Path(output_dir) / "category_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(category_stats, f, indent=2)
    
    return category_stats

def categorize_product(product_name):
    """Categorize products into broader groups"""
    product_lower = product_name.lower()
    
    # Define category keywords
    if any(x in product_lower for x in ['bourbon', 'whiskey', 'whisky', 'rye']):
        return "Whiskey/Bourbon"
    elif any(x in product_lower for x in ['vodka']):
        return "Vodka"
    elif any(x in product_lower for x in ['rum']):
        return "Rum"
    elif any(x in product_lower for x in ['tequila', 'mezcal']):
        return "Tequila/Mezcal"
    elif any(x in product_lower for x in ['gin']):
        return "Gin"
    elif any(x in product_lower for x in ['cognac', 'brandy']):
        return "Cognac/Brandy"
    elif any(x in product_lower for x in ['liqueur', 'cream', 'schnapps']):
        return "Liqueur/Cream"
    elif any(x in product_lower for x in ['blend', 'canadian']):
        return "Blended/Canadian"
    else:
        return "Other"

def compare_models_by_category():
    """Compare all models across product categories"""
    
    models = {
        "YOLOv8_dual": "/home/saumil/Desktop/360paper/data/models/dual_best.pt",
        "YOLOv8_quad": "/home/saumil/Desktop/360paper/data/models/quad_best.pt",
        "YOLOv8_octal": "/home/saumil/Desktop/360paper/data/models/octal_best.pt",
        "YOLOv8_full": "/home/saumil/Desktop/360paper/data/models/full_best.pt"
    }
    
    # Use quad dataset for validation (it has proper structure)
    data_yaml = "/home/saumil/Desktop/360paper/data/datasets/quad_dataset/data.yaml"
    output_dir = Path("/home/saumil/Desktop/360paper/data/results/category_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, model_path in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}")
        print(f"{'='*80}")
        
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            category_stats = analyze_category_performance(model_path, data_yaml, model_output_dir)
            all_results[model_name] = category_stats
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            all_results[model_name] = {"error": str(e)}
    
    # Save combined results
    with open(output_dir / "all_models_category_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("CATEGORY PERFORMANCE SUMMARY")
    print("="*80)
    
    for model_name, results in all_results.items():
        if "error" not in results:
            print(f"\n{model_name}:")
            sorted_categories = sorted(results.items(), key=lambda x: x[1]['mean_ap50'], reverse=True)
            for category, stats in sorted_categories:
                print(f"  {category:25s} | Products: {stats['product_count']:3d} | "
                      f"mAP50: {stats['mean_ap50']:.4f} ± {stats['std_ap50']:.4f} | "
                      f"mAP50-95: {stats['mean_ap50_95']:.4f} ± {stats['std_ap50_95']:.4f}")
    
    print("\n" + "="*80)
    print("Category analysis complete! Results saved to:", output_dir)
    print("="*80)

if __name__ == "__main__":
    compare_models_by_category()
