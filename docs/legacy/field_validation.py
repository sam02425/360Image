"""
Field Validation Framework
Simulates and validates model performance in actual retail environment conditions
"""

import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from collections import defaultdict

class RetailEnvironmentSimulator:
    """Simulates various retail environment conditions"""
    
    @staticmethod
    def add_lighting_variation(image, factor=0.5):
        """Simulate varying lighting conditions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def add_motion_blur(image, kernel_size=15):
        """Simulate motion blur from camera movement"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def add_occlusion(image, occlusion_ratio=0.2):
        """Simulate partial occlusion by other objects"""
        h, w = image.shape[:2]
        mask = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Random rectangular occlusions
        num_occlusions = np.random.randint(1, 4)
        for _ in range(num_occlusions):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = min(x1 + int(w * occlusion_ratio), w)
            y2 = min(y1 + int(h * occlusion_ratio), h)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return cv2.bitwise_and(image, mask)
    
    @staticmethod
    def add_perspective_distortion(image, strength=0.1):
        """Simulate perspective distortion from different viewing angles"""
        h, w = image.shape[:2]
        
        # Define source and destination points
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        offset = int(min(h, w) * strength)
        dst_points = np.float32([
            [np.random.randint(0, offset), np.random.randint(0, offset)],
            [w - np.random.randint(0, offset), np.random.randint(0, offset)],
            [np.random.randint(0, offset), h - np.random.randint(0, offset)],
            [w - np.random.randint(0, offset), h - np.random.randint(0, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h))
    
    @staticmethod
    def add_noise(image, noise_level=25):
        """Add gaussian noise to simulate poor camera quality"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

def validate_in_retail_conditions(model_path, test_images_dir, output_dir):
    """Validate model under various retail conditions"""
    
    print(f"\nValidating model: {model_path}")
    print(f"Test images: {test_images_dir}")
    
    model = YOLO(model_path)
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test conditions
    conditions = {
        "baseline": lambda img: img,
        "low_light": lambda img: RetailEnvironmentSimulator.add_lighting_variation(img, 0.5),
        "bright_light": lambda img: RetailEnvironmentSimulator.add_lighting_variation(img, 1.5),
        "motion_blur": lambda img: RetailEnvironmentSimulator.add_motion_blur(img),
        "partial_occlusion": lambda img: RetailEnvironmentSimulator.add_occlusion(img),
        "perspective_distortion": lambda img: RetailEnvironmentSimulator.add_perspective_distortion(img),
        "camera_noise": lambda img: RetailEnvironmentSimulator.add_noise(img)
    }
    
    # Collect test images
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    image_files = image_files[:100]  # Limit to 100 images for validation
    
    results_by_condition = {}
    
    for condition_name, transform_fn in conditions.items():
        print(f"\nTesting condition: {condition_name}")
        
        condition_results = {
            "detections": [],
            "inference_times": [],
            "confidences": []
        }
        
        for img_path in image_files:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Apply transformation
            transformed = transform_fn(image.copy())
            
            # Run inference
            start_time = time.time()
            predictions = model(transformed, verbose=False)
            inference_time = time.time() - start_time
            
            # Extract results
            if len(predictions) > 0 and predictions[0].boxes is not None:
                boxes = predictions[0].boxes
                num_detections = len(boxes)
                confidences = boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else []
                
                condition_results["detections"].append(num_detections)
                condition_results["confidences"].extend(confidences)
            else:
                condition_results["detections"].append(0)
            
            condition_results["inference_times"].append(inference_time)
        
        # Calculate statistics
        results_by_condition[condition_name] = {
            "mean_detections": float(np.mean(condition_results["detections"])),
            "std_detections": float(np.std(condition_results["detections"])),
            "mean_confidence": float(np.mean(condition_results["confidences"])) if condition_results["confidences"] else 0.0,
            "mean_inference_time": float(np.mean(condition_results["inference_times"])),
            "std_inference_time": float(np.std(condition_results["inference_times"])),
            "total_images_tested": len(image_files)
        }
        
        print(f"  Mean detections: {results_by_condition[condition_name]['mean_detections']:.2f}")
        print(f"  Mean confidence: {results_by_condition[condition_name]['mean_confidence']:.4f}")
        print(f"  Mean inference time: {results_by_condition[condition_name]['mean_inference_time']:.4f}s")
    
    return results_by_condition

def comprehensive_field_validation():
    """Run comprehensive field validation for all models"""
    
    print("\n" + "="*80)
    print("FIELD VALIDATION IN RETAIL ENVIRONMENT")
    print("="*80)
    
    models = {
        "YOLOv8_dual": "/home/saumil/Desktop/360paper/data/models/dual_best.pt",
        "YOLOv8_quad": "/home/saumil/Desktop/360paper/data/models/quad_best.pt",
        "YOLOv8_octal": "/home/saumil/Desktop/360paper/data/models/octal_best.pt",
        "YOLOv8_full": "/home/saumil/Desktop/360paper/data/models/full_best.pt"
    }
    
    # Use quad dataset test images
    test_images_dir = "/home/saumil/Desktop/360paper/data/datasets/quad_dataset/test/images"
    output_dir = Path("/home/saumil/Desktop/360paper/data/results/field_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, model_path in models.items():
        if Path(model_path).exists():
            print(f"\n{'='*80}")
            print(f"Validating {model_name}")
            print(f"{'='*80}")
            
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                results = validate_in_retail_conditions(model_path, test_images_dir, model_output_dir)
                all_results[model_name] = results
                
                # Save individual model results
                with open(model_output_dir / "validation_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"Error validating {model_name}: {str(e)}")
                all_results[model_name] = {"error": str(e)}
    
    # Save combined results
    with open(output_dir / "all_models_field_validation.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    print("\n" + "="*80)
    print("FIELD VALIDATION SUMMARY")
    print("="*80)
    
    for model_name, results in all_results.items():
        if "error" not in results:
            print(f"\n{model_name}:")
            
            baseline = results.get("baseline", {})
            print(f"  Baseline: {baseline.get('mean_detections', 0):.2f} detections, "
                  f"{baseline.get('mean_confidence', 0):.4f} confidence, "
                  f"{baseline.get('mean_inference_time', 0):.4f}s inference")
            
            # Compare other conditions to baseline
            for condition, stats in results.items():
                if condition != "baseline" and baseline.get('mean_detections', 0) > 0:
                    det_change = ((stats['mean_detections'] - baseline.get('mean_detections', 0)) / 
                                 (baseline.get('mean_detections', 1))) * 100
                    conf_change = ((stats['mean_confidence'] - baseline.get('mean_confidence', 0)) / 
                                  max(baseline.get('mean_confidence', 0.0001), 0.0001)) * 100
                    
                    print(f"  {condition:25s}: {stats['mean_detections']:.2f} detections ({det_change:+.1f}%), "
                          f"{stats['mean_confidence']:.4f} conf ({conf_change:+.1f}%)")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*80)
    
    recommendations = [
        "1. Ensure adequate lighting in retail environment (avoid extreme low/high light)",
        "2. Use stable camera mounting to minimize motion blur",
        "3. Position cameras to minimize occlusions from other products",
        "4. Consider calibration for specific viewing angles in deployment",
        "5. Implement quality checks for noisy camera sensors",
        "6. Best performing model under adverse conditions should be prioritized for deployment"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "="*80)
    print("Field validation complete! Results saved to:", output_dir)
    print("="*80)

if __name__ == "__main__":
    comprehensive_field_validation()
