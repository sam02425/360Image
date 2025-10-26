#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Selection Utility for Retail Detection
--------------------------------------------
This utility helps users compare and select the best model for their retail detection needs.
It provides benchmarking, visualization, and recommendation features.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import mlflow
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import precision_recall_curve, average_precision_score
import yaml
from tabulate import tabulate

# Import local modules
try:
    from ensemble_models import ModelEnsemble
    from alternative_models import RetailFasterRCNN, RetailRetinaNet, RetailCenterNet
except ImportError:
    print("Warning: Some model modules could not be imported.")


class ModelBenchmark:
    """Benchmark different models on retail detection datasets."""
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "./benchmark_results",
                 device: str = None):
        """
        Initialize the benchmark utility.
        
        Args:
            dataset_path: Path to the validation/test dataset
            output_dir: Directory to save benchmark results
            device: Device to run benchmarks on ('cuda', 'cpu', etc.)
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")
        
    def load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """
        Load a model from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint
            model_type: Type of model to load
            
        Returns:
            Loaded model
        """
        if model_type == "efficientformer_yolo":
            # Import dynamically to avoid circular imports
            from train_efficientformer_yolo import HybridYOLOWithEfficientFormer, TrainingConfig
            
            # Load config if available
            config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config = TrainingConfig(**config_dict)
            else:
                # Default config
                config = TrainingConfig()
            
            model = HybridYOLOWithEfficientFormer(config)
            
        elif model_type == "faster_rcnn":
            model = RetailFasterRCNN(num_classes=config.num_classes)
            
        elif model_type == "retinanet":
            model = RetailRetinaNet(num_classes=config.num_classes)
            
        elif model_type == "centernet":
            model = RetailCenterNet(num_classes=config.num_classes)
            
        elif model_type == "ensemble":
            # Load ensemble configuration
            with open(os.path.join(os.path.dirname(model_path), "ensemble_config.json"), 'r') as f:
                ensemble_config = json.load(f)
            
            model = ModelEnsemble(ensemble_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_dataset(self, subset: str = "val"):
        """
        Load the dataset for benchmarking.
        
        Args:
            subset: Dataset subset to use ('val' or 'test')
        
        Returns:
            DataLoader for the dataset
        """
        # Import dynamically to avoid circular imports
        from train_efficientformer_yolo import RetailDatasetWithAugmentation
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = RetailDatasetWithAugmentation(
            root_dir=self.dataset_path,
            split=subset,
            transform=None,  # No augmentation for evaluation
            use_cache=True,
            use_dvc=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        
        return dataloader
    
    def benchmark_model(self, 
                        model: torch.nn.Module, 
                        dataloader: torch.utils.data.DataLoader,
                        model_name: str) -> Dict[str, Any]:
        """
        Benchmark a model on the dataset.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader for the dataset
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            "model_name": model_name,
            "inference_times": [],
            "ap_scores": [],
            "precision": [],
            "recall": [],
            "f1_scores": [],
            "memory_usage": 0
        }
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"benchmark_{model_name}"):
            # Log model architecture
            mlflow.log_param("model_type", model_name)
            
            # Track memory usage
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            all_predictions = []
            all_targets = []
            
            # Run inference
            with torch.no_grad():
                for batch in dataloader:
                    images, targets = batch
                    images = [img.to(self.device) for img in images]
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(images)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    results["inference_times"].append(end_time - start_time)
                    
                    # Convert outputs and targets to a common format for evaluation
                    # This will depend on the model output format
                    pred_boxes, pred_scores, pred_labels = self._extract_predictions(outputs)
                    target_boxes, target_labels = self._extract_targets(targets)
                    
                    all_predictions.append((pred_boxes, pred_scores, pred_labels))
                    all_targets.append((target_boxes, target_labels))
            
            # Calculate metrics
            ap, precision, recall, f1 = self._calculate_metrics(all_predictions, all_targets)
            
            results["ap_scores"] = ap
            results["precision"] = precision
            results["recall"] = recall
            results["f1_scores"] = f1
            
            # Calculate memory usage
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            results["memory_usage"] = (end_mem - start_mem) / 1024 / 1024  # MB
            
            # Calculate average inference time
            results["avg_inference_time"] = np.mean(results["inference_times"])
            results["fps"] = 1.0 / results["avg_inference_time"]
            
            # Log metrics to MLflow
            mlflow.log_metric("mAP", np.mean(ap))
            mlflow.log_metric("avg_precision", np.mean(precision))
            mlflow.log_metric("avg_recall", np.mean(recall))
            mlflow.log_metric("avg_f1", np.mean(f1))
            mlflow.log_metric("avg_inference_time", results["avg_inference_time"])
            mlflow.log_metric("fps", results["fps"])
            mlflow.log_metric("memory_usage_mb", results["memory_usage"])
            
            # Log model
            mlflow.pytorch.log_model(model, f"{model_name}_model")
            
        return results
    
    def _extract_predictions(self, outputs) -> Tuple[List, List, List]:
        """
        Extract predictions from model outputs.
        This method should be adapted based on the model output format.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Tuple of (boxes, scores, labels)
        """
        # Implementation depends on model output format
        # This is a placeholder
        pred_boxes, pred_scores, pred_labels = [], [], []
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # YOLO-style outputs
            if 'boxes' in outputs:
                pred_boxes = outputs['boxes']
                pred_scores = outputs['scores']
                pred_labels = outputs['labels']
            # Faster R-CNN style outputs
            elif 'pred_boxes' in outputs:
                pred_boxes = outputs['pred_boxes']
                pred_scores = outputs['pred_scores']
                pred_labels = outputs['pred_classes']
        elif isinstance(outputs, list):
            # Some models return a list of dictionaries (one per image)
            for output in outputs:
                if isinstance(output, dict):
                    pred_boxes.append(output.get('boxes', []))
                    pred_scores.append(output.get('scores', []))
                    pred_labels.append(output.get('labels', []))
        
        return pred_boxes, pred_scores, pred_labels
    
    def _extract_targets(self, targets) -> Tuple[List, List]:
        """
        Extract ground truth from targets.
        
        Args:
            targets: Ground truth targets
            
        Returns:
            Tuple of (boxes, labels)
        """
        # Implementation depends on dataset format
        # This is a placeholder
        target_boxes, target_labels = [], []
        
        for target in targets:
            if isinstance(target, dict):
                target_boxes.append(target.get('boxes', []))
                target_labels.append(target.get('labels', []))
            elif isinstance(target, tuple) and len(target) >= 2:
                target_boxes.append(target[0])
                target_labels.append(target[1])
        
        return target_boxes, target_labels
    
    def _calculate_metrics(self, predictions, targets) -> Tuple[List, List, List, List]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Tuple of (ap_scores, precision, recall, f1_scores)
        """
        # This is a simplified implementation
        # In practice, you would use a proper evaluation library
        
        ap_scores = []
        precision_list = []
        recall_list = []
        f1_scores = []
        
        for (pred_boxes, pred_scores, pred_labels), (target_boxes, target_labels) in zip(predictions, targets):
            # Calculate IoU between predictions and targets
            ious = self._calculate_iou(pred_boxes, target_boxes)
            
            # Calculate AP
            ap = self._calculate_ap(ious, pred_scores, pred_labels, target_labels)
            ap_scores.append(ap)
            
            # Calculate precision and recall
            precision, recall = self._calculate_precision_recall(ious, pred_scores, pred_labels, target_labels)
            precision_list.append(precision)
            recall_list.append(recall)
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        return ap_scores, precision_list, recall_list, f1_scores
    
    def _calculate_iou(self, pred_boxes, target_boxes) -> np.ndarray:
        """
        Calculate IoU between prediction and target boxes.
        
        Args:
            pred_boxes: Predicted bounding boxes
            target_boxes: Target bounding boxes
            
        Returns:
            IoU matrix
        """
        # Convert to numpy arrays if they're not already
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
            
        # Ensure boxes are in [x1, y1, x2, y2] format
        # This is a placeholder implementation
        ious = np.zeros((len(pred_boxes), len(target_boxes)))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, target_box in enumerate(target_boxes):
                # Calculate intersection area
                x1 = max(pred_box[0], target_box[0])
                y1 = max(pred_box[1], target_box[1])
                x2 = min(pred_box[2], target_box[2])
                y2 = min(pred_box[3], target_box[3])
                
                if x2 < x1 or y2 < y1:
                    ious[i, j] = 0
                    continue
                
                intersection = (x2 - x1) * (y2 - y1)
                
                # Calculate union area
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                union = pred_area + target_area - intersection
                
                ious[i, j] = intersection / union if union > 0 else 0
        
        return ious
    
    def _calculate_ap(self, ious, pred_scores, pred_labels, target_labels) -> float:
        """
        Calculate Average Precision.
        
        Args:
            ious: IoU matrix
            pred_scores: Prediction confidence scores
            pred_labels: Predicted class labels
            target_labels: Target class labels
            
        Returns:
            Average Precision score
        """
        # This is a simplified implementation
        # In practice, you would use a proper evaluation library
        
        # Convert to numpy arrays if they're not already
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()
            
        # Calculate AP for each class
        unique_classes = np.unique(np.concatenate([pred_labels, target_labels]))
        ap_per_class = []
        
        for cls in unique_classes:
            # Get predictions and targets for this class
            cls_pred_indices = np.where(pred_labels == cls)[0]
            cls_target_indices = np.where(target_labels == cls)[0]
            
            if len(cls_pred_indices) == 0 or len(cls_target_indices) == 0:
                continue
            
            # Get scores and IoUs for this class
            cls_scores = pred_scores[cls_pred_indices]
            cls_ious = ious[cls_pred_indices][:, cls_target_indices]
            
            # A prediction is considered correct if IoU > 0.5 with any target of the same class
            correct = np.max(cls_ious, axis=1) > 0.5
            
            # Calculate precision and recall
            precision, recall, _ = precision_recall_curve(correct, cls_scores)
            
            # Calculate AP as area under PR curve
            ap = average_precision_score(correct, cls_scores)
            ap_per_class.append(ap)
        
        # Return mean AP across classes
        return np.mean(ap_per_class) if len(ap_per_class) > 0 else 0
    
    def _calculate_precision_recall(self, ious, pred_scores, pred_labels, target_labels) -> Tuple[float, float]:
        """
        Calculate precision and recall.
        
        Args:
            ious: IoU matrix
            pred_scores: Prediction confidence scores
            pred_labels: Predicted class labels
            target_labels: Target class labels
            
        Returns:
            Tuple of (precision, recall)
        """
        # This is a simplified implementation
        # In practice, you would use a proper evaluation library
        
        # Convert to numpy arrays if they're not already
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()
            
        # A prediction is considered correct if IoU > 0.5 with any target of the same class
        correct = np.zeros(len(pred_scores), dtype=bool)
        
        for i, (pred_score, pred_label) in enumerate(zip(pred_scores, pred_labels)):
            for j, target_label in enumerate(target_labels):
                if pred_label == target_label and ious[i, j] > 0.5:
                    correct[i] = True
                    break
        
        # Calculate precision and recall
        if len(correct) > 0:
            precision = np.sum(correct) / len(correct)
        else:
            precision = 0
            
        if len(target_labels) > 0:
            recall = np.sum(correct) / len(target_labels)
        else:
            recall = 0
            
        return precision, recall
    
    def run_benchmarks(self, models_config: List[Dict]) -> Dict[str, Dict]:
        """
        Run benchmarks on multiple models.
        
        Args:
            models_config: List of model configurations to benchmark
                Each config should have 'path', 'type', and 'name' keys
                
        Returns:
            Dictionary of benchmark results
        """
        # Load dataset
        dataloader = self.load_dataset(subset="val")
        
        # Run benchmarks for each model
        for model_config in models_config:
            model_path = model_config['path']
            model_type = model_config['type']
            model_name = model_config['name']
            
            print(f"Benchmarking {model_name}...")
            
            # Load model
            model = self.load_model(model_path, model_type)
            
            # Run benchmark
            results = self.benchmark_model(model, dataloader, model_name)
            
            # Store results
            self.results[model_name] = results
            
            # Save results
            self._save_results(model_name, results)
            
        return self.results
    
    def _save_results(self, model_name: str, results: Dict):
        """
        Save benchmark results to disk.
        
        Args:
            model_name: Name of the model
            results: Benchmark results
        """
        # Create model directory
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save results as JSON
        with open(os.path.join(model_dir, "results.json"), 'w') as f:
            # Convert numpy arrays to lists
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                    serializable_results[k] = [arr.tolist() for arr in v]
                else:
                    serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=2)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare benchmark results across models.
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        # Create comparison DataFrame
        comparison = {
            "Model": [],
            "mAP": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "Inference Time (ms)": [],
            "FPS": [],
            "Memory Usage (MB)": []
        }
        
        for model_name, results in self.results.items():
            comparison["Model"].append(model_name)
            comparison["mAP"].append(np.mean(results["ap_scores"]))
            comparison["Precision"].append(np.mean(results["precision"]))
            comparison["Recall"].append(np.mean(results["recall"]))
            comparison["F1 Score"].append(np.mean(results["f1_scores"]))
            comparison["Inference Time (ms)"].append(results["avg_inference_time"] * 1000)
            comparison["FPS"].append(results["fps"])
            comparison["Memory Usage (MB)"].append(results["memory_usage"])
        
        # Create DataFrame
        df = pd.DataFrame(comparison)
        
        # Save comparison
        df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
        
        return df
    
    def visualize_comparison(self, metrics: List[str] = None):
        """
        Visualize model comparison.
        
        Args:
            metrics: List of metrics to visualize
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        # Get comparison DataFrame
        df = self.compare_models()
        
        # Default metrics to visualize
        if metrics is None:
            metrics = ["mAP", "Precision", "Recall", "F1 Score", "FPS"]
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            df.plot(x="Model", y=metric, kind="bar", ax=ax)
            ax.set_title(f"Model Comparison - {metric}")
            ax.set_ylabel(metric)
            ax.grid(axis="y")
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        plt.close()
    
    def recommend_model(self, 
                        priority: str = "balanced",
                        hardware_constraints: Dict = None) -> str:
        """
        Recommend the best model based on benchmark results.
        
        Args:
            priority: Priority for recommendation ('accuracy', 'speed', 'memory', 'balanced')
            hardware_constraints: Hardware constraints to consider
                e.g. {'max_memory_mb': 4000, 'min_fps': 10}
                
        Returns:
            Name of the recommended model
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        # Get comparison DataFrame
        df = self.compare_models()
        
        # Apply hardware constraints
        if hardware_constraints:
            if 'max_memory_mb' in hardware_constraints:
                df = df[df["Memory Usage (MB)"] <= hardware_constraints['max_memory_mb']]
            
            if 'min_fps' in hardware_constraints:
                df = df[df["FPS"] >= hardware_constraints['min_fps']]
            
            if df.empty:
                return "No model meets the hardware constraints."
        
        # Normalize metrics
        normalized_df = df.copy()
        
        for metric in ["mAP", "Precision", "Recall", "F1 Score", "FPS"]:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                
                if max_val > min_val:
                    normalized_df[f"norm_{metric}"] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[f"norm_{metric}"] = 1.0
        
        # Calculate score based on priority
        if priority == "accuracy":
            normalized_df["score"] = (
                normalized_df["norm_mAP"] * 0.4 +
                normalized_df["norm_F1 Score"] * 0.4 +
                normalized_df["norm_FPS"] * 0.2
            )
        elif priority == "speed":
            normalized_df["score"] = (
                normalized_df["norm_FPS"] * 0.6 +
                normalized_df["norm_mAP"] * 0.2 +
                normalized_df["norm_F1 Score"] * 0.2
            )
        elif priority == "memory":
            # Invert memory usage (lower is better)
            if "Memory Usage (MB)" in df.columns:
                min_mem = df["Memory Usage (MB)"].min()
                max_mem = df["Memory Usage (MB)"].max()
                
                if max_mem > min_mem:
                    normalized_df["norm_memory"] = 1 - (df["Memory Usage (MB)"] - min_mem) / (max_mem - min_mem)
                else:
                    normalized_df["norm_memory"] = 1.0
                
                normalized_df["score"] = (
                    normalized_df["norm_memory"] * 0.5 +
                    normalized_df["norm_mAP"] * 0.3 +
                    normalized_df["norm_FPS"] * 0.2
                )
            else:
                normalized_df["score"] = (
                    normalized_df["norm_mAP"] * 0.5 +
                    normalized_df["norm_FPS"] * 0.5
                )
        else:  # balanced
            normalized_df["score"] = (
                normalized_df["norm_mAP"] * 0.3 +
                normalized_df["norm_F1 Score"] * 0.3 +
                normalized_df["norm_FPS"] * 0.3
            )
            
            # Add memory score if available
            if "Memory Usage (MB)" in df.columns:
                min_mem = df["Memory Usage (MB)"].min()
                max_mem = df["Memory Usage (MB)"].max()
                
                if max_mem > min_mem:
                    normalized_df["norm_memory"] = 1 - (df["Memory Usage (MB)"] - min_mem) / (max_mem - min_mem)
                else:
                    normalized_df["norm_memory"] = 1.0
                
                normalized_df["score"] = normalized_df["score"] * 0.9 + normalized_df["norm_memory"] * 0.1
        
        # Get best model
        best_model = normalized_df.loc[normalized_df["score"].idxmax(), "Model"]
        
        return best_model


class ModelSelector:
    """Interactive model selector for retail detection."""
    
    def __init__(self, 
                 models_dir: str = "./models",
                 benchmark_results_dir: str = "./benchmark_results"):
        """
        Initialize the model selector.
        
        Args:
            models_dir: Directory containing model checkpoints
            benchmark_results_dir: Directory containing benchmark results
        """
        self.models_dir = models_dir
        self.benchmark_results_dir = benchmark_results_dir
        
        # Load available models
        self.available_models = self._load_available_models()
        
        # Load benchmark results if available
        self.benchmark_results = self._load_benchmark_results()
    
    def _load_available_models(self) -> List[Dict]:
        """
        Load available models from the models directory.
        
        Returns:
            List of available model configurations
        """
        available_models = []
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            print(f"Models directory {self.models_dir} does not exist.")
            return available_models
        
        # Find model checkpoints
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith(".pt") or file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    
                    # Try to determine model type
                    model_type = "unknown"
                    model_name = os.path.splitext(file)[0]
                    
                    # Check for config file
                    config_path = os.path.join(root, "config.yaml")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                            if 'model_type' in config:
                                model_type = config['model_type']
                    
                    # Infer model type from directory or file name
                    if "efficientformer" in root.lower() or "efficientformer" in file.lower():
                        model_type = "efficientformer_yolo"
                    elif "faster_rcnn" in root.lower() or "faster_rcnn" in file.lower():
                        model_type = "faster_rcnn"
                    elif "retinanet" in root.lower() or "retinanet" in file.lower():
                        model_type = "retinanet"
                    elif "centernet" in root.lower() or "centernet" in file.lower():
                        model_type = "centernet"
                    elif "ensemble" in root.lower() or "ensemble" in file.lower():
                        model_type = "ensemble"
                    
                    available_models.append({
                        "path": model_path,
                        "type": model_type,
                        "name": model_name
                    })
        
        return available_models
    
    def _load_benchmark_results(self) -> Dict:
        """
        Load benchmark results from the benchmark results directory.
        
        Returns:
            Dictionary of benchmark results
        """
        benchmark_results = {}
        
        # Check if benchmark results directory exists
        if not os.path.exists(self.benchmark_results_dir):
            print(f"Benchmark results directory {self.benchmark_results_dir} does not exist.")
            return benchmark_results
        
        # Find benchmark results
        for model_dir in os.listdir(self.benchmark_results_dir):
            results_path = os.path.join(self.benchmark_results_dir, model_dir, "results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    benchmark_results[model_dir] = json.load(f)
        
        return benchmark_results
    
    def list_available_models(self) -> pd.DataFrame:
        """
        List available models.
        
        Returns:
            DataFrame of available models
        """
        if not self.available_models:
            print("No models available.")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(self.available_models)
        
        # Add benchmark results if available
        if self.benchmark_results:
            df["mAP"] = df["name"].apply(
                lambda name: np.mean(self.benchmark_results.get(name, {}).get("ap_scores", [0]))
            )
            df["FPS"] = df["name"].apply(
                lambda name: self.benchmark_results.get(name, {}).get("fps", 0)
            )
        
        return df
    
    def run_benchmarks(self, 
                       dataset_path: str,
                       models: List[str] = None,
                       device: str = None):
        """
        Run benchmarks on selected models.
        
        Args:
            dataset_path: Path to the validation/test dataset
            models: List of model names to benchmark (None for all)
            device: Device to run benchmarks on ('cuda', 'cpu', etc.)
        """
        # Create benchmark utility
        benchmark = ModelBenchmark(
            dataset_path=dataset_path,
            output_dir=self.benchmark_results_dir,
            device=device
        )
        
        # Filter models if specified
        if models is not None:
            models_to_benchmark = [
                model for model in self.available_models
                if model["name"] in models
            ]
        else:
            models_to_benchmark = self.available_models
        
        # Run benchmarks
        results = benchmark.run_benchmarks(models_to_benchmark)
        
        # Update benchmark results
        self.benchmark_results.update(results)
        
        # Compare models
        comparison = benchmark.compare_models()
        
        # Visualize comparison
        benchmark.visualize_comparison()
        
        return comparison
    
    def recommend_model(self, 
                        priority: str = "balanced",
                        hardware_constraints: Dict = None) -> str:
        """
        Recommend the best model based on benchmark results.
        
        Args:
            priority: Priority for recommendation ('accuracy', 'speed', 'memory', 'balanced')
            hardware_constraints: Hardware constraints to consider
                e.g. {'max_memory_mb': 4000, 'min_fps': 10}
                
        Returns:
            Name of the recommended model
        """
        if not self.benchmark_results:
            print("No benchmark results available. Run benchmarks first.")
            return None
        
        # Create benchmark utility
        benchmark = ModelBenchmark(
            dataset_path="",  # Not needed for recommendation
            output_dir=self.benchmark_results_dir
        )
        
        # Set benchmark results
        benchmark.results = self.benchmark_results
        
        # Get recommendation
        recommendation = benchmark.recommend_model(
            priority=priority,
            hardware_constraints=hardware_constraints
        )
        
        return recommendation
    
    def export_model(self, 
                     model_name: str,
                     output_format: str = "onnx",
                     output_dir: str = "./exported_models",
                     quantize: bool = False,
                     optimize: bool = False):
        """
        Export a model to a specific format.
        
        Args:
            model_name: Name of the model to export
            output_format: Output format ('onnx', 'torchscript', 'tensorrt')
            output_dir: Directory to save exported model
            quantize: Whether to quantize the model
            optimize: Whether to optimize the model
        """
        # Find model config
        model_config = None
        for model in self.available_models:
            if model["name"] == model_name:
                model_config = model
                break
        
        if model_config is None:
            print(f"Model {model_name} not found.")
            return
        
        # Import export utilities
        try:
            from export_utils import export_to_onnx, quantize_model, optimize_for_tensorrt
        except ImportError:
            print("Export utilities not found. Please make sure export_utils.py is available.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export model
        if output_format == "onnx":
            # Load model
            model = self._load_model(model_config["path"], model_config["type"])
            
            # Export to ONNX
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            export_to_onnx(model, onnx_path)
            
            # Quantize if requested
            if quantize:
                quantized_path = os.path.join(output_dir, f"{model_name}_quantized.onnx")
                quantize_model(onnx_path, quantized_path, quantization_type="int8")
            
            # Optimize if requested
            if optimize:
                optimized_path = os.path.join(output_dir, f"{model_name}_optimized.onnx")
                optimize_for_tensorrt(onnx_path, optimized_path)
        
        elif output_format == "torchscript":
            # Load model
            model = self._load_model(model_config["path"], model_config["type"])
            
            # Export to TorchScript
            script_path = os.path.join(output_dir, f"{model_name}.pt")
            
            # Create example input
            example_input = [torch.rand(3, 640, 640)]
            
            # Export to TorchScript
            scripted_model = torch.jit.trace(model, example_input)
            torch.jit.save(scripted_model, script_path)
            
            print(f"Model exported to TorchScript: {script_path}")
        
        elif output_format == "tensorrt":
            # Check if TensorRT is available
            try:
                import tensorrt as trt
            except ImportError:
                print("TensorRT not available. Please install TensorRT.")
                return
            
            # Export to ONNX first
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            
            # Load model
            model = self._load_model(model_config["path"], model_config["type"])
            
            # Export to ONNX
            export_to_onnx(model, onnx_path)
            
            # Optimize for TensorRT
            tensorrt_path = os.path.join(output_dir, f"{model_name}.trt")
            optimize_for_tensorrt(onnx_path, tensorrt_path)
        
        else:
            print(f"Unsupported output format: {output_format}")
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """
        Load a model from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint
            model_type: Type of model to load
            
        Returns:
            Loaded model
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type == "efficientformer_yolo":
            # Import dynamically to avoid circular imports
            from train_efficientformer_yolo import HybridYOLOWithEfficientFormer, TrainingConfig
            
            # Load config if available
            config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config = TrainingConfig(**config_dict)
            else:
                # Default config
                config = TrainingConfig()
            
            model = HybridYOLOWithEfficientFormer(config)
            
        elif model_type == "faster_rcnn":
            model = RetailFasterRCNN(num_classes=80)  # Default to 80 classes (COCO)
            
        elif model_type == "retinanet":
            model = RetailRetinaNet(num_classes=80)  # Default to 80 classes (COCO)
            
        elif model_type == "centernet":
            model = RetailCenterNet(num_classes=80)  # Default to 80 classes (COCO)
            
        elif model_type == "ensemble":
            # Load ensemble configuration
            ensemble_config_path = os.path.join(os.path.dirname(model_path), "ensemble_config.json")
            if os.path.exists(ensemble_config_path):
                with open(ensemble_config_path, 'r') as f:
                    ensemble_config = json.load(f)
            else:
                # Default config
                ensemble_config = {"models": []}
            
            model = ModelEnsemble(ensemble_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        return model


def main():
    """Main function for the model selection utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Selection Utility for Retail Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing model checkpoints")
    list_parser.add_argument("--benchmark-results-dir", type=str, default="./benchmark_results", help="Directory containing benchmark results")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    benchmark_parser.add_argument("--dataset-path", type=str, required=True, help="Path to the validation/test dataset")
    benchmark_parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing model checkpoints")
    benchmark_parser.add_argument("--benchmark-results-dir", type=str, default="./benchmark_results", help="Directory containing benchmark results")
    benchmark_parser.add_argument("--models", type=str, nargs="+", help="List of model names to benchmark (None for all)")
    benchmark_parser.add_argument("--device", type=str, default=None, help="Device to run benchmarks on ('cuda', 'cpu', etc.)")
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Recommend the best model")
    recommend_parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing model checkpoints")
    recommend_parser.add_argument("--benchmark-results-dir", type=str, default="./benchmark_results", help="Directory containing benchmark results")
    recommend_parser.add_argument("--priority", type=str, default="balanced", choices=["accuracy", "speed", "memory", "balanced"], help="Priority for recommendation")
    recommend_parser.add_argument("--max-memory", type=float, default=None, help="Maximum memory usage in MB")
    recommend_parser.add_argument("--min-fps", type=float, default=None, help="Minimum FPS")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model")
    export_parser.add_argument("--model-name", type=str, required=True, help="Name of the model to export")
    export_parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing model checkpoints")
    export_parser.add_argument("--benchmark-results-dir", type=str, default="./benchmark_results", help="Directory containing benchmark results")
    export_parser.add_argument("--output-format", type=str, default="onnx", choices=["onnx", "torchscript", "tensorrt"], help="Output format")
    export_parser.add_argument("--output-dir", type=str, default="./exported_models", help="Directory to save exported model")
    export_parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model")
    export_parser.add_argument("--optimize", action="store_true", help="Whether to optimize the model")
    
    args = parser.parse_args()
    
    if args.command == "list":
        # Create model selector
        selector = ModelSelector(
            models_dir=args.models_dir,
            benchmark_results_dir=args.benchmark_results_dir
        )
        
        # List available models
        models_df = selector.list_available_models()
        
        if not models_df.empty:
            print(tabulate(models_df, headers="keys", tablefmt="grid"))
        else:
            print("No models available.")
    
    elif args.command == "benchmark":
        # Create model selector
        selector = ModelSelector(
            models_dir=args.models_dir,
            benchmark_results_dir=args.benchmark_results_dir
        )
        
        # Run benchmarks
        comparison = selector.run_benchmarks(
            dataset_path=args.dataset_path,
            models=args.models,
            device=args.device
        )
        
        # Print comparison
        print(tabulate(comparison, headers="keys", tablefmt="grid"))
    
    elif args.command == "recommend":
        # Create model selector
        selector = ModelSelector(
            models_dir=args.models_dir,
            benchmark_results_dir=args.benchmark_results_dir
        )
        
        # Create hardware constraints
        hardware_constraints = {}
        
        if args.max_memory is not None:
            hardware_constraints["max_memory_mb"] = args.max_memory
        
        if args.min_fps is not None:
            hardware_constraints["min_fps"] = args.min_fps
        
        # Get recommendation
        recommendation = selector.recommend_model(
            priority=args.priority,
            hardware_constraints=hardware_constraints
        )
        
        print(f"Recommended model: {recommendation}")
    
    elif args.command == "export":
        # Create model selector
        selector = ModelSelector(
            models_dir=args.models_dir,
            benchmark_results_dir=args.benchmark_results_dir
        )
        
        # Export model
        selector.export_model(
            model_name=args.model_name,
            output_format=args.output_format,
            output_dir=args.output_dir,
            quantize=args.quantize,
            optimize=args.optimize
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()