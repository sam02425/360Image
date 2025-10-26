"""
EfficientFormer-Enhanced YOLO Training for High-Accuracy Retail Detection
Integrates EfficientFormer-L1 image embeddings with YOLOv8/v11 for production retail environments
Author: Research Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import time
from datetime import datetime
import os
import shutil
import subprocess
from functools import partial

# Vision and detection imports
import cv2
from ultralytics import YOLO
import timm  # For EfficientFormer
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# System monitoring
import psutil
import GPUtil

# MLflow for experiment tracking
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch

# DVC for dataset and model versioning
try:
    import dvc.api
    from dvc.repo import Repo
    HAS_DVC = True
except ImportError:
    HAS_DVC = False
    logging.warning("DVC not installed. Dataset versioning will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for EfficientFormer-YOLO training"""
    # Model settings
    yolo_model: str = 'yolov8n'  # or 'yolov11n'
    efficientformer_variant: str = 'efficientformer_l1'
    use_hybrid_features: bool = True
    model_name: str = 'efficientformer_yolo'  # For tracking

    # Training parameters
    batch_size: int = 32  # Increased for large datasets
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: int = 3

    # Dataset settings
    image_size: int = 640
    num_classes: int = 414
    data_augmentation: bool = True
    cache_images: bool = True  # Cache images in RAM for faster training
    mosaic_prob: float = 0.5  # Mosaic augmentation probability
    mixup_prob: float = 0.3   # Mixup augmentation probability
    cutmix_prob: float = 0.3  # CutMix augmentation probability

    # Optimization
    optimizer: str = 'AdamW'
    scheduler: str = 'cosine'
    mixed_precision: bool = True
    gradient_clip: float = 10.0
    gradient_accumulation_steps: int = 2  # Accumulate gradients for larger effective batch size

    # Retail-specific settings
    multi_scale_training: bool = True
    focal_loss: bool = True  # Better for class imbalance
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    class_weights: Optional[str] = None  # Path to class weights file

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 8  # Increased for faster data loading
    pin_memory: bool = True
    distributed: bool = torch.cuda.device_count() > 1  # Enable distributed training if multiple GPUs

    # Paths
    dataset_path: str = 'data/datasets/quad_dataset'
    output_dir: str = 'models/efficientformer_yolo'
    checkpoint_dir: str = 'checkpoints'
    
    # MLflow tracking
    use_mlflow: bool = True
    mlflow_tracking_uri: str = 'mlruns'
    mlflow_experiment_name: str = 'retail_detection'
    
    # DVC configuration
    use_dvc: bool = True
    dvc_remote: str = 'remote'  # DVC remote storage name
    dvc_dataset_version: Optional[str] = None  # Specific dataset version to use
    
    # Large dataset handling
    shard_dataset: bool = True  # Shard dataset for distributed training
    prefetch_factor: int = 4  # Prefetch factor for DataLoader
    persistent_workers: bool = True  # Keep workers alive between epochs
    
    # Advanced training features
    ema: bool = True  # Exponential moving average of weights
    label_smoothing: float = 0.1  # Label smoothing factor
    auto_augment: bool = True  # Use AutoAugment policies
    
    # Callbacks and logging
    save_period: int = 5  # Save checkpoint every N epochs
    eval_period: int = 5  # Evaluate every N epochs
    log_metrics_period: int = 1  # Log metrics every N batches
    
    def __post_init__(self):
        """Validate and adjust configuration based on environment"""
        # Adjust batch size based on available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory < 8 and self.batch_size > 16:
                self.batch_size = 16
                logging.warning(f"Reduced batch size to {self.batch_size} due to limited GPU memory ({gpu_memory:.1f}GB)")
        
        # Adjust workers based on CPU cores
        cpu_cores = os.cpu_count()
        if cpu_cores and self.num_workers > cpu_cores:
            self.num_workers = max(1, cpu_cores - 2)  # Leave some cores for system
            
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class EfficientFormerFeatureExtractor(nn.Module):
    """
    EfficientFormer-L1 feature extractor for enhanced image embeddings
    Lightweight yet powerful for retail product detection
    """

    def __init__(self, pretrained: bool = True, embedding_dim: int = 512):
        super().__init__()

        # Load EfficientFormer-L1 model
        self.backbone = timm.create_model(
            'efficientformer_l1',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # Multi-scale features
        )

        # Feature dimensions from EfficientFormer-L1
        # [48, 96, 224, 448] for the 4 stages
        self.feature_dims = [48, 96, 224, 448]

        # Feature projection layers for YOLO integration
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embedding_dim, 1),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            ) for dim in self.feature_dims
        ])

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim * 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim * 2, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Attention mechanism for retail-specific features
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embedding_dim, embedding_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 16, embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features with attention
        """
        # Get multi-scale features from EfficientFormer
        features = self.backbone(x)

        # Project features to common dimension
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            projected.append(proj(feat))

        # Resize all features to same spatial resolution for fusion
        target_size = projected[2].shape[2:]  # Use middle scale as reference
        resized = []
        for feat in projected:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized.append(feat)

        # Concatenate and fuse features
        concatenated = torch.cat(resized, dim=1)
        fused = self.fusion(concatenated)

        # Apply attention
        attention_weights = self.attention(fused)
        enhanced_features = fused * attention_weights

        return {
            'multi_scale': projected,  # Original multi-scale features
            'fused': fused,  # Fused features
            'enhanced': enhanced_features,  # Attention-enhanced features
            'attention': attention_weights  # Attention weights for visualization
        }


class HybridYOLOWithEfficientFormer(nn.Module):
    """
    Hybrid model combining EfficientFormer features with YOLO detection
    Optimized for high-accuracy retail product detection from mounted cameras
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # EfficientFormer feature extractor
        self.feature_extractor = EfficientFormerFeatureExtractor(
            pretrained=True,
            embedding_dim=512
        )

        # YOLO detector
        self.yolo = YOLO(f'{config.yolo_model}.pt')

        # Feature integration layer
        self.feature_integration = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),  # Combine YOLO and EfficientFormer features
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Retail-specific enhancement modules
        self.product_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, config.num_classes)
        )

        # Angle-invariant feature module
        self.angle_invariant = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass with hybrid features
        """
        # Extract EfficientFormer features
        ef_features = self.feature_extractor(x)

        # Get YOLO predictions and features
        if self.training and targets is not None:
            yolo_output = self.yolo.model(x, augment=False)
        else:
            yolo_output = self.yolo.predict(x, verbose=False)

        # Combine features if in feature mode
        if self.config.use_hybrid_features:
            # Extract intermediate YOLO features (customize based on YOLO architecture)
            # This is a simplified example - actual implementation depends on YOLO internals
            combined_features = torch.cat([
                ef_features['enhanced'],
                yolo_output  # Adjust based on actual YOLO output structure
            ], dim=1)

            integrated_features = self.feature_integration(combined_features)

            # Additional retail-specific processing
            angle_invariant_features = self.angle_invariant(ef_features['enhanced'])
            product_logits = self.product_classifier(ef_features['enhanced'])

            return {
                'detections': yolo_output,
                'features': integrated_features,
                'angle_invariant': angle_invariant_features,
                'product_logits': product_logits,
                'attention': ef_features['attention']
            }

        return yolo_output


class RetailDatasetWithAugmentation(torch.utils.data.Dataset):
    """
    Dataset class with retail-specific augmentations
    Optimized for large datasets with DVC versioning support
    """

    def __init__(self, data_path: str, split: str = 'train', config: TrainingConfig = None):
        self.data_path = Path(data_path)
        self.split = split
        self.config = config or TrainingConfig()
        self.cache = {}  # Cache for faster loading
        
        # Initialize DVC if available and configured
        self.dvc_enabled = HAS_DVC and self.config.use_dvc
        if self.dvc_enabled:
            try:
                self.dvc_repo = Repo(os.getcwd())
                logging.info(f"DVC initialized. Using dataset: {data_path}")
                
                # Pull specific dataset version if specified
                if self.config.dvc_dataset_version:
                    logging.info(f"Pulling dataset version: {self.config.dvc_dataset_version}")
                    subprocess.run(["dvc", "checkout", self.config.dvc_dataset_version, data_path], check=True)
                    
                # Track dataset usage for reproducibility
                with open("dataset_version.txt", "w") as f:
                    f.write(f"Dataset: {data_path}\n")
                    f.write(f"Version: {self.config.dvc_dataset_version or 'latest'}\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                
                self.dvc_repo.add("dataset_version.txt")
                self.dvc_repo.commit("Track dataset version for reproducibility")
            except Exception as e:
                logging.warning(f"Failed to initialize DVC: {e}")
                self.dvc_enabled = False
        
        # Load image paths and labels efficiently
        logging.info(f"Loading dataset from {self.data_path}")
        images_dir = self.data_path / split / 'images'
        labels_dir = self.data_path / split / 'labels'
        
        # Use faster file listing for large datasets
        if os.path.exists(images_dir):
            self.images = sorted([images_dir / f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            self.labels = sorted([labels_dir / f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') 
                                for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        else:
            raise FileNotFoundError(f"Dataset directory not found: {images_dir}")
            
        logging.info(f"Loaded {len(self.images)} images for {split} split")
        
        # Create image index for faster access
        self.image_index = {img.stem: i for i, img in enumerate(self.images)}
        
        # Preload labels for faster access
        self.preloaded_labels = {}
        if self.config.cache_images and split == 'train':
            logging.info("Preloading labels for faster training...")
            for label_path in self.labels:
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        self.preloaded_labels[label_path.stem] = np.array(
                            [list(map(float, line.strip().split())) for line in f.readlines()]
                        )

        # Define augmentation pipeline
        self.transform = self.get_augmentation_pipeline()

    def get_augmentation_pipeline(self):
        """
        Retail-specific augmentation pipeline for mounted camera scenarios
        """
        if self.split == 'train' and self.config.data_augmentation:
            return A.Compose([
                # Geometric transforms for angle variance
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),

                # Perspective transform for camera angle simulation
                A.Perspective(scale=(0.05, 0.1), p=0.3),

                # Lighting conditions in retail environment
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                # Simulate different retail lighting
                A.OneOf([
                    A.RandomToneCurve(scale=0.1),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5
                    ),
                ], p=0.5),

                # Blur for motion or focus issues
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.3),

                # Normalize and convert
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
        else:
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get item with caching and advanced augmentations for large datasets
        """
        img_path = self.images[idx]
        label_path = self.labels[idx]
        img_id = img_path.stem
        
        # Use cached data if available
        if self.config.cache_images and img_id in self.cache:
            return self.cache[img_id]
            
        # Load image efficiently
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a valid but empty sample to avoid training crashes
            return {
                'image': torch.zeros((3, self.config.image_size, self.config.image_size)),
                'boxes': torch.zeros((0, 5)),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': img_id
            }
            
        # Load labels from preloaded cache or file
        if img_id in self.preloaded_labels:
            labels = self.preloaded_labels[img_id]
        elif label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    labels = np.array([list(map(float, line.strip().split())) for line in f.readlines()])
            except Exception as e:
                logging.error(f"Error loading labels {label_path}: {e}")
                labels = np.zeros((0, 5))
        else:
            labels = np.zeros((0, 5))
            
        # Extract bounding boxes and class labels
        if len(labels) > 0:
            boxes = labels[:, 1:5]  # YOLO format: [class, x_center, y_center, width, height]
            class_labels = labels[:, 0].astype(int)
        else:
            boxes = np.zeros((0, 4))
            class_labels = np.zeros(0)
            
        # Apply advanced augmentations for training
        if self.split == 'train' and self.config.data_augmentation:
            # Apply mosaic augmentation with probability
            if np.random.random() < self.config.mosaic_prob and len(labels) > 0:
                img, boxes, class_labels = self._apply_mosaic(img, boxes, class_labels)
                
            # Apply mixup augmentation with probability
            if np.random.random() < self.config.mixup_prob and len(labels) > 0:
                img, boxes, class_labels = self._apply_mixup(img, boxes, class_labels)
                
            # Apply cutmix augmentation with probability
            if np.random.random() < self.config.cutmix_prob and len(labels) > 0:
                img, boxes, class_labels = self._apply_cutmix(img, boxes, class_labels)
                
        # Apply standard transformations
        transformed = self.transform(image=img, bboxes=boxes, labels=class_labels)
        image = transformed['image']
        
        # Prepare final output
        if len(transformed.get('bboxes', [])) > 0:
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            
        # Create sample
        sample = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id
        }
        
        # Cache sample if enabled
        if self.config.cache_images:
            self.cache[img_id] = sample
            
        return sample
        
    def _apply_mosaic(self, img, boxes, class_labels):
        """Apply mosaic augmentation by combining 4 images"""
        # Implementation details omitted for brevity
        # This would combine 4 random images in a grid
        return img, boxes, class_labels
        
    def _apply_mixup(self, img, boxes, class_labels):
        """Apply mixup augmentation by blending two images"""
        # Implementation details omitted for brevity
        # This would blend current image with another random image
        return img, boxes, class_labels
        
    def _apply_cutmix(self, img, boxes, class_labels):
        """Apply cutmix augmentation by cutting and pasting regions"""
        # Implementation details omitted for brevity
        # This would cut a region from another image and paste it
        return img, boxes, class_labels
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = self.labels[idx]
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [list(map(float, line.strip().split())) for line in f]
        else:
            labels = []

        # Apply augmentations
        if labels:
            bboxes = [label[1:5] for label in labels]
            class_labels = [int(label[0]) for label in labels]

            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                labels=class_labels
            )

            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['labels']

            # Reconstruct labels
            labels = [[cl] + list(bbox) for cl, bbox in zip(class_labels, bboxes)]
        else:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, torch.tensor(labels) if labels else torch.zeros((0, 5))


class EfficientFormerYOLOTrainer:
    """
    Trainer class for EfficientFormer-enhanced YOLO models
    Optimized for production retail detection with mounted cameras
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = HybridYOLOWithEfficientFormer(config).to(self.device)

        # Setup directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize training components
        self.setup_training()

        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_map50': [],
            'val_map50_95': [],
            'precision': [],
            'recall': []
        }

        self.best_map50 = 0.0

    def setup_training(self):
        """
        Setup training components: optimizer, scheduler, loss functions
        """
        # Setup optimizer
        if self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )

        # Setup scheduler
        if self.config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )

        # Setup mixed precision training
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Setup datasets and dataloaders
        self.setup_dataloaders()

    def setup_dataloaders(self):
        """
        Setup data loaders for training and validation
        """
        # Training dataset
        train_dataset = RetailDatasetWithAugmentation(
            self.config.dataset_path,
            split='train',
            config=self.config
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        # Validation dataset
        val_dataset = RetailDatasetWithAugmentation(
            self.config.dataset_path,
            split='val',
            config=self.config
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def train_epoch(self, epoch: int):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, targets)

                    # Calculate combined loss
                    detection_loss = outputs.get('loss', 0)  # YOLO loss

                    # Additional losses for retail-specific features
                    if 'product_logits' in outputs:
                        classification_loss = F.cross_entropy(
                            outputs['product_logits'],
                            targets[:, 0].long()  # Assuming first column is class
                        )
                    else:
                        classification_loss = 0

                    total_batch_loss = detection_loss + 0.5 * classification_loss

                # Backward pass with gradient scaling
                self.scaler.scale(total_batch_loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard training without mixed precision
                outputs = self.model(images, targets)
                total_batch_loss = outputs.get('loss', 0)

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.optimizer.zero_grad()

            total_loss += total_batch_loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_batch_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        return total_loss / batch_count

    def validate(self):
        """
        Validation step with comprehensive metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)

                # Store predictions for metric calculation
                all_predictions.extend(outputs.get('detections', []))
                all_targets.extend(targets)

                # Calculate loss
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

        # Calculate metrics (simplified - actual implementation would use COCO metrics)
        metrics = self.calculate_metrics(all_predictions, all_targets)

        return total_loss / len(self.val_loader), metrics

    def calculate_metrics(self, predictions, targets):
        """
        Calculate detection metrics
        """
        # This is a simplified version - actual implementation would use COCO evaluation
        # For production, integrate with pycocotools or similar

        metrics = {
            'map50': 0.0,  # Placeholder
            'map50_95': 0.0,  # Placeholder
            'precision': 0.0,  # Placeholder
            'recall': 0.0  # Placeholder
        }

        # In production, you would:
        # 1. Convert predictions to COCO format
        # 2. Use COCOeval for comprehensive metrics
        # 3. Calculate per-class metrics for retail categories

        return metrics

    def train(self):
        """
        Full training loop
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Configuration: {self.config}")

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss, val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_map50'].append(val_metrics['map50'])
            self.metrics['val_map50_95'].append(val_metrics['map50_95'])

            logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"mAP@50: {val_metrics['map50']:.4f}, "
                f"mAP@50-95: {val_metrics['map50_95']:.4f}"
            )

            # Save best model
            if val_metrics['map50'] > self.best_map50:
                self.best_map50 = val_metrics['map50']
                self.save_checkpoint(epoch, is_best=True)

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_final_model()
        self.save_metrics()

        logger.info("Training completed!")
        logger.info(f"Best mAP@50: {self.best_map50:.4f}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map50': self.best_map50,
            'metrics': self.metrics,
            'config': self.config
        }

        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        filepath = Path(self.config.checkpoint_dir) / filename

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")

    def save_final_model(self):
        """
        Save final model for production deployment
        """
        # Save complete model
        model_path = Path(self.config.output_dir) / 'efficientformer_yolo_final.pth'
        torch.save(self.model.state_dict(), model_path)

        # Export to ONNX for deployment
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        onnx_path = Path(self.config.output_dir) / 'efficientformer_yolo_final.onnx'

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        logger.info(f"Saved final model: {model_path}")
        logger.info(f"Exported ONNX model: {onnx_path}")

    def save_metrics(self):
        """
        Save training metrics
        """
        metrics_path = Path(self.config.output_dir) / 'training_metrics.json'

        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Create visualization
        self.plot_training_curves()

    def plot_training_curves(self):
        """
        Plot and save training curves
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # mAP curves
        axes[0, 1].plot(self.metrics['val_map50'], label='mAP@50')
        axes[0, 1].plot(self.metrics['val_map50_95'], label='mAP@50-95')
        axes[0, 1].set_title('mAP Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision/Recall
        axes[1, 0].plot(self.metrics['precision'], label='Precision')
        axes[1, 0].plot(self.metrics['recall'], label='Recall')
        axes[1, 0].set_title('Precision/Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot([self.scheduler.get_last_lr()[0] for _ in range(len(self.metrics['train_loss']))])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'training_curves.png', dpi=300)
        plt.close()


def main():
    """
    Main training function with MLflow tracking and optimizations for large datasets
    """
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='EfficientFormer-YOLO Training for Retail Detection')

    # Model arguments
    parser.add_argument('--yolo-model', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov11n'],
                       help='YOLO model variant')
    parser.add_argument('--efficientformer', type=str, default='efficientformer_l1',
                       help='EfficientFormer variant')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='data/datasets/quad_dataset',
                       help='Path to dataset')
    parser.add_argument('--num-classes', type=int, default=414,
                       help='Number of product classes')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/efficientformer_yolo',
                       help='Output directory for models')
    
    # MLflow arguments
    parser.add_argument('--use-mlflow', action='store_true',
                       help='Enable MLflow tracking')
    parser.add_argument('--mlflow-uri', type=str, default='mlruns',
                       help='MLflow tracking URI')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        yolo_model=args.yolo_model,
        efficientformer_variant=args.efficientformer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        dataset_path=args.dataset,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_uri
    )
    
    # Initialize MLflow if enabled
    if config.use_mlflow:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "yolo_model": config.yolo_model,
                "efficientformer_variant": config.efficientformer_variant,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "image_size": config.image_size,
                "num_classes": config.num_classes
            })
            
            # Initialize trainer
            trainer = EfficientFormerYOLOTrainer(config)
            
            # Start training with MLflow tracking
            trainer.train()
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                trainer.model, 
                "model",
                registered_model_name=f"{config.model_name}"
            )
            
            # Log system metrics
            if torch.cuda.is_available():
                gpu_info = GPUtil.getGPUs()[0]
                mlflow.log_metrics({
                    "gpu_memory_total": gpu_info.memoryTotal,
                    "gpu_memory_used": gpu_info.memoryUsed,
                    "gpu_load": gpu_info.load
                })
            
            print(f"Training completed with MLflow tracking. Run ID: {mlflow.active_run().info.run_id}")
    else:
        # Initialize trainer
        trainer = EfficientFormerYOLOTrainer(config)
        
        # Start training without MLflow
        trainer.train()


if __name__ == "__main__":
    main()