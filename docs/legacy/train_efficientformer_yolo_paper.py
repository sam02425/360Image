#!/usr/bin/env python3
"""
EfficientFormer-YOLO Training Script for Retail Object Detection
This script implements training for the hybrid EfficientFormer-YOLO model
for retail product detection with improved efficiency and accuracy.
"""

import os
import sys
import argparse
import yaml
import time
import logging
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientformer_yolo_hybrid import EfficientFormerYOLOHybrid, ModelOptimizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retail_detection_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetailDataset(torch.utils.data.Dataset):
    """
    Dataset class for retail product detection.
    Supports multiple camera views and various data augmentation strategies.
    """
    
    def __init__(
        self,
        img_dir,
        anno_path,
        img_size=640,
        augment=True,
        multi_view=True,
        view_count=4
    ):
        """
        Initialize the retail dataset.
        
        Args:
            img_dir: Directory containing images
            anno_path: Path to annotation file
            img_size: Input image size for the model
            augment: Whether to apply data augmentation
            multi_view: Whether to use multi-view data
            view_count: Number of views to use (2, 4, or 8)
        """
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.multi_view = multi_view
        self.view_count = view_count
        
        # Load annotations
        with open(anno_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter annotations based on view count if using multi-view
        if multi_view:
            self.annotations = [
                anno for anno in self.annotations 
                if len(anno['views']) >= view_count
            ]
        
        # Define augmentations
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        For multi-view data, returns a batch of images from different views.
        For single-view data, returns a single image.
        
        Returns:
            images: Tensor of shape (view_count, 3, img_size, img_size) or (3, img_size, img_size)
            targets: Dictionary containing bounding boxes, labels, etc.
        """
        anno = self.annotations[idx]
        
        if self.multi_view:
            # Select views based on view_count
            selected_views = anno['views'][:self.view_count]
            
            # Load images from all selected views
            images = []
            for view in selected_views:
                img_path = self.img_dir / view['image_path']
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply transformations
                transformed = self.transform(image=img)
                images.append(transformed['image'])
            
            # Stack images along batch dimension
            images = torch.stack(images)
            
            # Prepare targets
            targets = {
                'boxes': [torch.tensor(view['boxes']) for view in selected_views],
                'labels': [torch.tensor(view['labels']) for view in selected_views],
                'image_id': torch.tensor([idx] * len(selected_views)),
                'area': [torch.tensor(view['area']) for view in selected_views],
                'iscrowd': [torch.tensor(view['iscrowd']) for view in selected_views]
            }
        else:
            # Single view mode - just take the first view
            view = anno['views'][0]
            img_path = self.img_dir / view['image_path']
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            transformed = self.transform(image=img)
            images = transformed['image']
            
            # Prepare targets
            targets = {
                'boxes': torch.tensor(view['boxes']),
                'labels': torch.tensor(view['labels']),
                'image_id': torch.tensor([idx]),
                'area': torch.tensor(view['area']),
                'iscrowd': torch.tensor(view['iscrowd'])
            }
        
        return images, targets


class Trainer:
    """
    Trainer class for EfficientFormer-YOLO hybrid model.
    """
    
    def __init__(self, args):
        """
        Initialize the trainer.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_map = 0
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up distributed training if needed
        self.setup_distributed()
        
        # Create model
        self.create_model()
        
        # Create datasets and dataloaders
        self.create_dataloaders()
        
        # Set up optimizer and scheduler
        self.create_optimizer()
        
        # Set up mixed precision training
        self.scaler = GradScaler(enabled=args.mixed_precision)
        
        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint()
    
    def setup_distributed(self):
        """Set up distributed training if needed."""
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
        
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device('cuda', self.args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def create_model(self):
        """Create the EfficientFormer-YOLO hybrid model."""
        logger.info(f"Creating EfficientFormer-YOLO hybrid model with {self.args.yolo_model} detection head")
        
        self.model = EfficientFormerYOLOHybrid(
            yolo_model=self.args.yolo_model,
            num_classes=self.args.num_classes,
            pretrained=True,
            freeze_backbone=self.args.freeze_backbone
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Wrap model with DDP if using distributed training
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
    
    def create_dataloaders(self):
        """Create datasets and dataloaders for training and validation."""
        logger.info(f"Creating dataloaders with batch size {self.args.batch_size}")
        
        # Create training dataset
        train_dataset = RetailDataset(
            img_dir=os.path.join(self.args.dataset, 'images', 'train'),
            anno_path=os.path.join(self.args.dataset, 'annotations', 'instances_train.json'),
            img_size=self.args.img_size,
            augment=True,
            multi_view=self.args.multi_view,
            view_count=self.args.view_count
        )
        
        # Create validation dataset
        val_dataset = RetailDataset(
            img_dir=os.path.join(self.args.dataset, 'images', 'val'),
            anno_path=os.path.join(self.args.dataset, 'annotations', 'instances_val.json'),
            img_size=self.args.img_size,
            augment=False,
            multi_view=self.args.multi_view,
            view_count=self.args.view_count
        )
        
        # Create samplers for distributed training
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.
        
        Args:
            batch: List of (images, targets) tuples
            
        Returns:
            images: Tensor of shape (batch_size, view_count, 3, H, W) or (batch_size, 3, H, W)
            targets: List of target dictionaries
        """
        images, targets = zip(*batch)
        
        # Check if using multi-view
        if isinstance(images[0], torch.Tensor) and images[0].dim() == 4:
            # Multi-view: (batch_size, view_count, 3, H, W)
            images = torch.stack(images)
        else:
            # Single view: (batch_size, 3, H, W)
            images = torch.stack(images)
        
        return images, targets
    
    def create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        # Separate parameters for backbone and detection head
        backbone_params = []
        head_params = []
        
        if hasattr(self.model, 'module'):  # For distributed training
            model = self.model.module
        else:
            model = self.model
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': backbone_params, 'lr': self.args.lr * 0.1},
            {'params': head_params}
        ]
        
        # Create optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:  # Default to AdamW
            self.optimizer = optim.AdamW(
                param_groups,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        
        # Create learning rate scheduler
        if self.args.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.lr_step_size,
                gamma=self.args.lr_gamma
            )
        else:  # Default to ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
    
    def load_checkpoint(self):
        """Load checkpoint for resuming training."""
        checkpoint_path = Path(self.args.resume)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state if it exists
            if 'scheduler' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load other training states
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_map = checkpoint['best_map']
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        if self.rank != 0:
            return
        
        # Prepare checkpoint
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_map': self.best_map,
            'args': self.args
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def train_one_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        
        # Set up progress bar
        if self.rank == 0:
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}/{self.args.epochs}")
        
        # Train for one epoch
        for i, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.args.mixed_precision):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass with gradient scaling
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update progress bar
            if self.rank == 0:
                epoch_loss += losses.item()
                pbar.set_postfix({'loss': losses.item()})
                pbar.update(1)
        
        # Close progress bar
        if self.rank == 0:
            pbar.close()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            mAP: Mean Average Precision
        """
        self.model.eval()
        
        # Set up metrics
        all_predictions = []
        all_targets = []
        
        # Validate
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating", disable=self.rank != 0):
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Collect predictions and targets
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        # Log metrics
        if self.rank == 0:
            logger.info(f"Validation - mAP: {metrics['mAP']:.4f}, mAP@0.5: {metrics['mAP_50']:.4f}")
            
            # Save metrics to file
            metrics_path = self.output_dir / 'metrics.csv'
            if not metrics_path.exists():
                with open(metrics_path, 'w') as f:
                    f.write('epoch,mAP,mAP_50,precision,recall,f1\n')
            
            with open(metrics_path, 'a') as f:
                f.write(f"{self.start_epoch},{metrics['mAP']:.4f},{metrics['mAP_50']:.4f},"
                        f"{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f}\n")
        
        return metrics['mAP']
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate detection metrics.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of metrics
        """
        # Calculate mAP
        mAP = 0
        mAP_50 = 0
        precision = 0
        recall = 0
        f1 = 0
        
        # TODO: Implement proper mAP calculation
        # This is a placeholder for actual mAP calculation
        # In a real implementation, you would use a proper evaluation function
        
        return {
            'mAP': mAP,
            'mAP_50': mAP_50,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.args.epochs} epochs")
        
        # Training loop
        for epoch in range(self.start_epoch, self.args.epochs):
            # Set epoch for distributed sampler
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)
            
            # Validate
            mAP = self.validate()
            
            # Update learning rate scheduler
            if self.args.lr_scheduler == 'plateau':
                self.scheduler.step(mAP)
            else:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = mAP > self.best_map
            if is_best:
                self.best_map = mAP
            
            self.save_checkpoint(epoch, is_best)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Final message
        logger.info(f"Training completed. Best mAP: {self.best_map:.4f}")
        
        # Export best model to ONNX if requested
        if self.args.export_onnx and self.rank == 0:
            best_model_path = self.output_dir / 'checkpoint_best.pth'
            onnx_path = self.output_dir / 'best_model.onnx'
            
            logger.info(f"Exporting best model to ONNX: {onnx_path}")
            
            # Load best model
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model'])
                model_to_export = self.model.module
            else:
                self.model.load_state_dict(checkpoint['model'])
                model_to_export = self.model
            
            # Export to ONNX
            dummy_input = torch.randn(1, 3, self.args.img_size, self.args.img_size).to(self.device)
            torch.onnx.export(
                model_to_export,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to {onnx_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train EfficientFormer-YOLO hybrid model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of classes to detect (default: 80)')
    parser.add_argument('--multi-view', action='store_true',
                        help='Use multi-view data')
    parser.add_argument('--view-count', type=int, default=4, choices=[2, 4, 8],
                        help='Number of views to use (default: 4)')
    
    # Model parameters
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model variant to use (default: yolov8n.pt)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone during training')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer to use (default: adamw)')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler (default: 0.1)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    
    # Distributed training parameters
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Path to output directory (default: output)')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export best model to ONNX format')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = Trainer(args)
    
    # Train model
    trainer.train()


if __name__ == '__main__':
    main()