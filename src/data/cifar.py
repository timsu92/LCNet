from __future__ import annotations

from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.utils.config import Config


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns the training and validation DataLoaders for CIFAR-10.
    """
    
    # Standard CIFAR-10 normalization stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    # Download and load datasets
    train_set = torchvision.datasets.CIFAR10(
        root=config.data.data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_set = torchvision.datasets.CIFAR10(
        root=config.data.data_path,
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
