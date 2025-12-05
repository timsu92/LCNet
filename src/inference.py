"""
Inference script for LCNet model.
Supports three modes (auto-detected):
1. No input: CIFAR-10 validation set inference
2. Image file: Single image inference
3. Directory: Batch inference on all images in directory
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .data.cifar import get_dataloaders
from .models.lcnet import LCNet
from .utils.config import Config, InferenceConfig
from .utils.logging import get_logger
from .utils.system_info import auto_batch_size

logger = get_logger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a directory"""

    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, str(image_path)


def load_model(
    config: Config,
    checkpoint_path: Path,
    device: torch.device,
) -> LCNet:
    """Load trained model from checkpoint"""
    logger.info(f"Loading model: {config.model.variant}")
    logger.info(f"  - use_kan: {config.model.use_kan}")
    logger.info(f"  - checkpoint: {checkpoint_path}")

    # Create model
    model = LCNet(
        num_classes=config.model.num_classes,
        variant=config.model.variant,
        use_kan=config.model.use_kan,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint (legacy format)")

    model.to(device)
    model.eval()

    return model


def get_image_transform(image_size: int = 32):
    """Get transformation pipeline for input images"""
    # CIFAR-10 normalization stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
    )


def calculate_topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, None], axis=1)
    return correct.mean()


def infer_single_image(
    model: LCNet,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """Infer on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor: torch.Tensor = transform(image).unsqueeze(0).to(device)  # type: ignore

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs[0], top_k)

    # Format results
    predictions = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        predictions.append(
            {
                "class_id": int(idx),
                "class_name": CIFAR10_CLASSES[idx],
                "probability": float(prob),
            }
        )

    return {
        "image_path": str(image_path),
        "predictions": predictions,
    }


def infer_directory(
    model: LCNet,
    directory_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int = 5,
) -> list[dict]:
    """Infer on all images in a directory with batching"""
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = [
        f
        for f in directory_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        logger.warning(f"No images found in {directory_path}")
        return []

    logger.info(f"Found {len(image_files)} images in {directory_path}")

    # Create dataset and dataloader for batch processing
    dataset = ImageFolderDataset(image_files, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Batch inference
    results = []
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Inferring directory"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # Get top-k for each image
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)

            # Format results for each image in batch
            for i, path in enumerate(paths):
                predictions = []
                for prob, idx in zip(
                    top_probs[i].cpu().numpy(), top_indices[i].cpu().numpy()
                ):
                    predictions.append(
                        {
                            "class_id": int(idx),
                            "class_name": CIFAR10_CLASSES[idx],
                            "probability": float(prob),
                        }
                    )

                results.append(
                    {
                        "image_path": path,
                        "predictions": predictions,
                    }
                )

    return results


def infer_cifar10(
    model: LCNet,
    config: Config,
    inf_config: InferenceConfig,
    device: torch.device,
) -> dict:
    """Infer on CIFAR-10 validation set with metrics"""
    logger.info("Loading CIFAR-10 validation set...")

    # Get validation loader
    _, val_loader = get_dataloaders(config)

    # Override batch size and num_workers
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=inf_config.batch_size,
        shuffle=False,
        num_workers=inf_config.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Inference
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Inferring CIFAR-10"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Top-1 accuracy (standard accuracy)
    top1_accuracy = (all_predictions == all_labels).mean()

    # Top-k accuracy
    topk_accuracy = calculate_topk_accuracy(all_probs, all_labels, inf_config.top_k)

    # Per-class accuracy (top-1)
    per_class_acc = {}
    for i, class_name in enumerate(CIFAR10_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == all_labels[mask]).mean()
            per_class_acc[class_name] = float(class_acc)

    # Confusion matrix
    num_classes = len(CIFAR10_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_predictions):
        confusion_matrix[true_label, pred_label] += 1

    logger.info(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    logger.info(f"Top-{inf_config.top_k} Accuracy: {topk_accuracy:.4f}")
    logger.info("Per-class Accuracy (Top-1):")
    for class_name, acc in per_class_acc.items():
        logger.info(f"  {class_name:12s}: {acc:.4f}")

    return {
        "top1_accuracy": float(top1_accuracy),
        f"top{inf_config.top_k}_accuracy": float(topk_accuracy),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.tolist(),
        "num_samples": len(all_labels),
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
    }


def save_results(results: dict | list, output_dir: Path):
    """Save inference results to JSON"""
    output_file = output_dir / "results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Save confusion matrix separately if available (CIFAR-10 mode)
    if isinstance(results, dict) and "confusion_matrix" in results:
        cm_file = output_dir / "confusion_matrix.csv"
        confusion_matrix = np.array(results["confusion_matrix"])

        # Save as CSV with headers
        with open(cm_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Header row
            writer.writerow([""] + CIFAR10_CLASSES)
            # Data rows
            for i, row in enumerate(confusion_matrix):
                writer.writerow([CIFAR10_CLASSES[i]] + row.tolist())

        logger.info(f"Confusion matrix saved to {cm_file}")


def main():
    """Main inference function"""
    # Parse arguments
    inf_config = InferenceConfig.from_args()

    # Load training config
    logger.info(f"Loading config from {inf_config.config_path}")
    config = Config.load_yaml(inf_config.config_path)

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Auto-detect batch size if enabled
    if inf_config.auto_batch_size and device.type == "cuda":
        logger.info("Auto-detecting optimal batch size for inference...")

        def build_model():
            return LCNet(
                num_classes=config.model.num_classes,
                variant=config.model.variant,
                use_kan=config.model.use_kan,
            )

        detected_batch = auto_batch_size(
            device=device,
            build_model=build_model,
            sample_shape=(1, 3, 32, 32),
            max_batch=inf_config.max_batch_size,
            base_batch=inf_config.batch_size,
        )
        inf_config.batch_size = detected_batch
        logger.info(f"Auto-detected batch size: {detected_batch}")
    else:
        logger.info(f"Using batch size: {inf_config.batch_size}")

    logger.info(f"Inference output will be saved to: {inf_config.output_dir}")

    # Load model
    model = load_model(config, inf_config.checkpoint_path, device)

    # Run inference based on mode
    if inf_config.mode == "image":
        assert inf_config.input_path is not None
        logger.info(f"Inferring on single image: {inf_config.input_path}")
        transform = get_image_transform(image_size=32)
        results = infer_single_image(
            model,
            inf_config.input_path,
            transform,
            device,
            inf_config.top_k,
        )

        # Print results
        logger.info("Predictions:")
        for pred in results["predictions"]:
            logger.info(f"  {pred['class_name']:12s}: {pred['probability']:.4f}")

    elif inf_config.mode == "directory":
        assert inf_config.input_path is not None
        logger.info(f"Inferring on directory: {inf_config.input_path}")
        transform = get_image_transform(image_size=32)
        results = infer_directory(
            model,
            inf_config.input_path,
            transform,
            device,
            inf_config.batch_size,
            inf_config.num_workers,
            inf_config.top_k,
        )

        logger.info(f"Processed {len(results)} images")

    elif inf_config.mode == "cifar10":
        logger.info("Inferring on CIFAR-10 validation set")
        results = infer_cifar10(model, config, inf_config, device)

    else:
        raise ValueError(f"Unknown mode: {inf_config.mode}")

    # Save results
    save_results(results, inf_config.output_dir)

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
