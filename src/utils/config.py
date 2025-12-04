from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    variant: str = "tiny"  # tiny, small, base
    num_classes: int = 10
    input_channels: int = 3
    # Additional model params can be added here or derived from variant in the model code


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128  # Initial value, can be updated by auto-detection
    lr: float = 0.001
    weight_decay: float = 0.001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    num_workers: int = 4
    eval_interval: int = 1
    save_interval: int = 10
    auto_batch_size: bool = True


@dataclass
class DataConfig:
    data_path: str = "./data"
    dataset: str = "cifar10"


@dataclass
class OutputConfig:
    base_dir: str = "./out"
    model_id: Optional[int] = None

    # Computed paths
    model_dir: Path = field(init=False)

    def __post_init__(self):
        # Initialize with a dummy path or None, will be resolved later
        self.model_dir = Path(self.base_dir)


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def from_args(cls) -> Config:
        parser = argparse.ArgumentParser(description="LCNet Training")

        # Model args
        parser.add_argument(
            "--variant",
            type=str,
            default="tiny",
            choices=["tiny", "small", "base"],
            help="LCNet variant",
        )

        # Training args
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="Batch size (if auto_batch_size is False or as base)",
        )
        parser.add_argument(
            "--no-auto-batch",
            action="store_true",
            help="Disable auto batch size detection",
        )
        parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        parser.add_argument(
            "--num-workers", type=int, default=4, help="Number of data loader workers"
        )
        parser.add_argument(
            "--eval-interval", type=int, default=1, help="Epoch interval for evaluation"
        )
        parser.add_argument(
            "--save-interval",
            type=int,
            default=10,
            help="Epoch interval for saving checkpoints",
        )

        # Data args
        parser.add_argument(
            "--data-path", type=str, default="./data", help="Path to dataset"
        )

        # Output args
        parser.add_argument(
            "--output-dir", type=str, default="./out", help="Base output directory"
        )
        parser.add_argument(
            "--model-id",
            type=int,
            default=None,
            help="Force a specific model ID (default: auto-increment)",
        )

        args = parser.parse_args()

        return cls(
            model=ModelConfig(variant=args.variant),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                auto_batch_size=not args.no_auto_batch,
            ),
            data=DataConfig(data_path=args.data_path),
            output=OutputConfig(base_dir=args.output_dir, model_id=args.model_id),
        )

    def resolve_paths(self):
        """Determines model_id and creates directories."""
        base_path = Path(self.output.base_dir)
        models_path = base_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)

        if self.output.model_id is None:
            # Auto-increment: find the max existing integer folder in models/
            existing = [
                int(p.name)
                for p in models_path.iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
            self.output.model_id = max(existing) + 1 if existing else 1

        self.output.model_dir = models_path / str(self.output.model_id)
        self.output.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Model ID assigned: {self.output.model_id}")
        print(f"Output directory: {self.output.model_dir}")

    def save_yaml(self, path: Optional[Path] = None):
        """Saves the current configuration to a YAML file."""
        if path is None:
            path = self.output.model_dir / "config.yaml"

        # Helper to convert Paths to str for YAML serialization
        def convert_to_dict(obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return convert_to_dict(asdict(obj))
            if isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_dict(v) for v in obj]
            return obj

        config_dict = convert_to_dict(self)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)
        print(f"Config saved to {path}")

    @classmethod
    def load_yaml(cls, path: Path) -> Config:
        """Loads configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        # Note: We need to handle Path objects conversion back if needed,
        # but here we just pass strings to dataclasses which is fine for now
        # as long as __post_init__ handles them or they are strings.

        # OutputConfig needs special handling because model_dir is init=False
        output_data = config_dict["output"]
        # Remove computed fields if present in yaml (though save_yaml might have saved them)
        if "model_dir" in output_data:
            del output_data["model_dir"]

        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            output=OutputConfig(**output_data),
        )
