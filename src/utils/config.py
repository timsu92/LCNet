from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
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
    max_batch_size: int = 4096
    lr: float = 0.001
    weight_decay: float = 0.001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    num_workers: int = 4
    eval_interval: int = 1
    save_interval: int = 10
    auto_batch_size: bool = True
    resume: Optional[str] = None


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
            "--max-batch-size",
            type=int,
            default=4096,
            help="Maximum batch size for auto-detection",
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
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint to resume from",
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

        # If resuming, load config from checkpoint directory
        if args.resume:
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            # Load config from the checkpoint's directory
            config_path = checkpoint_path.parent / "config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            config = cls.load_yaml(config_path)

            # Override with CLI args that make sense during resume
            if args.epochs != parser.get_default("epochs"):
                config.training.epochs = args.epochs
            if args.num_workers != parser.get_default("num_workers"):
                config.training.num_workers = args.num_workers

            # Set resume path
            config.training.resume = args.resume

            return config

        # Normal config creation from CLI args
        return cls(
            model=ModelConfig(variant=args.variant),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                auto_batch_size=not args.no_auto_batch,
                resume=args.resume,
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

        # Convert dataclasses to dict but SKIP computed fields (init=False)
        # so that fields like `OutputConfig.model_dir` (computed in __post_init__) are
        # not persisted and won't need special handling on load.

        def dataclass_to_dict(obj):
            if isinstance(obj, Path):
                return str(obj)
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    # Skip fields that are not part of __init__ (computed)
                    if not f.init:
                        continue
                    val = getattr(obj, f.name)
                    result[f.name] = dataclass_to_dict(val)
                return result
            if isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [dataclass_to_dict(v) for v in obj]
            return obj

        config_dict = dataclass_to_dict(self)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)
        print(f"Config saved to {path}")

    @classmethod
    def load_yaml(cls, path: Path) -> Config:
        """Loads configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Reconstruct nested dataclasses but only pass init=True fields so that
        # computed fields (init=False) are not restored from YAML.
        def filter_init_fields(dc_cls, raw: Dict[str, Any]) -> Dict[str, Any]:
            allowed = {f.name for f in fields(dc_cls) if f.init}
            return {k: v for k, v in raw.items() if k in allowed}

        model_kwargs = filter_init_fields(ModelConfig, config_dict.get("model", {}))
        training_kwargs = filter_init_fields(
            TrainingConfig, config_dict.get("training", {})
        )
        data_kwargs = filter_init_fields(DataConfig, config_dict.get("data", {}))
        output_kwargs = filter_init_fields(OutputConfig, config_dict.get("output", {}))

        return cls(
            model=ModelConfig(**model_kwargs),
            training=TrainingConfig(**training_kwargs),
            data=DataConfig(**data_kwargs),
            output=OutputConfig(**output_kwargs),
        )
