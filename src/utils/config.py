from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class ModelConfig:
    variant: Literal["tiny", "small", "base"] = "base"  # 預設改為 base
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 224  # [新參數] 圖片尺寸

@dataclass
class TrainingConfig:
    epochs: int = 300  # 預設 300
    batch_size: int = 128
    max_batch_size: int = 4096 # [原參數] 保留
    lr: float = 0.001
    weight_decay: float = 0.05 # [建議] 配合 Mixup 調大
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixup_alpha: float = 0.8  # [新參數] Mixup 強度

    num_workers: int = 4  # [原參數] 保留
    eval_interval: int = 1  # [原參數] 保留
    save_interval: int = 10  # [原參數] 保留
    auto_batch_size: bool = True  # [原參數] 保留
    resume: Optional[str] = None  # [原參數] 保留


@dataclass
class DataConfig:
    data_path: str = "./data"
    dataset: str = "cifar10"


@dataclass
class OutputConfig:
    base_dir: str = "./out"
    model_id: Optional[int] = None
    model_dir: Path = field(init=False)

    def __post_init__(self):
        self.model_dir = Path(self.base_dir)


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def from_args(cls) -> Config:
        parser = argparse.ArgumentParser(description="LCNet Training with Mixup")

        # --- Model Args ---
        parser.add_argument(
            "--variant", type=str, default="base", choices=["tiny", "small", "base"]
        )
        parser.add_argument(
            "--image-size", type=int, default=224, help="Input resolution"
        )

        # --- Training Args ---
        parser.add_argument("--epochs", type=int, default=300)
        parser.add_argument("--batch-size", type=int, default=128)

        # [修復] 加回原本的參數
        parser.add_argument("--max-batch-size", type=int, default=4096)
        parser.add_argument("--no-auto-batch", action="store_true", help="Disable auto batch size detection")

        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument(
            "--mixup-alpha",
            type=float,
            default=0.8,
            help="Mixup alpha value (0 to disable)",
        )

        # [修復] 加回原本的參數
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--eval-interval", type=int, default=1)
        parser.add_argument("--save-interval", type=int, default=10)

        parser.add_argument("--resume", type=str, default=None)

        # --- Data & Output Args ---
        parser.add_argument("--data-path", type=str, default="./data")
        parser.add_argument("--output-dir", type=str, default="./out")
        parser.add_argument("--model-id", type=int, default=None)

        args = parser.parse_args()

        # Resuming logic
        if args.resume and Path(args.resume).exists():
             checkpoint_path = Path(args.resume)
             config_path = checkpoint_path.parent / "config.yaml"
             if config_path.exists():
                 # Load saved config but override with CLI args if provided
                 config = cls.load_yaml(config_path)
                 config.training.resume = args.resume
                 return config

        return cls(
            model=ModelConfig(variant=args.variant, image_size=args.image_size),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size, # 傳入
                lr=args.lr,
                mixup_alpha=args.mixup_alpha,
                num_workers=args.num_workers,       # 傳入
                eval_interval=args.eval_interval,   # 傳入
                save_interval=args.save_interval,   # 傳入
                auto_batch_size=not args.no_auto_batch, # 邏輯反轉
                resume=args.resume
            ),
            data=DataConfig(data_path=args.data_path),
            output=OutputConfig(base_dir=args.output_dir, model_id=args.model_id),
        )

    def resolve_paths(self):
        base_path = Path(self.output.base_dir)
        models_path = base_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        if self.output.model_id is None:
            existing = [int(p.name) for p in models_path.iterdir() if p.is_dir() and p.name.isdigit()]
            self.output.model_id = max(existing) + 1 if existing else 1
        self.output.model_dir = models_path / str(self.output.model_id)
        self.output.model_dir.mkdir(parents=True, exist_ok=True)

    def save_yaml(self, path: Optional[Path] = None):
        if path is None:
            path = self.output.model_dir / "config.yaml"

        # Helper to convert dataclass to dict
        def dataclass_to_dict(obj):
            if isinstance(obj, Path):
                return str(obj)
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
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

        with open(path, "w") as f:
            yaml.dump(dataclass_to_dict(self), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Path) -> Config:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Filtering helper
        def filter_init_fields(dc_cls, raw):
            allowed = {f.name for f in fields(dc_cls) if f.init}
            return {k: v for k, v in raw.items() if k in allowed}

        return cls(
            model=ModelConfig(**filter_init_fields(ModelConfig, config_dict.get("model", {}))),
            training=TrainingConfig(**filter_init_fields(TrainingConfig, config_dict.get("training", {}))),
            data=DataConfig(**filter_init_fields(DataConfig, config_dict.get("data", {}))),
            output=OutputConfig(**filter_init_fields(OutputConfig, config_dict.get("output", {}))),
        )