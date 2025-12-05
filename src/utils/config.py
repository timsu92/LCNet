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
    use_kan: bool = False  # [新參數] 是否使用 KAN 替換 MLP

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
class InferenceConfig:
    """Configuration for inference runs"""

    checkpoint_path: Path
    input_path: Optional[Path] = None  # None = CIFAR-10, file/dir = auto-detect
    output_base: Path = field(default_factory=lambda: Path("./out/eval/proc"))
    batch_size: int = 256  # Will be auto-detected if not specified
    max_batch_size: int = 2048  # Higher limit for inference
    num_workers: int = 8
    top_k: int = 5
    auto_batch_size: bool = True

    # Derived fields
    model_dir: Path = field(init=False)
    config_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    mode: str = field(init=False)  # "cifar10", "image", or "directory"

    def __post_init__(self):
        # Validate checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Derive model directory and config path
        self.model_dir = self.checkpoint_path.parent
        self.config_path = self.model_dir / "config.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        # Determine mode
        if self.input_path is None:
            self.mode = "cifar10"
        elif self.input_path.is_file():
            self.mode = "image"
        elif self.input_path.is_dir():
            self.mode = "directory"
        else:
            raise ValueError(f"Input path does not exist: {self.input_path}")

        # Create output directory with auto-incremented run_id
        self.output_dir = self._get_next_run_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_run_dir(self) -> Path:
        """Auto-increment run_id for output directory"""
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Extract model_id from model_dir
        model_id = self.model_dir.name

        # Find existing runs for this model
        existing_runs = [
            d
            for d in self.output_base.iterdir()
            if d.is_dir() and d.name.startswith(f"{model_id}-")
        ]

        if not existing_runs:
            run_id = 1
        else:
            # Extract run numbers
            run_nums = []
            for d in existing_runs:
                try:
                    num = int(d.name.split("-")[1])
                    run_nums.append(num)
                except (IndexError, ValueError):
                    continue
            run_id = max(run_nums, default=0) + 1

        return self.output_base / f"{model_id}-{run_id}"

    @classmethod
    def from_args(cls) -> InferenceConfig:
        parser = argparse.ArgumentParser(
            description="LCNet Inference Script",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Infer on CIFAR-10 validation set
  python -m src.inference --checkpoint ./out/models/10/best_model.pt

  # Infer on a single image
  python -m src.inference --checkpoint ./out/models/10/best_model.pt --input ./test.jpg

  # Infer on all images in a directory
  python -m src.inference --checkpoint ./out/models/10/best_model.pt --input ./test_images/
            """,
        )

        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to checkpoint file (e.g., ./out/models/10/best_model.pt or ./out/models/10/checkpoint_epoch_100.pt)",
        )
        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="Input path (image file or directory). If not specified, uses CIFAR-10 validation set.",
        )
        parser.add_argument(
            "--output-base",
            type=str,
            default="./out/eval/proc",
            help="Base directory for inference outputs",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=256,
            help="Batch size for inference (will be auto-detected if --auto-batch-size is enabled)",
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=2048,
            help="Maximum batch size for auto-detection",
        )
        parser.add_argument(
            "--num-workers", type=int, default=8, help="Number of data loader workers"
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of top predictions to calculate accuracy for",
        )
        parser.add_argument(
            "--no-auto-batch",
            action="store_true",
            help="Disable auto batch size detection",
        )

        args = parser.parse_args()

        return cls(
            checkpoint_path=Path(args.checkpoint),
            input_path=Path(args.input) if args.input else None,
            output_base=Path(args.output_base),
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
            auto_batch_size=not args.no_auto_batch,
        )


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
        parser.add_argument(
            "--use-kan", action="store_true", help="Use KAN activation in output_trans"
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
            model=ModelConfig(variant=args.variant, image_size=args.image_size, use_kan=args.use_kan),
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

        # [向後兼容] 為舊配置文件添加缺失的參數
        model_config = config_dict.get("model", {})
        if "use_kan" not in model_config:
            model_config["use_kan"] = False  # 舊配置預設不使用 KAN

        return cls(
            model=ModelConfig(**filter_init_fields(ModelConfig, model_config)),
            training=TrainingConfig(**filter_init_fields(TrainingConfig, config_dict.get("training", {}))),
            data=DataConfig(**filter_init_fields(DataConfig, config_dict.get("data", {}))),
            output=OutputConfig(**filter_init_fields(OutputConfig, config_dict.get("output", {}))),
        )