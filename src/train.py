import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.cifar import get_dataloaders
from .models.lcnet import LCNet
from .utils.config import Config
from .utils.logging import append_metrics_csv, get_logger
from .utils.system_info import auto_batch_size

logger = get_logger(__name__)


def train_one_epoch(
    model: LCNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: LCNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    # 1. Load Config
    config = Config.from_args()
    config.resolve_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Auto Batch Size Detection
    if config.training.auto_batch_size and device.type == "cuda":
        logger.info("Detecting optimal batch size...")

        def build_model_fn():
            return LCNet(
                num_classes=config.model.num_classes, variant=config.model.variant
            )

        optimal_batch = auto_batch_size(
            device=device,
            build_model=build_model_fn,
            sample_shape=(1, 3, 32, 32),  # CIFAR-10 shape
            base_batch=config.training.batch_size,  # Start from user provided or default 128
            max_batch=4096,
        )
        logger.info(f"Optimal batch size detected: {optimal_batch}")
        config.training.batch_size = optimal_batch

    # 3. Save Config
    config.save_yaml()

    # 4. DataLoaders
    train_loader, val_loader = get_dataloaders(config)

    # 5. Model Setup
    model = LCNet(
        num_classes=config.model.num_classes, variant=config.model.variant
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if config.training.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    else:
        scheduler = None

    # 6. Logging Setup
    log_path = config.output.model_dir / "metrics.csv"

    best_acc = 0.0
    val_acc = 0.0

    # 7. Training Loop
    logger.info("Starting training...")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        if epoch % config.training.eval_interval == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

            # Log
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}% | LR={current_lr:.6f}"
            )

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
            append_metrics_csv(log_path, metrics)

            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(), config.output.model_dir / "best_model.pt"
                )
                logger.info(f"New best model saved! ({best_acc:.2f}%)")

        if scheduler:
            scheduler.step()

        # Save Checkpoint
        if epoch % config.training.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "acc": val_acc if "val_acc" in locals() else 0.0,
                },
                config.output.model_dir / f"checkpoint_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    main()
