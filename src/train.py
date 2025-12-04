from pathlib import Path
import numpy as np  # [新增] 用於 Beta 分布採樣
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


# --- [新增] Mixup Helper Functions ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Loss calculation for mixup'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# ------------------------------------


def train_one_epoch(
    model: LCNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    mixup_alpha: float = 0.0,  # [新增] 接收 mixup 參數
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # --- [修改] Mixup Logic ---
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # -------------------------

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # --- [修改] 準確率計算 (針對 Mixup 做加權) ---
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        if mixup_alpha > 0:
            # Mixup 時的準確率通常是參考用，這裡計算加權後的正確率
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()
        # -------------------------------------------

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
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            logger.warning(f"NaN detected in model outputs during validation at epoch {epoch}")
            # Skip this batch
            continue
            
        loss = criterion(outputs, targets)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            logger.warning(f"NaN detected in loss during validation at epoch {epoch}")
            continue

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else float('nan')
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    # 1. Load Config
    config = Config.from_args()

    # If resuming, extract model_id from checkpoint path
    if config.training.resume:
        resume_path = Path(config.training.resume)
        # Extract model_id from checkpoint path
        model_id = int(resume_path.parent.name)
        config.output.model_id = model_id
        logger.info(f"Resuming training from model_id: {model_id}")

    config.resolve_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # [新增] 讀取 Mixup 參數 (預設為 0.0 若 config 中沒有該欄位)
    mixup_alpha = getattr(config.training, 'mixup_alpha', 0.0)
    logger.info(f"Mixup Alpha: {mixup_alpha}")

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
            max_batch=config.training.max_batch_size,
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
    start_epoch = 1

    # Resume from checkpoint
    if config.training.resume:
        resume_path = Path(config.training.resume)
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            sched_state = checkpoint["scheduler_state_dict"]
            # Check if previous run finished its schedule
            last_epoch = sched_state.get("last_epoch", -1)
            t_max = sched_state.get("T_max", -1)

            # If the previous schedule was completed, we start a new one
            if t_max != -1 and last_epoch >= t_max - 1:
                logger.info(
                    "Previous scheduler finished. Starting new schedule and resetting LR."
                )
                # Reset optimizer LR to initial config value
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.training.lr
                # Do NOT load scheduler state, so it starts fresh
            else:
                scheduler.load_state_dict(sched_state)

        start_epoch = checkpoint["epoch"] + 1
        if "acc" in checkpoint:
            best_acc = checkpoint["acc"]

        logger.info(
            f"Resumed training from epoch {start_epoch}. Best acc so far: {best_acc:.2f}%"
        )

    # 7. Training Loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.training.epochs + 1):
        # [修改] 傳入 mixup_alpha
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=mixup_alpha
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
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acc": val_acc if "val_acc" in locals() else 0.0,
            }
            if scheduler:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(
                checkpoint,
                config.output.model_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    logger.info(f"Training completed. Saved at {config.output.model_dir}")


if __name__ == "__main__":
    main()