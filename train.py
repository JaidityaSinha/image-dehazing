import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import (
    TRAIN_HAZY_DIR, TRAIN_CLEAR_DIR, BATCH_SIZE, LEARNING_RATE,
    MODEL_PATH, MODEL_NAME, DATASET_NAME, LOSS_TYPE,
    VALIDATION_SPLIT, EARLY_STOP_PATIENCE, MAX_EPOCHS
)
from dataset import DehazeDataset

MIN_DELTA = 1e-4


def get_model():
    if MODEL_NAME == "baseline":
        from models.unet import UNet
        return UNet()
    elif MODEL_NAME == "attention_gate":
        from models.unet_attention import AttentionUNet
        return AttentionUNet()
    elif MODEL_NAME == "channel_attention":
        from models.unet_channel_attention import ChannelAttentionUNet
        return ChannelAttentionUNet()
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_criterion():
    if LOSS_TYPE == "mse":
        return nn.MSELoss()
    elif LOSS_TYPE == "ssim":
        from pytorch_msssim import ssim

        def ssim_loss(pred, target):
            return 1 - ssim(pred, target, data_range=1.0, size_average=True)

        return ssim_loss
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
    print(f"Saving best checkpoint to: {MODEL_PATH}")

    full_dataset = DehazeDataset(
        TRAIN_HAZY_DIR,
        TRAIN_CLEAR_DIR
    )

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    # fixed generator seed so the same split is reproducible across runs
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Train samples: {train_size} | Validation samples: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = get_model().to(device)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print(f"Loss function: {LOSS_TYPE}")

    criterion = get_criterion()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):

        # ---- Training ----
        model.train()
        train_loss = 0.0

        print(f"\nEpoch [{epoch + 1}/{MAX_EPOCHS}]")

        for batch_idx, (hazy_images, clear_images) in enumerate(train_loader):
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)

            optimizer.zero_grad()
            predicted_images = model(hazy_images)
            loss = criterion(predicted_images, clear_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hazy_images, clear_images in val_loader:
                hazy_images = hazy_images.to(device)
                clear_images = clear_images.to(device)
                predicted_images = model(hazy_images)
                val_loss += criterion(predicted_images, clear_images).item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        # ---- Early stopping / checkpointing on VALIDATION loss ----
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("New best model saved! (validation loss improved)")
        else:
            epochs_no_improve += 1

        print(
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"| Best Val Loss: {best_val_loss:.4f} | No improve: {epochs_no_improve}/{EARLY_STOP_PATIENCE} "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} "
                  f"(no validation improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()