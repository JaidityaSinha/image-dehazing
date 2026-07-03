import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from dataset import DehazeDataset
from models.unet_channel_attention import ChannelAttentionUNet


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DehazeDataset(
        TRAIN_HAZY_DIR,
        TRAIN_CLEAR_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = ChannelAttentionUNet().to(device)

    criterion = nn.MSELoss()

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

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        for batch_idx, (hazy_images, clear_images) in enumerate(train_loader):

            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)

            optimizer.zero_grad()

            predicted_images = model(hazy_images)

            loss = criterion(predicted_images, clear_images)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = epoch_loss / len(train_loader)

        scheduler.step(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), MODEL_PATH_CHANNEL_6K)
            print("New best model saved!")

        print(f"Epoch Loss: {average_loss:.4f} | Best Loss: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()