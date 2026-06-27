import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from dataset import DehazeDataset
from models.unet import UNet


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DehazeDataset(
        TRAIN_HAZY_DIR,
        TRAIN_CLEAR_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = UNet()
    model.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")

    model.train()

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        for hazy_images, clear_images in train_loader:

            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)

            predicted_images = model(hazy_images)

            loss = criterion(predicted_images, clear_images)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    print("Training completed successfully!")


if __name__ == "__main__":
    train()