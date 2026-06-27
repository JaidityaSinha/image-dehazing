from dataset import DehazeDataset
from config import TRAIN_HAZY_DIR, TRAIN_CLEAR_DIR

dataset = DehazeDataset(
    TRAIN_HAZY_DIR,
    TRAIN_CLEAR_DIR
)

print(f"Dataset size: {len(dataset)}")

hazy, clear = dataset[0]

print("Hazy image shape :", hazy.shape)
print("Clear image shape:", clear.shape)