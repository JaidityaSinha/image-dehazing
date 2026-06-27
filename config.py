# config.py

IMAGE_SIZE = 256

BATCH_SIZE = 8

LEARNING_RATE = 0.001

EPOCHS = 20

TRAIN_HAZY_DIR = "data/train/hazy"
TRAIN_CLEAR_DIR = "data/train/clear"

VAL_HAZY_DIR = "data/val/hazy"
VAL_CLEAR_DIR = "data/val/clear"

TEST_HAZY_DIR = "data/test/hazy"
TEST_CLEAR_DIR = "data/test/clear"

MODEL_PATH = "outputs/unet_model.pth"