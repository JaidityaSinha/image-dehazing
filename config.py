# config.py

IMAGE_SIZE = 256

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

EPOCHS = 50


TRAIN_HAZY_DIR = "data/train/hazy"
TRAIN_CLEAR_DIR = "data/train/clear"

VAL_HAZY_DIR = "data/val/hazy"
VAL_CLEAR_DIR = "data/val/clear"

TEST_HAZY_DIR = "data/test/hazy"
TEST_CLEAR_DIR = "data/test/clear"

MODEL_PATH = "outputs/unet_its.pth"
MODEL_PATH_ATTENTION = "outputs/unet_attention_its.pth"