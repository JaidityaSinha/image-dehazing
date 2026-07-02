# config.py

IMAGE_SIZE = 256

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

EPOCHS = 70


TRAIN_HAZY_DIR = "data/train/hazy"
TRAIN_CLEAR_DIR = "data/train/clear"

VAL_HAZY_DIR = "data/val/hazy"
VAL_CLEAR_DIR = "data/val/clear"

TEST_HAZY_DIR = "data/test/hazy"
TEST_CLEAR_DIR = "data/test/clear"

MODEL_PATH = "outputs/unet_its.pth"
MODEL_PATH_6K = 'outputs/unet_6k.pth'

MODEL_PATH_CHANNEL = "outputs/unet_channel_its.pth"
MODEL_PATH_CHANNEL_6K = "outputs/unet_channel_6k.pth"

CLEAR_EXT = ".png"