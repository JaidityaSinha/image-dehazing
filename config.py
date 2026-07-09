IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# ----------------------------------------------------------------------
# Model selection
# Options: "baseline" | "attention_gate" | "channel_attention"
# ----------------------------------------------------------------------
MODEL_NAME = "channel_attention"

# ----------------------------------------------------------------------
# Dataset selection
# Options: "its" | "reside6k"
# ----------------------------------------------------------------------
DATASET_NAME = "reside6k"

# ----------------------------------------------------------------------
# Epochs per model/dataset combo (based on what worked during experiments)
# ----------------------------------------------------------------------
EPOCHS_LOOKUP = {
    ("baseline", "its"): 30,
    ("attention_gate", "its"): 50,
    ("channel_attention", "its"): 35,
    ("baseline", "reside6k"): 50,
    ("channel_attention", "reside6k"): 60,
}
EPOCHS = EPOCHS_LOOKUP.get((MODEL_NAME, DATASET_NAME), 50)

# ----------------------------------------------------------------------
# Early stopping / validation
# A slice of the training set is held out each run for validation, so
# training can stop automatically instead of being watched manually.
# ----------------------------------------------------------------------
VALIDATION_SPLIT = 0.1       # fraction of training data held out for validation
EARLY_STOP_PATIENCE = 5      # stop if val loss doesn't improve for this many epochs
MAX_EPOCHS = 100             # hard ceiling regardless of EPOCHS_LOOKUP above

# ----------------------------------------------------------------------
# Loss function
# Options: "mse" | "ssim"
# ----------------------------------------------------------------------
LOSS_TYPE = "ssim"

# ----------------------------------------------------------------------
# Dataset paths
# NOTE: swap the contents of data/train and data/test between ITS and
# RESIDE-6K before running, since both datasets share the same folder
# names but have different filename conventions (see CLEAR_EXT below).
# ----------------------------------------------------------------------
TRAIN_HAZY_DIR = "data/train/hazy"
TRAIN_CLEAR_DIR = "data/train/clear"
TEST_HAZY_DIR = "data/test/hazy"
TEST_CLEAR_DIR = "data/test/clear"

# ITS: hazy "1_1_0.90179.png" -> clear "1.png"      (CLEAR_EXT = ".png")
# RESIDE-6K: hazy and clear share identical filenames (CLEAR_EXT = ".jpg")
CLEAR_EXT_LOOKUP = {
    "its": ".png",
    "reside6k": ".jpg",
}
CLEAR_EXT = CLEAR_EXT_LOOKUP[DATASET_NAME]

# ----------------------------------------------------------------------
# Model checkpoint paths (one per model/dataset combo)
# ----------------------------------------------------------------------
MODEL_PATH_LOOKUP = {
    ("baseline", "its"): "outputs/unet_its.pth",
    ("attention_gate", "its"): "outputs/unet_attention_its.pth",
    ("channel_attention", "its"): "outputs/unet_channel_its.pth",
    ("baseline", "reside6k"): "outputs/unet_6k.pth",
    ("channel_attention", "reside6k"): "outputs/unet_channel_6k.pth",
}

if LOSS_TYPE == "mse":
    MODEL_PATH = MODEL_PATH_LOOKUP[(MODEL_NAME, DATASET_NAME)]
else:
    # non-MSE loss runs (e.g. ssim) get their own suffixed checkpoint
    # so they never overwrite the original MSE-trained weights
    base_path = MODEL_PATH_LOOKUP[(MODEL_NAME, DATASET_NAME)]
    MODEL_PATH = base_path.replace(".pth", f"_{LOSS_TYPE}.pth")

# Kept for backward compatibility with older scripts that import these directly
MODEL_PATH_ATTENTION = MODEL_PATH_LOOKUP[("attention_gate", "its")]
MODEL_PATH_CHANNEL = MODEL_PATH_LOOKUP[("channel_attention", "its")]
MODEL_PATH_CHANNEL_6K = MODEL_PATH_LOOKUP[("channel_attention", "reside6k")]
MODEL_PATH_6K = MODEL_PATH_LOOKUP[("baseline", "reside6k")]