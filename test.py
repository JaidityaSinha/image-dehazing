import os
import torch
from PIL import Image
from torchvision import transforms

from config import (
    IMAGE_SIZE, TEST_HAZY_DIR, TEST_CLEAR_DIR, MODEL_PATH,
    MODEL_NAME, DATASET_NAME, CLEAR_EXT, OHAZE_TEST_HAZY, OHAZE_TEST_CLEAR
)


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
    elif MODEL_NAME == "depthwise_separable":
        from models.unet_depthwise import DepthwiseSeparableUNet
        return DepthwiseSeparableUNet()
    elif MODEL_NAME == "depthwise_ca_skip":
        from models.unet_depthwise_ca_skip import DepthwiseSeparableUNetCASkip
        return DepthwiseSeparableUNetCASkip()
    elif MODEL_NAME == "depthwise_ca_skip_fft":
        from models.unet_depthwise_ca_skip_fft import DepthwiseSeparableUNetCASkipFFT
        return DepthwiseSeparableUNetCASkipFFT()
    elif MODEL_NAME == "depthwise_channel_attention":
        from models.unet_depthwise_channel_attention import DepthwiseSeparableChannelAttentionUNet
        return DepthwiseSeparableChannelAttentionUNet()
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")


def get_clear_filename(hazy_filename):
    if DATASET_NAME == "its":
        # e.g. "1_1_0.90179.png" -> "1.png"
        base_id = hazy_filename.split("_")[0]
        return f"{base_id}{CLEAR_EXT}"
    elif DATASET_NAME == "reside6k":
        # hazy and clear share identical filenames
        return hazy_filename
    elif DATASET_NAME == "ohaze":
        # e.g. "oh41_hazy.png" -> "oh41.png"
        base_id = hazy_filename.rsplit("_hazy", 1)[0]
        return f"{base_id}{CLEAR_EXT}"
    else:
        raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")


def get_test_dirs():
    if DATASET_NAME == "ohaze":
        return OHAZE_TEST_HAZY, OHAZE_TEST_CLEAR
    return TEST_HAZY_DIR, TEST_CLEAR_DIR


def get_sample_files(hazy_dir, n=5):
    """Pick n visually distinct sample images.
    For ITS, multiple hazy files can share the same base scene at different
    haze levels (e.g. 1_1_0.9.png, 1_2_0.8.png) -- only the first variant
    per base id is kept so samples show different scenes, not haze levels.
    For RESIDE-6K and O-HAZE, filenames are already unique per scene.
    """
    all_files = sorted(os.listdir(hazy_dir))

    if DATASET_NAME == "its":
        seen_ids = set()
        samples = []
        for fname in all_files:
            base_id = fname.split("_")[0]
            if base_id not in seen_ids:
                seen_ids.add(base_id)
                samples.append(fname)
            if len(samples) == n:
                break
        return samples
    else:
        return all_files[:n]


def test(hazy_path, clear_path, output_path, model, device, transform):
    hazy_image = Image.open(hazy_path).convert("RGB")
    hazy_tensor = transform(hazy_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(hazy_tensor)

    output = output.squeeze(0).cpu()
    output = torch.clamp(output, 0, 1)
    output_image = transforms.ToPILImage()(output)

    clear_image = Image.open(clear_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    hazy_resized = hazy_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    combined = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
    combined.paste(hazy_resized, (0, 0))
    combined.paste(output_image, (IMAGE_SIZE, 0))
    combined.paste(clear_image, (IMAGE_SIZE * 2, 0))
    combined.save(output_path)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    device = torch.device("cpu")  # avoids competing with any in-progress training run

    print(f"Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
    print(f"Loading weights from: {MODEL_PATH}")

    model = get_model().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    output_dir = f"outputs/comparison_{MODEL_NAME}_{DATASET_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    hazy_dir, clear_dir = get_test_dirs()
    sample_files = get_sample_files(hazy_dir, n=5)

    for fname in sample_files:
        hazy_path = os.path.join(hazy_dir, fname)
        clear_fname = get_clear_filename(fname)
        clear_path = os.path.join(clear_dir, clear_fname)

        if not os.path.exists(clear_path):
            print(f"Skipping {fname}: no matching clear image at {clear_path}")
            continue

        out_path = os.path.join(output_dir, f"compare_{fname}")
        test(hazy_path, clear_path, out_path, model, device, transform)