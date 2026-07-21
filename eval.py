import os
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

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


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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

    hazy_dir, clear_dir = get_test_dirs()
    hazy_files = sorted(os.listdir(hazy_dir))

    psnr_scores = []
    ssim_scores = []
    skipped = 0

    for idx, fname in enumerate(hazy_files):
        hazy_path = os.path.join(hazy_dir, fname)
        clear_fname = get_clear_filename(fname)
        clear_path = os.path.join(clear_dir, clear_fname)

        if not os.path.exists(clear_path):
            skipped += 1
            continue

        hazy_image = Image.open(hazy_path).convert("RGB")
        hazy_tensor = transform(hazy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(hazy_tensor)

        output = output.squeeze(0).cpu()
        output = torch.clamp(output, 0, 1)
        output_np = output.permute(1, 2, 0).numpy()

        clear_image = Image.open(clear_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        clear_np = np.array(clear_image).astype(np.float32) / 255.0

        score_psnr = psnr(clear_np, output_np, data_range=1.0)
        score_ssim = ssim(clear_np, output_np, data_range=1.0, channel_axis=2)

        psnr_scores.append(score_psnr)
        ssim_scores.append(score_ssim)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(hazy_files)} images...")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"\n--- Evaluation Results ---")
    print(f"Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
    print(f"Images evaluated: {len(psnr_scores)} (skipped: {skipped})")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    evaluate()