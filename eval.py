import os
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

from config import IMAGE_SIZE, MODEL_PATH, MODEL_PATH_6K,TEST_HAZY_DIR, TEST_CLEAR_DIR, MODEL_PATH_CHANNEL
from models.unet import UNet


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH_6K, map_location=device, weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    hazy_images = sorted(
        f for f in os.listdir(TEST_HAZY_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for image_name in hazy_images[:5]:
        clear_name = image_name
        clear_path = os.path.join(TEST_CLEAR_DIR, clear_name)
        print(f"{image_name} -> {clear_name} | exists: {os.path.exists(clear_path)}")

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for image_name in hazy_images:

        clear_name = image_name

        hazy_path = os.path.join(TEST_HAZY_DIR, image_name)
        clear_path = os.path.join(TEST_CLEAR_DIR, clear_name)

        if not os.path.exists(clear_path):
            continue

        hazy_image = Image.open(hazy_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        input_tensor = transform(hazy_image).unsqueeze(0).to(device)
        clear_tensor = transform(clear_image)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target = clear_tensor.numpy().transpose(1, 2, 0)

        output = np.clip(output, 0, 1)
        target = np.clip(target, 0, 1)

        p = psnr(target, output, data_range=1.0)
        s = ssim(target, output, data_range=1.0, channel_axis=2)

        total_psnr += p
        total_ssim += s
        count += 1

        if count % 100 == 0:
            print(f"Processed {count}/{len(hazy_images)} | Avg PSNR: {total_psnr/count:.4f} | Avg SSIM: {total_ssim/count:.4f}")

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\nResults on RESIDE_6K Test Set ({count} images)")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    evaluate()