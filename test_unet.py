import os
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, TEST_HAZY_DIR, TEST_CLEAR_DIR
from models.unet import UNet

MODEL_PATH_ITS = "outputs/unet_its.pth"
CLEAR_EXT_ITS = ".png"


def get_clear_filename(hazy_filename):
    # e.g. "1_1_0.90179.png" -> "1.png"
    base_id = hazy_filename.split("_")[0]
    return f"{base_id}{CLEAR_EXT_ITS}"


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
    device = torch.device("cpu")

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH_ITS, map_location=device, weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    output_dir = "outputs/its comparison baseline"
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(os.listdir(TEST_HAZY_DIR))
    seen_ids = set()
    sample_files = []
    for fname in all_files:
        base_id = fname.split("_")[0]
        if base_id not in seen_ids:
            seen_ids.add(base_id)
            sample_files.append(fname)
        if len(sample_files) == 5:
            break

    for fname in sample_files:
        hazy_path = os.path.join(TEST_HAZY_DIR, fname)
        clear_fname = get_clear_filename(fname)
        clear_path = os.path.join(TEST_CLEAR_DIR, clear_fname)

        if not os.path.exists(clear_path):
            print(f"Skipping {fname}: no matching clear image at {clear_path}")
            continue

        out_path = os.path.join(output_dir, f"compare_{fname}")
        test(hazy_path, clear_path, out_path, model, device, transform)