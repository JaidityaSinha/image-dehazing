import os
import random
import torch
from PIL import Image
from torchvision import transforms

from models.unet import UNet
from models.unet_channel_attention import ChannelAttentionUNet
from models.unet_depthwise import DepthwiseSeparableUNet

IMAGE_SIZE = 256
NUM_SAMPLES = 3
SEED = 42

# ----------------------------------------------------------------------
# Dataset paths -- update these if your folder layout differs.
# Both datasets are referenced directly so no manual swapping of
# data/train or data/test is needed to run this script.
# ----------------------------------------------------------------------
ITS_TEST_HAZY = r"data\test\hazy"
ITS_TEST_CLEAR = r"data\test\clear"

RESIDE6K_TEST_HAZY = r"reside6k\RESIDE-6K\test\hazy"
RESIDE6K_TEST_CLEAR = r"reside6k\RESIDE-6K\test\clear"

OUTPUT_ROOT = "outputs/comparisons"

# ----------------------------------------------------------------------
# Every model/dataset/loss checkpoint trained so far.
# Add new rows here as new experiments (e.g. ablation variants) complete.
# ----------------------------------------------------------------------
CHECKPOINTS = {
    ("baseline", "its", "mse"): "outputs/unet_its.pth",
    ("baseline", "its", "ssim"): "outputs/unet_its_ssim.pth",
    ("channel_attention", "its", "mse"): "outputs/unet_channel_its.pth",
    ("channel_attention", "its", "ssim"): "outputs/unet_channel_its_ssim.pth",
    ("depthwise_separable", "its", "mse"): "outputs/unet_depthwise_its.pth",
    ("depthwise_separable", "its", "ssim"): "outputs/unet_depthwise_its_ssim.pth",

    ("baseline", "reside6k", "mse"): "outputs/unet_6k.pth",
    ("baseline", "reside6k", "ssim"): "outputs/unet_6k_ssim.pth",
    ("channel_attention", "reside6k", "mse"): "outputs/unet_channel_6k.pth",
    ("channel_attention", "reside6k", "ssim"): "outputs/unet_channel_6k_ssim.pth",
    ("depthwise_separable", "reside6k", "mse"): "outputs/unet_depthwise_6k.pth",
    ("depthwise_separable", "reside6k", "ssim"): "outputs/unet_depthwise_6k_ssim.pth",
}


def get_model(model_name):
    if model_name == "baseline":
        return UNet()
    elif model_name == "channel_attention":
        return ChannelAttentionUNet()
    elif model_name == "depthwise_separable":
        return DepthwiseSeparableUNet()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def get_clear_filename(hazy_filename, dataset_name):
    if dataset_name == "its":
        # e.g. "1_1_0.90179.png" -> "1.png"
        base_id = hazy_filename.split("_")[0]
        return f"{base_id}.png"
    elif dataset_name == "reside6k":
        # hazy and clear share identical filenames
        return hazy_filename
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


def get_random_samples(hazy_dir, clear_dir, dataset_name, n, seed):
    """Pick n random samples that have a matching clear image.
    For ITS, sampling is done by unique base scene id so the same
    scene isn't picked multiple times at different haze levels.
    Uses a fixed seed so the SAME 3 images are picked across every
    model/loss combo for a given dataset, making comparisons fair."""
    all_files = sorted(os.listdir(hazy_dir))

    if dataset_name == "its":
        scenes = {}
        for fname in all_files:
            base_id = fname.split("_")[0]
            scenes.setdefault(base_id, []).append(fname)
        candidates = [files[0] for files in scenes.values()]
    else:
        candidates = all_files

    valid = [
        f for f in candidates
        if os.path.exists(os.path.join(clear_dir, get_clear_filename(f, dataset_name)))
    ]

    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:n]


def generate_comparison(hazy_path, clear_path, output_path, model, device, transform):
    hazy_image = Image.open(hazy_path).convert("RGB")
    hazy_tensor = transform(hazy_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(hazy_tensor)

    output = output.squeeze(0).cpu()
    output = torch.clamp(output, 0, 1)
    output_image = transforms.ToPILImage()(output)

    clear_image = Image.open(clear_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    hazy_resized = hazy_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # 768 x 256: hazy | dehazed | ground truth, side by side
    combined = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
    combined.paste(hazy_resized, (0, 0))
    combined.paste(output_image, (IMAGE_SIZE, 0))
    combined.paste(clear_image, (IMAGE_SIZE * 2, 0))
    combined.save(output_path)


def run():
    device = torch.device("cpu")  # avoids competing with any in-progress training run
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset_paths = {
        "its": (ITS_TEST_HAZY, ITS_TEST_CLEAR),
        "reside6k": (RESIDE6K_TEST_HAZY, RESIDE6K_TEST_CLEAR),
    }

    # cache sample filenames per dataset so every model uses the SAME
    # 3 images for that dataset, making cross-model comparison fair
    sample_cache = {}

    completed = 0
    skipped = 0

    for (model_name, dataset_name, loss_type), ckpt_path in CHECKPOINTS.items():
        print(f"\n=== {model_name} | {dataset_name} | {loss_type} ===")

        if not os.path.exists(ckpt_path):
            print(f"Skipping: checkpoint not found at {ckpt_path}")
            skipped += 1
            continue

        hazy_dir, clear_dir = dataset_paths[dataset_name]
        if not os.path.exists(hazy_dir):
            print(f"Skipping: test hazy dir not found at {hazy_dir}")
            skipped += 1
            continue

        model = get_model(model_name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        if dataset_name not in sample_cache:
            sample_cache[dataset_name] = get_random_samples(
                hazy_dir, clear_dir, dataset_name, NUM_SAMPLES, SEED
            )
        samples = sample_cache[dataset_name]

        out_dir = os.path.join(OUTPUT_ROOT, f"{model_name}_{dataset_name}_{loss_type}")
        os.makedirs(out_dir, exist_ok=True)

        for fname in samples:
            hazy_path = os.path.join(hazy_dir, fname)
            clear_fname = get_clear_filename(fname, dataset_name)
            clear_path = os.path.join(clear_dir, clear_fname)

            out_path = os.path.join(out_dir, f"compare_{fname}")
            generate_comparison(hazy_path, clear_path, out_path, model, device, transform)
            print(f"Saved {out_path}")

        completed += 1

    print(f"\nDone. {completed} model/dataset/loss combinations processed, {skipped} skipped (checkpoint or data not found).")


if __name__ == "__main__":
    run()