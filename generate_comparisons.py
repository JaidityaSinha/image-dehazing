import torch
import os
import shutil
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(r'C:\Users\jaidi\PycharmProjects\image-dehazing')

from config import IMAGE_SIZE, MODEL_PATH, MODEL_PATH_ATTENTION, TEST_HAZY_DIR, TEST_CLEAR_DIR
from models.unet import UNet
from models.unet_attention import AttentionUNet

# Pick 5 test images
TEST_IMAGES = [
    "1400_1.png",
    "1400_3.png",
    "1401_1.png",
    "1402_1.png",
    "1403_1.png",
]

OUTPUT_DIR = "outputs/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load both models
baseline = UNet().to(device)
baseline.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
baseline.eval()

attention = AttentionUNet().to(device)
attention.load_state_dict(torch.load(MODEL_PATH_ATTENTION, map_location=device, weights_only=True))
attention.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

for idx, image_name in enumerate(TEST_IMAGES, 1):
    hazy_path = os.path.join(TEST_HAZY_DIR, image_name)
    clear_name = image_name.split("_")[0] + ".png"
    clear_path = os.path.join(TEST_CLEAR_DIR, clear_name)

    if not os.path.exists(hazy_path):
        print(f"Skipping {image_name} - not found")
        continue

    # Save hazy image
    hazy_img = Image.open(hazy_path).convert("RGB")
    hazy_img.resize((256, 256)).save(f"{OUTPUT_DIR}/img{idx}_hazy.png")

    # Save ground truth
    if os.path.exists(clear_path):
        clear_img = Image.open(clear_path).convert("RGB")
        clear_img.resize((256, 256)).save(f"{OUTPUT_DIR}/img{idx}_clear.png")

    # Run baseline UNet
    input_tensor = transform(hazy_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = baseline(input_tensor)
    output = output.squeeze(0).cpu().clamp(0, 1)
    transforms.ToPILImage()(output).save(f"{OUTPUT_DIR}/img{idx}_baseline.png")

    # Run attention UNet
    with torch.no_grad():
        output = attention(input_tensor)
    output = output.squeeze(0).cpu().clamp(0, 1)
    transforms.ToPILImage()(output).save(f"{OUTPUT_DIR}/img{idx}_attention.png")

    print(f"Done: image {idx} ({image_name})")

print(f"\nAll comparison images saved to {OUTPUT_DIR}/")