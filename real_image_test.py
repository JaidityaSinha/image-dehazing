import os
import torch
from PIL import Image
from torchvision import transforms

from models.unet_channel_attention import ChannelAttentionUNet

# -------------------------------------------------
# Configuration
# -------------------------------------------------

IMAGE_SIZE = 256

IMAGE_PATH = "real_images/hazy.png"      # Your downloaded hazy image
MODEL_PATH = "outputs/unet_channel_6k_ssim.pth"

OUTPUT_PATH = "outputs/real_world/dehazed_result.png"

# -------------------------------------------------


def main():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    model = ChannelAttentionUNet().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output = output.squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output)

    comparison = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE))
    comparison.paste(image_resized, (0, 0))
    comparison.paste(output_image, (IMAGE_SIZE, 0))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    comparison.save(OUTPUT_PATH)

    print(f"Saved comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()