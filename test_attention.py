import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, MODEL_PATH_ATTENTION
from models.unet_attention import AttentionUNet


def test(image_path, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUNet().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH_ATTENTION, map_location=device, weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    output = output.squeeze(0).cpu()
    output = torch.clamp(output, 0, 1)
    output = transforms.ToPILImage()(output)
    output.save(output_path)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    test(
        "data/test/hazy/1400_1.png",
        "outputs/attention_dehazed.png"
    )