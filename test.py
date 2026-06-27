import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, MODEL_PATH
from models.unet import UNet


def test(image_path, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

    output = output.squeeze(0).cpu()

    output = transforms.ToPILImage()(output)

    output.save(output_path)

    print(f"Dehazed image saved to {output_path}")


if __name__ == "__main__":

    test(
        "sample_hazy.png",
        "outputs/dehazed_image.png"
    )