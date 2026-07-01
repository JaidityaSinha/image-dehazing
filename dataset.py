import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, CLEAR_EXT


class DehazeDataset(Dataset):

    def __init__(self, hazy_dir, clear_dir):

        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir

        self.images = sorted(
            image
            for image in os.listdir(hazy_dir)
            if image.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]

        clear_name = image_name.split("_")[0] + CLEAR_EXT

        hazy_path = os.path.join(self.hazy_dir, image_name)
        clear_path = os.path.join(self.clear_dir, clear_name)

        hazy_image = Image.open(hazy_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        hazy_image = self.transform(hazy_image)
        clear_image = self.transform(clear_image)

        return hazy_image, clear_image