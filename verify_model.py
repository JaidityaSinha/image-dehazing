import torch

from models.unet import UNet

model = UNet()

x = torch.randn(1, 3, 256, 256)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)