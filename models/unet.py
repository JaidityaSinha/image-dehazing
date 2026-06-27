import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.input = DoubleConv(3, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Bottleneck
        self.bottleneck = DoubleConv(1024, 2048)

        # Decoder
        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)

        # Final Output Layer
        self.output = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1 = self.input(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        # Bottleneck
        x6 = self.bottleneck(x5)

        # Decoder
        x = self.up1(x6, x5)

        x = self.up2(x, x4)

        x = self.up3(x, x3)

        x = self.up4(x, x2)

        return self.output(x)