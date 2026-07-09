import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution: a depthwise conv (one filter per input
    channel, groups=in_channels) followed by a pointwise 1x1 conv to mix
    channels. Approximates a standard conv with far fewer parameters.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConvDW(nn.Module):
    """Two depthwise separable convolutions in sequence, replacing the
    standard DoubleConv block used in the baseline UNet."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableUNet(nn.Module):
    """
    UNet with the same encoder-decoder-skip-connection structure as the
    baseline, but every DoubleConv block uses depthwise separable
    convolutions instead of standard convolutions.
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConvDW(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvDW(features[-1], features[-1] * 2)

        # Decoder
        reversed_features = features[::-1]
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConvDW(feature * 2, feature))
            prev_channels = feature

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[idx]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = decoder(x)

        x = self.final_conv(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    # quick sanity check + parameter count
    model = DepthwiseSeparableUNet()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")