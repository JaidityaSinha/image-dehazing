import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
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
    """Two depthwise separable convolutions -- no attention. Used for the
    encoder, bottleneck, and decoder feature extraction in this variant;
    channel attention is applied separately, only on the skip connections."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention block, applied here
    to skip connection feature maps before they reach the decoder."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class DepthwiseSeparableUNetCASkip(nn.Module):
    """
    Depthwise Separable UNet where channel attention is applied ONLY to
    the skip connections (between encoder and decoder), not inside the
    encoder/decoder DoubleConv blocks themselves. Encoder, bottleneck,
    and decoder all use plain depthwise separable convolutions with no
    attention; the skip connection tensors are recalibrated by a
    Channel Attention block immediately before being concatenated into
    the decoder path.
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512), reduction_ratio=16):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.skip_attentions = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConvDW(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.skip_attentions.append(ChannelAttention(feature, reduction_ratio=reduction_ratio))
            prev_channels = feature

        self.bottleneck = DoubleConvDW(features[-1], features[-1] * 2)

        reversed_features = features[::-1]
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConvDW(feature * 2, feature))
            prev_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for encoder, pool, skip_ca in zip(self.encoders, self.pools, self.skip_attentions):
            x = encoder(x)
            skip = skip_ca(x)  # channel attention applied here, on the skip tensor only
            skip_connections.append(skip)
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
    model = DepthwiseSeparableUNetCASkip()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {tuple(out.shape)}, params: {params:,}")