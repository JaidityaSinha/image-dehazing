import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Standard double convolution block (no attention) -- used in the
    encoder and bottleneck for the decoder-only ablation variant."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention block.
    reduction_ratio controls the SE bottleneck size (C / reduction_ratio)."""

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


class DoubleConvWithCA(nn.Module):
    """Double convolution followed by a channel attention block."""

    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.ca = ChannelAttention(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = self.conv(x)
        return self.ca(x)


class ChannelAttentionUNetDecoderOnly(nn.Module):
    """
    UNet with channel attention applied ONLY in the decoder blocks.
    The encoder and bottleneck use plain DoubleConv (no attention),
    isolating the effect of decoder-only channel attention for the
    ablation study. reduction_ratio controls the SE bottleneck size
    used in every decoder channel attention block (e.g. 8, 16, 32).
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512), reduction_ratio=16):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder -- plain DoubleConv, no attention
        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        # Bottleneck -- plain DoubleConv, no attention
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder -- DoubleConv + Channel Attention
        reversed_features = features[::-1]
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConvWithCA(feature * 2, feature, reduction_ratio=reduction_ratio))
            prev_channels = feature

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
    for ratio in (8, 16, 32):
        model = ChannelAttentionUNetDecoderOnly(reduction_ratio=ratio)
        dummy = torch.randn(1, 3, 256, 256)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"SE ratio C/{ratio}: output shape {tuple(out.shape)}, params: {params:,}")