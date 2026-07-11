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


class ChannelAttention(nn.Module):
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


class DoubleConvDWWithCA(nn.Module):
    """Two depthwise separable convolutions followed by a channel
    attention block. Used in every encoder, bottleneck, and decoder
    stage in this variant -- combining both modifications requested by
    the mentor (depthwise separable convs + channel attention) into a
    single architecture, unlike the two skip-only variants above."""

    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        self.ca = ChannelAttention(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = self.conv(x)
        return self.ca(x)


class DepthwiseSeparableChannelAttentionUNet(nn.Module):
    """
    UNet combining both architecture modifications: every convolution
    is depthwise separable (as in the Depthwise Separable UNet), AND
    every encoder, bottleneck, and decoder block includes a channel
    attention module (as in the full Channel Attention UNet). This is
    the "both changes combined" variant, distinct from the two
    skip-connection-only variants which keep channel attention isolated
    to the skip path.
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512), reduction_ratio=16):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConvDWWithCA(prev_channels, feature, reduction_ratio=reduction_ratio))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        self.bottleneck = DoubleConvDWWithCA(features[-1], features[-1] * 2, reduction_ratio=reduction_ratio)

        reversed_features = features[::-1]
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConvDWWithCA(feature * 2, feature, reduction_ratio=reduction_ratio))
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
    model = DepthwiseSeparableChannelAttentionUNet()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {tuple(out.shape)}, params: {params:,}")