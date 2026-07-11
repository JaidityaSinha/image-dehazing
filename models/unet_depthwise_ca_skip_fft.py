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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)


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


class FrequencyUnit(nn.Module):
    """
    FFT-based frequency-domain processing block. Converts a real-valued
    feature map to the frequency domain (rfft2), learns a filter over
    the real and imaginary components jointly via a 1x1 conv, then
    converts back to the spatial domain (irfft2). Applied after channel
    attention, so this refines the already-recalibrated skip features
    using global frequency information rather than local spatial filters.
    """

    def __init__(self, channels):
        super().__init__()
        # operates on concatenated [real, imag] channels -> 2x channel count
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # forward FFT (real input -> complex output, half-spectrum for efficiency)
        x_freq = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1), complex

        # split complex tensor into real/imag, stack along channel dim
        x_real = x_freq.real
        x_imag = x_freq.imag
        x_stack = torch.cat([x_real, x_imag], dim=1)  # (B, 2C, H, W//2+1)

        # learnable filtering in frequency domain
        x_stack = self.freq_conv(x_stack)

        # split back into real/imag and reconstruct complex tensor
        x_real, x_imag = torch.chunk(x_stack, 2, dim=1)
        x_freq = torch.complex(x_real, x_imag)

        # inverse FFT back to spatial domain
        x_spatial = torch.fft.irfft2(x_freq, s=(h, w), norm="ortho")

        # residual connection: frequency-refined features added back to input
        return x + x_spatial


class DepthwiseSeparableUNetCASkipFFT(nn.Module):
    """
    Depthwise Separable UNet where skip connections are processed by
    Channel Attention followed by an FFT-based frequency-domain block
    (Channel Attention -> FFT -> filter -> inverse FFT), before being
    concatenated into the decoder. Encoder, bottleneck, and decoder
    convolutions are all plain depthwise separable convs with no
    attention -- identical to the skip-only Channel Attention variant,
    with the addition of frequency-domain refinement on the skip path.
    """

    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512), reduction_ratio=16):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.skip_attentions = nn.ModuleList()
        self.skip_freq_units = nn.ModuleList()

        prev_channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConvDW(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.skip_attentions.append(ChannelAttention(feature, reduction_ratio=reduction_ratio))
            self.skip_freq_units.append(FrequencyUnit(feature))
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

        for encoder, pool, skip_ca, skip_freq in zip(
            self.encoders, self.pools, self.skip_attentions, self.skip_freq_units
        ):
            x = encoder(x)
            skip = skip_ca(x)       # channel attention on skip tensor
            skip = skip_freq(skip)  # then FFT-based frequency refinement
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
    model = DepthwiseSeparableUNetCASkipFFT()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {tuple(out.shape)}, params: {params:,}")