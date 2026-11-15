"""Small four-downsampling 2D U-Net; outputs raw binary logits."""
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.in_channels = in_channels
        widths = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(double_conv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[i+1], widths[i], 2, stride=2)
                                 for i in reversed(range(4))])
        self.decoders = nn.ModuleList([double_conv(2*widths[i], widths[i])
                                      for i in reversed(range(4))])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 32:
            raise ValueError("Expected BCHW input with configured channels and H,W >=32")
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.output(x)
