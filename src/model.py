import torch
from torch import nn

import math


class Conv1dSame(nn.Conv1d):
    """Copied from: https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351

    Modified it so it works with the `nn.Conv1d` layer.
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih = x.size(-1)
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=1)

        if pad_h > 0:
            x = nn.functional.pad(x, [pad_h // 2, pad_h - pad_h // 2])
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            1,
            1,
        )


class Conv2dSame(torch.nn.Conv2d):
    """Copied from: https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351"""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelShuffle1D(torch.nn.Module):
    """Copied from: https://github.com/serkansulun/pytorch-pixelshuffle1d/"""

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view(
            [batch_size, self.upscale_factor, long_channel_len, short_width]
        )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(ResidualBlock, self).__init__()

        self.conv_a = Conv1dSame(in_channels, out_channels, **kwargs)
        self.conv_b = Conv1dSame(in_channels, out_channels, **kwargs)
        self.norm1 = nn.InstanceNorm1d(out_channels)

        self.conv = Conv1dSame(out_channels, out_channels // 2, **kwargs)
        self.norm2 = nn.InstanceNorm1d(out_channels // 2)

    def forward(self, x):
        # conv1d + instancenorm1d + glu
        A = self.conv_a(x)
        A = self.norm1(A)
        B = self.conv_b(x)
        B = self.norm1(B)
        out = A * B.sigmoid()

        out = self.conv(out)
        out = self.norm2(out)
        return x + out


class DownSampleBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DownSampleBlock1d, self).__init__()

        self.conv = Conv1dSame(in_channels, out_channels, **kwargs)
        self.norm = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        A = self.conv(x)
        A = self.norm(A)
        B = self.conv(x)
        B = self.norm(B)
        return A * B.sigmoid()


class UpSampleBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(UpSampleBlock1d, self).__init__()

        self.conv = Conv1dSame(in_channels, out_channels, **kwargs)
        self.pixelsh = PixelShuffle1D(2)
        self.norm = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        A = self.conv(x)
        A = self.pixelsh(A)
        A = self.norm(A)

        B = self.conv(x)
        B = self.pixelsh(B)
        B = self.norm(B)

        return A * B.sigmoid()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_a = Conv1dSame(24, 128, kernel_size=15, stride=1)
        self.conv_b = Conv1dSame(24, 128, kernel_size=15, stride=1)

        self.ds1 = DownSampleBlock1d(128, 256, kernel_size=5, stride=2)
        self.ds2 = DownSampleBlock1d(256, 512, kernel_size=5, stride=2)

        self.us1 = UpSampleBlock1d(512, 1024, kernel_size=5, stride=1)
        self.us2 = UpSampleBlock1d(1024 // 2, 512, kernel_size=5, stride=1)

        self.rb = ResidualBlock(512, 1024, kernel_size=3, stride=1)

        self.conv = Conv1dSame(512 // 2, 24, kernel_size=15, stride=1)

    def forward(self, x):
        A = self.conv_a(x)
        B = self.conv_b(x)
        out = A * B.sigmoid()

        # downsample
        out = self.ds1(out)
        out = self.ds2(out)

        # residual blocks
        for _ in range(6):
            out = self.rb(out)

        # upsample
        out = self.us1(out)
        out = self.us2(out)

        out = self.conv(out)

        return out


class DownSampleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DownSampleBlock2d, self).__init__()

        self.conv = Conv2dSame(in_channels, out_channels, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        A = self.conv(x)
        A = self.norm(A)
        B = self.conv(x)
        B = self.norm(B)

        return A * B.sigmoid()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_a = Conv2dSame(1, 128, kernel_size=(3, 3), stride=(1, 2))
        self.conv_b = Conv2dSame(1, 128, kernel_size=(3, 3), stride=(1, 2))

        self.ds1 = DownSampleBlock2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
        self.ds2 = DownSampleBlock2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
        self.ds3 = DownSampleBlock2d(512, 1024, kernel_size=(6, 3), stride=(1, 2))

        self.linear = nn.Linear(49152, 1)

    def forward(self, x):
        A = self.conv_a(x)
        B = self.conv_b(x)
        out = A * B.sigmoid()

        # downsample
        out = self.ds1(out)
        out = self.ds2(out)
        out = self.ds3(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.sigmoid()
