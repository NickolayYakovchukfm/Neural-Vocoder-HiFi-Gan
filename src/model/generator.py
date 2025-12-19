import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=dilation,
                            padding=self.get_padding(kernel_size, dilation),
                        )
                    ),
                )
            )
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=self.get_padding(kernel_size, 1),
                        )
                    ),
                )
            )

    def get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x):
        for i in range(len(self.convs) // 2):
            residual = x
            x = self.convs[2 * i](x)
            x = self.convs[2 * i + 1](x)
            x = x + residual
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l[1])


class MultiReceptiveField(nn.Module):
    def __init__(self, channels, kernel_sizes, dilation_sizes):
        super().__init__()
        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResidualBlock(channels, k, d))

    def forward(self, x):
        output = 0
        for block in self.blocks:
            output += block(x)
        return output

    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()


class HiFiGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernels=[16, 16, 4, 4],
        resblock_kernels=[3, 7, 11],
        resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernels)
        self.num_upsamples = len(upsample_rates)

        self.pre_conv = weight_norm(
            nn.Conv1d(in_channels, initial_channel, 7, 1, padding=3)
        )

        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        initial_channel // (2**i),
                        initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.mrfs = nn.ModuleList()
        for i in range(len(upsample_rates)):
            channels = initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                MultiReceptiveField(channels, resblock_kernels, resblock_dilations)
            )

        self.post_conv = weight_norm(
            nn.Conv1d(
                initial_channel // (2 ** len(upsample_rates)), 1, 7, 1, padding=3
            )
        )
        self.activation = nn.Tanh()

    def forward(self, spectrogram, **batch):
        x = self.pre_conv(spectrogram)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.upsamples[i](x)
            x = self.mrfs[i](x)
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = self.activation(x)
        return {"predict": x}

    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.post_conv)