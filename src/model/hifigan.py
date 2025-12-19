import torch.nn as nn


class HiFiGAN(nn.Module):
    """
    HiFiGAN model.
    """

    def __init__(self, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator):
        super().__init__()
        self.Generator = Generator
        self.MultiPeriodDiscriminator = MultiPeriodDiscriminator
        self.MultiScaleDiscriminator = MultiScaleDiscriminator

    def forward(self, **batch):
        return self.Generator(**batch)