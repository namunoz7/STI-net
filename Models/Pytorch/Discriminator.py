from _collections import OrderedDict

import torch.nn
import torch.nn as nn
from abc import ABC

KERNEL = (3, 3, 3)
SLOPE = 0.2
FEATURES = 32
dim_img = 48


class Discriminator(nn.Module, ABC):
    def __init__(self, in_channels=6, init_features=FEATURES):
        super(Discriminator, self).__init__()
        features = init_features
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=KERNEL, padding=1)
        self.block1 = Discriminator._block(features, 2*features, "block1")
        self.leaky1 = nn.LeakyReLU(negative_slope=SLOPE, inplace=True)
        self.block2 = Discriminator._block(2*features, 4*features, name="block2")
        self.block3 = Discriminator._block(4*features, 8*features, name="block3")
        self.block4 = Discriminator._block(8*features, 16*features, name="block4")
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=int(16*features), out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, dims=(0, 4, 1, 2, 3))
        conv0 = self.conv0(x)
        block1 = self.block1(conv0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        linear = self.linear(self.flatten(block4))
        out = self.sigmoid(linear)
        return out.squeeze()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv", nn.Conv3d(in_channels=in_channels, out_channels=features,
                                              kernel_size=(4, 4, 4), stride=(2, 2, 2))),
                    (name + "leaky_relu", nn.LeakyReLU(negative_slope=SLOPE, inplace=True))
                ]
            )
        )
