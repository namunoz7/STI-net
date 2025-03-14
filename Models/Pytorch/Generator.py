from _collections import OrderedDict
import torch.nn as nn
from abc import ABC

KERNEL = 3
KERNEL1 = 7
SLOPE = 0.4
RES_BLOCKS = 9
FEATURES = 32


class Generator(nn.Module, ABC):
    def __init__(self, in_channels=6, out_channels=6, init_features=FEATURES):
        super(Generator, self).__init__()
        features = init_features
        self.in_pad = nn.ConstantPad3d(padding=3, value=0)
        # Init conv block
        # self.in_block = Generator._block(in_channels, features, KERNEL1, "in_block")
        self.conv1 = nn.Conv3d(in_channels, features, kernel_size=KERNEL1, padding=3)
        self.norm1 = nn.InstanceNorm3d(num_features=features)
        self.relu1 = nn.LeakyReLU(negative_slope=SLOPE)
        # Down sampling
        self.block1 = Generator._block(features, 2*features, KERNEL, "block1")
        self.block2 = Generator._block(2*features, 4*features, KERNEL, "block2")
        # Residual blocks
        self.res = Generator._residual_block(4 * features, "res")
        # Up sampling
        self.up_block1 = Generator._up_block(4*features, 2*features, "up_block1")
        self.up_block2 = Generator._up_block(2*features, features, "up_block2")
        # Output layer
        self.out_conv = nn.Conv3d(features, out_channels, KERNEL1, padding=3)
        self.tan = nn.Tanh()

    def forward(self, x):
        # in_pad = self.in_pad(x)
        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        act1 = self.relu1(norm1)
        # in_block = self.in_block(act1)
        block1 = self.block1(act1)
        block2 = self.block2(block1)
        res1 = self.res(block2) + block2
        res2 = self.res(res1) + res1
        res3 = self.res(res2) + res2
        res4 = self.res(res3) + res3
        res5 = self.res(res4) + res1
        res6 = self.res(res5) + res5
        res7 = self.res(res6) + res6
        res8 = self.res(res7) + res7
        res9 = self.res(res8) + res8
        up_block1 = self.up_block1(res9)
        up_block2 = self.up_block2(up_block1)
        out_conv = self.out_conv(up_block2)
        return self.tan(out_conv)

    @staticmethod
    def _block(in_channels, features, kernel, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv", nn.Conv3d(in_channels=in_channels, out_channels=features,
                                              kernel_size=kernel, stride=2, padding=1)),
                    (name + "norm", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu", nn.LeakyReLU(negative_slope=SLOPE, inplace=True))
                ]
            )
        )

    @staticmethod
    def _up_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "up_conv", nn.ConvTranspose3d(in_channels=in_channels, out_channels=features,
                                                          kernel_size=KERNEL, stride=2, padding=1, output_padding=1)),
                    (name + "norm", nn.InstanceNorm3d(features)),
                    (name + "relu", nn.LeakyReLU(negative_slope=SLOPE, inplace=True))
                ]
            )
        )

    @staticmethod
    def _residual_block(in_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                               kernel_size=3, stride=1, padding=1, dilation=1)),
                    (name + "norm1", nn.InstanceNorm3d(num_features=in_channels)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=SLOPE, inplace=True)),

                    (name + "conv2", nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                               kernel_size=3, stride=1, padding=1, dilation=1)),
                    (name + "norm2", nn.InstanceNorm3d(num_features=in_channels)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=SLOPE, inplace=True))
                ]
            )
        )
