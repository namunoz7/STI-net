import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(in_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3, 3),
                      stride=stride, padding=padding),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=(3, 3, 3), padding=padding),
        )
        self.skip_connection = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.skip_connection(x)


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super(UpSample, self).__init__()

        self.up_sample = nn.ConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.up_sample(x)
