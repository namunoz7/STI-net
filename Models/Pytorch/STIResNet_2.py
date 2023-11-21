from collections import OrderedDict
import torch
import torch.nn as nn


class STIResNet_2(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, init_features=32):
        super(STIResNet_2, self).__init__()
        features = init_features
        self.skip = nn.Identity()

        self.encoder1 = STIResNet_2._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = STIResNet_2._block(features, 2 * features, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = STIResNet_2._block(2 * features, 4 * features, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = STIResNet_2._block(4 * features, 8 * features, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = STIResNet_2._block(8 * features, 16 * features, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(16 * features, 8 * features, kernel_size=2, stride=2)
        self.decoder4 = STIResNet_2._block(2 * (8 * features), 8 * features, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(8 * features, 4 * features, kernel_size=2, stride=2)
        self.decoder3 = STIResNet_2._block(2 * (4 * features), 4 * features, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(4 * features, 2 * features, kernel_size=2, stride=2)
        self.decoder2 = STIResNet_2._block(2 * (2 * features), 2 * features, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(2 * features, features, kernel_size=2, stride=2)
        self.decoder1 = STIResNet_2._block(2 * features, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x) + self.skip(x)
        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2) + self.skip(enc2)
        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3) + self.skip(enc3)
        enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc4) + self.skip(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv3d(in_channels=in_channels, out_channels=features,
                                               kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

                    (name + "conv2", nn.Conv3d(in_channels=features, out_channels=features,
                                               kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.2, inplace=True))
                ]
            )
        )
