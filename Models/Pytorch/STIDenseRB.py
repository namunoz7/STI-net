from collections import OrderedDict
import torch
import torch.nn as nn


class STIDenseRB(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, init_features=32):
        super(STIDenseRB, self).__init__()
        features = init_features
        self.skip = nn.Identity()

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dense1 = STIDenseRB._basis_block(in_channels, features, name='dense1')
        self.conv_dense1 = nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3)

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dense2 = STIDenseRB._basis_block(features, 2*features, name='dense2')
        self.conv_dense2 = nn.Conv3d(in_channels=features, out_channels=2*features, kernel_size=3)

        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dense3 = STIDenseRB._basis_block(2*features, 4*features, name='dense3')
        self.conv_dense3 = nn.Conv3d(in_channels=2*features, out_channels=4*features, kernel_size=3)

        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dense4 = STIDenseRB._basis_block(4*features, 8*features, name='dense4')
        self.conv_dense4 = nn.Conv3d(in_channels=4 * features, out_channels=8 * features, kernel_size=3)

        self.bottleneck = STIDenseRB._block(8 * features, 16 * features, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(16 * features, 8 * features, kernel_size=2, stride=2)
        self.decoder4 = STIDenseRB._block(2 * (8 * features), 8 * features, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(8 * features, 4 * features, kernel_size=2, stride=2)
        self.decoder3 = STIDenseRB._block(2 * (4 * features), 4 * features, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(4 * features, 2 * features, kernel_size=2, stride=2)
        self.decoder2 = STIDenseRB._block(2 * (2 * features), 2 * features, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(2 * features, features, kernel_size=2, stride=2)
        self.decoder1 = STIDenseRB._block(2 * features, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = STIDenseRB._res_in_res(self, x, self.dense1, self.conv_dense1)
        enc2 = self.pool1(enc1)
        enc2 = STIDenseRB._res_in_res(self, enc2, self.dense2, self.conv_dense2)
        enc3 = self.pool2(enc2)
        enc3 = STIDenseRB._res_in_res(self, enc3, self.dense3, self.conv_dense3)
        enc4 = self.pool3(enc3)
        enc4 = STIDenseRB._res_in_res(self, enc4, self.dense4, self.conv_dense4)

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

    @staticmethod
    def _basis_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv", nn.Conv3d(in_channels=in_channels, out_channels=features,
                                              kernel_size=3, padding=1, bias=False)),
                    (name + "relu", nn.LeakyReLU(negative_slope=0.2, inplace=True))
                ]
            )
        )

    @staticmethod
    def _dense_block(x, basis_block, conv_basis):
        layer1 = basis_block(x)
        layer1 = torch.cat((layer1, x), dim=1)

        layer2 = basis_block(layer1)
        layer2 = torch.cat((layer2, layer1, x), dim=1)

        layer3 = basis_block(layer2)
        layer3 = torch.cat((layer3, layer2, layer1, x), dim=1)

        layer4 = basis_block(layer3)
        layer4 = torch.cat((layer4, layer2, layer1, x), dim=1)
        return conv_basis(layer4)

    def _res_in_res(self, x, basis_block, conv_basis):
        out1 = STIDenseRB._dense_block(x, basis_block, conv_basis) + self.skip(x)
        out2 = STIDenseRB._dense_block(out1, basis_block, conv_basis) + self.skip(out1)
        out3 = STIDenseRB._dense_block(out2, basis_block, conv_basis) + self.skip(out2)
        return STIDenseRB._dense_block(out3, basis_block, conv_basis) + self.skip(x)





