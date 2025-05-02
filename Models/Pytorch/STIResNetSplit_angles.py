from collections import OrderedDict
import torch
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STIResNetSplit(nn.Module):
    class BottleneckAttention(nn.Module):
        def __init__(self, in_channels, reduction_features=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_features=in_channels + 18, out_features=in_channels//reduction_features, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(in_features=in_channels // reduction_features, out_features=in_channels, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.spatial_attention = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1)

        def forward(self, x, parameter_angles):
            n_batch, n_channels, _, _, _ = x.size()
            x1 = torch.cat((self.avg_pool(x).squeeze(2).squeeze(2).squeeze(2), parameter_angles), dim=1)
            channel_weights = self.fc(x1).view(n_batch, n_channels, 1, 1, 1)
            spatial_weights = self.spatial_attention(x)
            weights = channel_weights * spatial_weights
            return weights * x

    class Decoder(nn.Module):
        def __init__(self, features, out_channels):
            super().__init__()
            self.up_conv4 = STIResNetSplit._up_sample_block(in_layers=16 * features, out_layers=16 * features)
            self.decoder4 = STIResNetSplit._dec_block(2 * (16 * features), 8 * features, name="dec4_i")
            self.up_conv3 = STIResNetSplit._up_sample_block(in_layers=8 * features, out_layers=8 * features)
            self.decoder3 = STIResNetSplit._dec_block(2 * (8 * features), 4 * features, name="dec3_i")
            self.up_conv2 = STIResNetSplit._up_sample_block(in_layers=4 * features, out_layers=4 * features)
            self.decoder2 = STIResNetSplit._dec_block(2 * (4 * features), 2 * features, name="dec2_i")
            self.up_conv1 = STIResNetSplit._up_sample_block(in_layers=2 * features, out_layers=2 * features)
            self.decoder1 = STIResNetSplit._dec_block(2 * (2 * features), features, name="dec1_i")
            self.up_conv0 = STIResNetSplit._up_sample_block(in_layers=features, out_layers=features)
            self.conv = nn.Sequential(nn.Conv3d(in_channels=2 * features, out_channels=out_channels,
                                                kernel_size=3, padding=1, device=DEVICE),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))

        def forward(self, bottleneck, enc0, enc1, enc2, enc3, enc4):
            dec4 = self.up_conv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)

            dec3 = self.up_conv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.up_conv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)

            dec1 = self.up_conv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)

            dec0 = self.up_conv0(dec1)
            dec0 = torch.cat((dec0, enc0), dim=1)
            iso_chi = self.conv(dec0)
            return torch.permute(iso_chi, dims=(0, 2, 3, 4, 1))

    def __init__(self, in_channels=12, out_channels=(3, 3), init_features=32, stride=(2, 2, 2)):
        super(STIResNetSplit, self).__init__()
        # ------------------------------------
        # Encoders
        # ------------------------------------
        features = init_features

        self.encoder0 = STIResNetSplit._encoder_0(input_layer=in_channels, features=features)
        # Res Block 1
        self.encoder1 = STIResNetSplit._enc_block(features, 2 * features, name="enc1_1", stride=stride)
        self.shortcut1 = STIResNetSplit._shortcut_layer(features, 2 * features, stride=stride)
        # Res Block 2
        self.encoder2 = STIResNetSplit._enc_block(2 * features, 4 * features, name="enc2_1", stride=stride)
        self.shortcut2 = STIResNetSplit._shortcut_layer(2 * features, 4 * features, stride=stride)
        # Res Block 3
        self.encoder3 = STIResNetSplit._enc_block(4 * features, 8 * features, name="enc3_1", stride=stride)
        self.shortcut3 = STIResNetSplit._shortcut_layer(4 * features, 8 * features, stride=stride)
        # Res Block 4
        self.encoder4 = STIResNetSplit._enc_block(8 * features, 16 * features, name="enc4_1", stride=stride)
        self.shortcut4 = STIResNetSplit._shortcut_layer(8 * features, 16 * features, stride=stride)

        # ------------------------------------
        # Bottleneck
        # ------------------------------------

        self.bottleneck1 = STIResNetSplit._bottleneck(16 * features, 32 * features, name="bottleneck", stride=(2, 2, 2))
        self.bottleneck_attention = self.BottleneckAttention(32 * features)
        self.bottleneck2 = STIResNetSplit._bottleneck(32 * features, 16 * features, name="bottleneck", stride=(1, 1, 1))

        # ------------------------------------
        # Isotropic decoder
        # ------------------------------------
        self.decoder_diag = self.Decoder(features=16 * features, out_channels=out_channels[0])
        self.decoder_no_diag = self.Decoder(features=16 * features, out_channels=out_channels[1])

    def forward(self, x, parameter_angles):
        # Encoder
        x = torch.permute(x, dims=(0, 4, 1, 2, 3))
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0) + self.shortcut1(enc0)
        enc2 = self.encoder2(enc1) + self.shortcut2(enc1)
        enc3 = self.encoder3(enc2) + self.shortcut3(enc2)
        enc4 = self.encoder4(enc3) + self.shortcut4(enc3)

        # Bottleneck
        bottleneck1 = self.bottleneck1(enc4)
        bottleneck_attention = self.bottleneck_attention(bottleneck1, parameter_angles)
        bottleneck = self.bottleneck2(bottleneck_attention)

        # Decoders
        out_i = self.decoder_diag(bottleneck, enc0, enc1, enc2, enc3, enc4)
        out_a = self.decoder_no_diag(bottleneck, enc0, enc1, enc2, enc3, enc4)

        # Final Tensor
        return torch.stack(tensors=[out_i[:, :, :, :, 0],
                                    out_a[:, :, :, :, 0],
                                    out_a[:, :, :, :, 1],
                                    out_i[:, :, :, :, 1],
                                    out_a[:, :, :, :, 2],
                                    out_i[:, :, :, :, 2]], dim=4)

    @staticmethod
    def _enc_block(in_channels, features, name, stride):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_norm1", nn.BatchNorm3d(num_features=in_channels, device=DEVICE)),
                    (name + "_conv1", nn.Conv3d(in_channels=in_channels, out_channels=features, stride=stride,
                                                kernel_size=(3, 3, 3), padding=1, bias=False, device=DEVICE)),
                    (name + "_relu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

                    (name + "_norm2", nn.BatchNorm3d(num_features=features, device=DEVICE)),
                    (name + "_conv2", nn.Conv3d(in_channels=features, out_channels=features,
                                                kernel_size=(3, 3, 3), padding=1, bias=False, device=DEVICE)),
                    (name + "_relu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )

    @staticmethod
    def _bottleneck(in_channels, features, name, stride):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_norm", nn.BatchNorm3d(num_features=in_channels, device=DEVICE)),
                    (name + "_conv", nn.Conv3d(in_channels=in_channels, out_channels=features, stride=stride,
                                               kernel_size=(3, 3, 3), padding=1, bias=False, device=DEVICE)),
                    (name + "_relu", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )

    @staticmethod
    def _shortcut_layer(in_channels, out_channels, stride):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), padding=0,
                         stride=stride, device=DEVICE)

    @staticmethod
    def _dec_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_conv1", nn.Conv3d(in_channels=in_channels, out_channels=features,
                                                kernel_size=(3, 3, 3), padding=1, bias=False, device=DEVICE)),
                    (name + "_norm1", nn.BatchNorm3d(num_features=features, device=DEVICE)),
                    (name + "_relu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

                    (name + "_conv2", nn.Conv3d(in_channels=features, out_channels=features,
                                                kernel_size=(3, 3, 3), padding=1, bias=False, device=DEVICE)),
                    (name + "_norm2", nn.BatchNorm3d(num_features=features, device=DEVICE)),
                    (name + "_relu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ]
            )
        )

    @staticmethod
    def _encoder_0(input_layer, features=32):
        return nn.Sequential(
            nn.Conv3d(in_channels=input_layer, kernel_size=(3, 3, 3), out_channels=features,
                      padding=1, bias=False, device=DEVICE),
            nn.BatchNorm3d(num_features=features, device=DEVICE),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    @staticmethod
    def _up_sample_block(in_layers, out_layers):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels=in_layers, out_channels=out_layers, kernel_size=(3, 3, 3), padding=1, device=DEVICE)
        )
