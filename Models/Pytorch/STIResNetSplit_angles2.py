"""
Modified 28-03-25 16:29
@author: Nestor Muñoz
Use four levels instead of three levels of decomposition in the encoder arm
Remove the Batch Normalization to work in Fourier Space
Create the convolutional layers to manage complex numbers
"""

import torch
import torch.nn as nn
from utils.complex_modules import ComplexUpSample, ComplexConv3d, ComplexLeakyReLU, ComplexBatchNorm3d
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STIResNetSplit(nn.Module):
    class BottleneckAttention(nn.Module):
        def __init__(self, in_channels, reduction_features=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                # nn.Linear(in_features=in_channels + 18, out_features=in_channels//reduction_features,
                #           bias=False, dtype=torch.complex64),
                nn.Linear(in_features=in_channels, out_features=in_channels // reduction_features,
                          bias=False, dtype=torch.complex64),
                ComplexLeakyReLU(),
                nn.Linear(in_features=in_channels // reduction_features, out_features=in_channels,
                          bias=False, dtype=torch.complex64),
                ComplexLeakyReLU()
            )
            self.spatial_attention = ComplexConv3d(in_channels, 1, kernel_size=1,
                                                   stride=1, padding=0)

        def forward(self, x):
            n_batch, n_channels, _, _, _ = x.size()
            x1 = self.avg_pool(x).squeeze()
            channel_weights = self.fc(x1).view(n_batch, n_channels, 1, 1, 1)
            spatial_weights = self.spatial_attention(x)
            weights = channel_weights * spatial_weights
            return weights * x

    class Encoder0(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self._conv = ComplexConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self._norm = ComplexBatchNorm3d(num_features=out_channels, device=DEVICE)
            self._l_relu = ComplexLeakyReLU()

        def forward(self, x):
            return self._l_relu(self._norm(self._conv(x)))

    class Encoder(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self._norm1 = ComplexBatchNorm3d(num_features=in_channels, device=DEVICE)
            self._conv1 = ComplexConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self._l_relu_1 = ComplexLeakyReLU()

            self._norm2 = ComplexBatchNorm3d(num_features=out_channels, device=DEVICE)
            self._conv2 = ComplexConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self._l_relu_2 = ComplexLeakyReLU()

        def forward(self, x):
            x = self._l_relu_1(self._conv1(self._norm1(x)))
            x = self._l_relu_2(self._conv2(self._norm2(x)))
            return x

    class ComplexUpSample3d(nn.Module):
        def __init__(self, in_layers, out_layers):
            super().__init__()
            self._up_sample = ComplexUpSample(scale_factor=2, mode='trilinear')
            self._conv1 = ComplexConv3d(in_layers, out_layers, kernel_size=3, stride=1, padding=1)

        def forward(self, x: torch.Tensor):
            x = self._up_sample(x)
            x = self._conv1(x)
            return x

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, features):
            super().__init__()
            self._conv1 = ComplexConv3d(in_channels, features, kernel_size=3, stride=1, padding=1)
            self._norm1 = ComplexBatchNorm3d(num_features=features, device=DEVICE)
            self._l_relu_1 = ComplexLeakyReLU()

            self._conv2 = ComplexConv3d(features, features, kernel_size=3, stride=1, padding=1)
            self._norm2 = ComplexBatchNorm3d(num_features=features, device=DEVICE)
            self._l_relu_2 = ComplexLeakyReLU()

        def forward(self, x):
            x = self._l_relu_1(self._norm1(self._conv1(x)))
            x = self._l_relu_2(self._norm2(self._conv2(x)))
            return x

    def __init__(self, in_channels=6, out_channels=(3, 3), init_features=32, stride=(2, 2, 2)):
        super(STIResNetSplit, self).__init__()
        # ------------------------------------
        # Encoders
        # ------------------------------------
        features = init_features
        # Encoder 0
        self.encoder0 = self.Encoder0(in_channels, features, stride=1)
        # Res Block 1
        self.encoder1 = self.Encoder(features, 2 * features, stride=stride)
        self.shortcut1 = ComplexConv3d(features, 2 * features, stride=stride, kernel_size=1, padding=0)
        # Res Block 2
        self.encoder2 = self.Encoder(2 * features, 4 * features, stride=stride)
        self.shortcut2 = ComplexConv3d(2 * features, 4 * features, stride=stride, kernel_size=1, padding=0)
        # Res Block 3
        self.encoder3 = self.Encoder(4 * features, 8 * features, stride=stride)
        self.shortcut3 = ComplexConv3d(4 * features, 8 * features, stride=stride, kernel_size=1, padding=0)
        # Res Block 4
        self.encoder4 = self.Encoder(8 * features, 16 * features, stride=stride)
        self.shortcut4 = ComplexConv3d(8 * features, 16 * features, stride=stride, kernel_size=1, padding=0)

        # ------------------------------------
        # Bottleneck
        # ------------------------------------

        self.bottleneck1 = self.Encoder0(16 * features, 32 * features, stride=2)
        self.bottleneck_attention = self.BottleneckAttention(32 * features)
        self.bottleneck2 = self.Encoder0(32 * features, 16 * features, stride=1)

        # ------------------------------------
        # Isotropic decoder
        # ------------------------------------
        self.up_conv4_i = self.ComplexUpSample3d(16 * features, 16 * features)
        self.decoder4_i = self.DecoderBlock(2 * (16 * features), 8 * features)
        self.up_conv3_i = self.ComplexUpSample3d(8 * features, 8 * features)
        self.decoder3_i = self.DecoderBlock(2 * (8 * features), 4 * features)
        self.up_conv2_i = self.ComplexUpSample3d(4 * features, 4 * features)
        self.decoder2_i = self.DecoderBlock(2 * (4 * features), 2 * features)
        self.up_conv1_i = self.ComplexUpSample3d(2 * features, 2 * features)
        self.decoder1_i = self.DecoderBlock(2 * (2 * features), features)
        self.up_conv0_i = self.ComplexUpSample3d(features, features)
        self.conv_i = self.Encoder0(2 * features, out_channels[0], stride=1)

        # ------------------------------------
        # Anisotropic decoder
        # ------------------------------------
        self.up_conv4_a = self.ComplexUpSample3d(16 * features, 16 * features)
        self.decoder4_a = self.DecoderBlock(2 * (16 * features), 8 * features)
        self.up_conv3_a = self.ComplexUpSample3d(8 * features, 8 * features)
        self.decoder3_a = self.DecoderBlock(2 * (8 * features), 4 * features)
        self.up_conv2_a = self.ComplexUpSample3d(4 * features, 4 * features)
        self.decoder2_a = self.DecoderBlock(2 * (4 * features), 2 * features)
        self.up_conv1_a = self.ComplexUpSample3d(2 * features, 2 * features)
        self.decoder1_a = self.DecoderBlock(2 * (2 * features), features)
        self.up_conv0_a = self.ComplexUpSample3d(features, features)
        self.conv_a = self.Encoder0(2 * features, out_channels[0], stride=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        x = torch.permute(x, dims=(0, 4, 1, 2, 3))
        x = STIResNetSplit.fft_phase(x)
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0) + self.shortcut1(enc0)
        enc2 = self.encoder2(enc1) + self.shortcut2(enc1)
        enc3 = self.encoder3(enc2) + self.shortcut3(enc2)
        enc4 = self.encoder4(enc3) + self.shortcut4(enc3)

        # Bottleneck
        bottleneck1 = self.bottleneck1(enc4)
        bottleneck_attention = self.bottleneck_attention(bottleneck1)
        bottleneck = self.bottleneck2(bottleneck_attention)

        # Isotropic Decoder
        dec4_i = self.up_conv4_i(bottleneck)
        dec4_i = torch.cat((dec4_i, enc4), dim=1)
        dec4_i = self.decoder4_i(dec4_i)

        dec3_i = self.up_conv3_i(dec4_i)
        dec3_i = torch.cat((dec3_i, enc3), dim=1)
        dec3_i = self.decoder3_i(dec3_i)

        dec2_i = self.up_conv2_i(dec3_i)
        dec2_i = torch.cat((dec2_i, enc2), dim=1)
        dec2_i = self.decoder2_i(dec2_i)

        dec1_i = self.up_conv1_i(dec2_i)
        dec1_i = torch.cat((dec1_i, enc1), dim=1)
        dec1_i = self.decoder1_i(dec1_i)

        dec0_i = self.up_conv0_i(dec1_i)
        dec0_i = torch.cat((dec0_i, enc0), dim=1)
        iso_chi = self.conv_i(dec0_i)
        out_i = torch.permute(iso_chi, dims=(0, 2, 3, 4, 1))

        # Anisotropic Decoder
        dec4_a = self.up_conv4_a(bottleneck)
        dec4_a = torch.cat((dec4_a, enc4), dim=1)
        dec4_a = self.decoder4_a(dec4_a)

        dec3_a = self.up_conv3_a(dec4_a)
        dec3_a = torch.cat((dec3_a, enc3), dim=1)
        dec3_a = self.decoder3_a(dec3_a)

        dec2_a = self.up_conv2_a(dec3_a)
        dec2_a = torch.cat((dec2_a, enc2), dim=1)
        dec2_a = self.decoder2_a(dec2_a)

        dec1_a = self.up_conv1_a(dec2_a)
        dec1_a = torch.cat((dec1_a, enc1), dim=1)
        dec1_a = self.decoder1_a(dec1_a)

        dec0_a = self.up_conv0_a(dec1_a)
        dec0_a = torch.cat((dec0_a, enc0), dim=1)
        ani_chi = self.conv_a(dec0_a)
        out_a = torch.permute(ani_chi, dims=(0, 2, 3, 4, 1))

        return STIResNetSplit.inv_fft_phase(torch.stack(tensors=[out_i[:, :, :, :, 0],
                                                                 out_a[:, :, :, :, 0],
                                                                 out_a[:, :, :, :, 1],
                                                                 out_i[:, :, :, :, 1],
                                                                 out_a[:, :, :, :, 2],
                                                                 out_i[:, :, :, :, 2]], dim=4)).real

    @staticmethod
    def fft_phase(inputs):
        """
        Gets the fourier transform of the input geometric figure images.
        :param inputs: Input image
        :return:
        """
        fft_input = torch.fft.fftn(input=inputs, dim=(1, 2, 3))
        fft_input = torch.fft.fftshift(fft_input, dim=(1, 2, 3))
        return fft_input

    @staticmethod
    def inv_fft_phase(fft_input):
        """
        Gets the inverse Fourier Transform
        :param fft_input:
        :return:
        """
        fft_input = torch.fft.ifftshift(fft_input, dim=(1, 2, 3))
        inputs = torch.fft.ifftn(fft_input, dim=(1, 2, 3))
        return inputs
