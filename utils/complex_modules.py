import torch
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComplexConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, bias=False):
        super().__init__()
        self._re_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias, device=DEVICE)
        self._im_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias, device=DEVICE)

    def forward(self, x):
        x_re = self._re_conv(x.real) - self._im_conv(x.imag)
        x_im = (self._re_conv(x.real) + self._im_conv(x.real))
        return torch.complex(x_re, x_im)


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.02, inplace=False):
        super().__init__()
        self._real_l_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self._imag_l_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        return torch.complex(self._real_l_relu(x.real), self._imag_l_relu(x.imag))


class ComplexSigmoid(nn.Module):
    def __init__(self, negative_slope=0.02, inplace=False):
        super().__init__()
        self._real_l_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self._imag_l_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        return torch.complex(self._real_l_relu(x.real), self._imag_l_relu(x.imag))


class ComplexUpSample(nn.Module):
    def __init__(self, scale_factor=2, mode='trilinear'):
        super().__init__()
        self.real_up_sample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.imag_up_sample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x_real = self.real_up_sample(x.real)
        x_imag = self.imag_up_sample(x.imag)
        return torch.complex(x_real, x_imag)


class ComplexBatchNorm3d(nn.Module):
    def __init__(self, num_features, device):
        super().__init__()
        self.batch_norm_3d_real = nn.BatchNorm3d(num_features=num_features, device=device)
        self.batch_norm_3d_imag = nn.BatchNorm3d(num_features=num_features, device=device)

    def forward(self, x):
        return torch.complex(self.batch_norm_3d_real(x.real), self.batch_norm_3d_imag(x.imag))
