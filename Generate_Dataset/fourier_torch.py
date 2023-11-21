import torch
from generate_mat import MAT_SIZE


def fft_phase(inputs):
    """
    Gets the fourier transform of the input geometric figure images.
    :param inputs: Input image
    :return:
    """
    fft_input = torch.fft.fftn(input=inputs, dim=(0, 1, 2))
    return fft_input


def shift_fft(x, dims):
    """
    Shift zero-frequency component to center of spectrum
    :param x: Input image
    :param dims: Dimensions to roll
    :return:
    """
    x = x.roll((MAT_SIZE[0] // 2,
                MAT_SIZE[1] // 2,
                MAT_SIZE[2] // 2), dims)
    return x


def inv_shift_fft(x, dims):
    """
    Shift zero-frequency component to position 0
    :param x: Input matrix
    :param dims: Dimensions to roll
    :return:
    """
    x = x.roll(((MAT_SIZE[0] + 1) // 2,
                (MAT_SIZE[1] + 1) // 2,
                (MAT_SIZE[2] + 1) // 2), dims)
    return x


def inv_fft_phase(fft_input):
    """
    Gets the inverse Fourier Transform
    :param fft_input:
    :return:
    """
    inputs = torch.fft.ifftn(fft_input, dim=(0, 1, 2))
    return inputs
