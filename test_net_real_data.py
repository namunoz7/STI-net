#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 12:46 2023
Test the trained net with the susceptibility tensor phantoms.
Currently, the trained net is the finished net.

@author: Nestor Munoz
"""

import torch
import os
import numpy as np
import sys
import nibabel as nib
import torch.nn.functional
from Models.Pytorch.STIResNetSplit_2 import STIResNetSplit

# IMG_ROOT = '../Phantom_real_data/'
# IMG_ROOT = '../../Imagenes/Phantom_real_data/'
DATASET_ROOT = '/mnt/researchers/cristian-tejos/datasets/'
IMG_ROOT = os.path.join(DATASET_ROOT, 'Phantom_real_data')
COSMOS_IMG_ROOT = os.path.join(DATASET_ROOT, 'RC1_tensor_data')

IS_COSMOS = False

if IS_COSMOS:
    MASK_FILE = os.path.join(COSMOS_IMG_ROOT, 'mask_rot.nii.gz')
    PHI_FILE = os.path.join(COSMOS_IMG_ROOT, 'phase_rot.nii.gz')
else:
    MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')
    PHI_FILE = os.path.join(IMG_ROOT, 'Susceptibility_data', 'bulk_phase.nii.gz')

# Parameters
CHECKPOINT_ROOT = os.path.join('checkpoints', 'Pytorch')
CHECKPOINT_FOLDER = 'STI_net_state'
CHECKPOINT_NAME = 'state_dicts_sti.pt'
SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Results_phantom/')
RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, 'Real_data_10/STI_net')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FOLDER, CHECKPOINT_NAME)
ALPHA = 1

# Files names
CHI_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'chi_real_data.nii.gz')

DEVICE = torch.device('cpu')
GAMMA = 42.58
B0 = 3  # T
phase_scale = 2*np.pi*GAMMA*B0
eps = sys.float_info.epsilon


def read_img(filename):
    """
    Read nifty image
    :param filename:
    :return:
    """
    nii_img = nib.load(filename)
    img = nii_img.get_fdata()
    img = torch.from_numpy(img).to(torch.float32)
    return img, nii_img


def fft_phase(inputs):
    """
    Gets the fourier transform of the input geometric figure images.
    :param inputs: Input image
    :return:
    """
    fft_input = torch.fft.fftn(input=inputs, dim=(0, 1, 2))
    return fft_input


def inv_fft_phase(fft_input):
    """
    Gets the inverse Fourier Transform
    :param fft_input:
    :return:
    """
    inputs = torch.fft.ifftn(fft_input, dim=(0, 1, 2))
    return inputs


def shift_fft(x, dims, mat_size):
    """
    Shift zero-frequency component to center of spectrum
    :param mat_size:
    :param x: Input image
    :param dims: Dimensions to roll
    :return:
    """
    x = x.roll((mat_size[0] // 2,
                mat_size[1] // 2,
                mat_size[2] // 2), dims)
    return x


def load_model(checkpoints_path):
    """
    Loads the model from the checkpoint obtained by the training
    :param checkpoints_path:
    :return: sit_generator
    """
    device_cpu = torch.device('cpu')
    print('... Generating new Generator model ...')
    sti_model = STIResNetSplit().to(device_cpu)
    print('... Loading checkpoints of the trained model')
    checkpoint = torch.load(checkpoints_path, map_location=device_cpu)
    sti_model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    sti_model.eval()
    print('... Done')
    return sti_model


def save_img(nii, img, name):
    """
    Save the image as a nifti file
    :param nii:
    :param img:
    :param name:
    """
    header = nii.header
    nii_phi = nib.Nifti1Image(img.detach().numpy(), header.get_best_affine())
    nib.save(nii_phi, name)
    print('...' + name + ' saved')


def check_root(root_file):
    """
    Check if the root file exists. If it does not, it creates the root file
    :param root_file:
    :return:
    """
    if not os.path.exists(root_file):
        os.mkdir(root_file)
        print("Directory " + root_file + " Created ")
    else:
        print("Directory " + root_file + " already exists")


def main():
    print('Load actual image')
    phi, nii = read_img(PHI_FILE)
    mask, _, = read_img(MASK_FILE)
    alpha_rc1 = ALPHA
    if not IS_COSMOS:
        phi = torch.nn.functional.pad(phi, (0, 0, 48, 48), mode='constant', value=0)
        mask = torch.nn.functional.pad(mask, (48, 48), mode='constant', value=0)
        alpha_rc1 = 0.8
    check_root(SIMULATED_FOLDER)
    check_root(RESULTS_FOLDER)

    print('Loading STInet model')
    sti_model = load_model(CHECKPOINT_PATH)

    print('Reconstructing sti image')
    phi = alpha_rc1 * phi[:, :, :, 0:6]
    chi_model = sti_model(torch.unsqueeze(phi, dim=0))

    chi_model = torch.squeeze(chi_model)
    chi_model = torch.mul(chi_model, mask.unsqueeze(-1).repeat([1, 1, 1, 6]))
    save_img(nii, chi_model, CHI_REAL_DATA_FILE)


if __name__ == '__main__':
    main()
