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
import nibabel as nib
import scipy
import torch.nn.functional
from Models.Pytorch.STIResNetSplit_2 import STIResNetSplit
from mms_msa_real_data import eig_decomposition
from mms_msa_real_data import get_angle_images
from time import time


IS_COSMOS = False

if IS_COSMOS:
    IMG_ROOT = '../RC1_tensor_data/'
    MASK_FILE = os.path.join(IMG_ROOT, 'mask_rot.nii.gz')
    PHI_FILE = os.path.join(IMG_ROOT, 'phase_rot.nii.gz')
    SIMULATED_FOLDER = IMG_ROOT
    RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, 'STI_net_angles')
else:
    IMG_ROOT = '../Phantom_real_data/'
    # IMG_ROOT = '../../Imagenes/Phantom_real_data/'
    is_flipped = False
    if is_flipped:
        MASK_FILE = os.path.join(IMG_ROOT, 'Masks/flipped_mask.nii.gz')
        PHI_FILE = os.path.join(IMG_ROOT, 'Susceptibility_data/flipped_phase.nii.gz')
    else:
        MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')
        PHI_FILE = os.path.join(IMG_ROOT, 'Susceptibility_data/bulk_phase2.nii.gz')
    SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Susceptibility_data/')
    # RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, 'Real_data/STI_net_angles')
    RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, 'Real_data/STI_shuffled_May26-12:15')
    ATLAS_NAME = os.path.join(IMG_ROOT, 'Masks', 'atlas_mask.nii.gz')
    DTI_PEV_FILE = os.path.join(IMG_ROOT, 'Diffusion_data/DTI_reg_files/V1_filtered.nii.gz')
    DIRECTION_FIELD_FILE = os.path.join(SIMULATED_FOLDER, 'direction_field.mat')

# Parameters
directions_used = [0, 5, 4, 1, 2, 3]
CHECKPOINT_ROOT = os.path.join('checkpoints', 'Pytorch')
# CHECKPOINT_FOLDER = 'STI_angles4'
CHECKPOINT_FOLDER = 'STInet_res-unet2'
CHECKPOINT_NAME = 'net_model.pt'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FOLDER, CHECKPOINT_NAME)
ALPHA = 1

# Files names
CHI_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'chi.nii.gz')
MMS_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'mms.nii.gz')
MSA_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'msa.nii.gz')
V1_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'v1.nii.gz')
V2_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'v2.nii.gz')
V3_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'v3.nii.gz')
EIG_REAL_DATA_FILE = os.path.join(RESULTS_FOLDER, 'eig_val.nii.gz')

DEVICE = torch.device('cpu')
GAMMA = 42.58
B0 = 3  # T
phase_scale = 2*np.pi*GAMMA*B0


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


def load_dir_field(dir_field_file, directions):
    """
    Load the direction field at which the scanning process was performed
    :param dir_field_file:
    :param directions:
    :return:
    """
    dir_field = scipy.io.loadmat(dir_field_file)
    dir_field = dir_field['direction_field']
    dir_field = torch.tensor(dir_field[directions, :], dtype=torch.float32).reshape(18).unsqueeze(0)
    return dir_field


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
    # sti_model.load_state_dict(checkpoint['model_state_dict'])
    sti_model.load_state_dict(checkpoint)
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
    # phi, nii = read_img(PHI_FILE)
    phi, nii = read_img(os.path.join(RESULTS_FOLDER, 'phi_2.nii.gz'))
    phi = torch.nn.functional.pad(phi, (0, 0, 48, 48), mode='constant', value=0).unsqueeze(0)
    # alpha_rc1 = ALPHA
    direction_field = load_dir_field(DIRECTION_FIELD_FILE, directions_used)
    # if not IS_COSMOS:
    #     phi = torch.nn.functional.pad(phi, (0, 0, 48, 48), mode='constant', value=0)
    #     alpha_rc1 = 1
    check_root(SIMULATED_FOLDER)
    check_root(RESULTS_FOLDER)
    # phi = alpha_rc1 * phi[:, :, :, [0, 8, 1, 4, 5, 3]]
    # phi = alpha_rc1 * phi[:, :, :, DIRECTIONS_USED]
    # save_img(nii, torch.squeeze(phi)[:, :, 48:-48, :], os.path.join(RESULTS_FOLDER, 'phi_2.nii.gz'))

    print('Loading STInet model')
    sti_model = load_model(CHECKPOINT_PATH)

    print('Reconstructing sti image')
    t = time()
    chi_model = sti_model(phi)
    print(f"Chi model shape: {chi_model.shape}")
    t2 = time() - t
    print('Elapsed time: %.2f seconds' % t2)
    del phi

    mask, _, = read_img(MASK_FILE)
    white_matter, _ = read_img(ATLAS_NAME)
    pev_dti, _ = read_img(DTI_PEV_FILE)
    white_matter = white_matter == 3

    if not IS_COSMOS:
        chi_model = torch.squeeze(chi_model)[:, :, 48:-48, :]
    print('Chi shape: ' + str(chi_model.shape))
    print('Mask shape: ' + str(mask.shape))
    chi_model = torch.mul(chi_model, mask.unsqueeze(-1).repeat([1, 1, 1, 6]))
    save_img(nii, chi_model, CHI_REAL_DATA_FILE)

    print('Starting Eigen-decomposition')
    print('...')
    l, v1, v2, v3, mms, msa = eig_decomposition(chi_model, mask)

    print('Saving images')
    save_img(nii, mms, MMS_REAL_DATA_FILE)
    save_img(nii, msa, MSA_REAL_DATA_FILE)
    save_img(nii, v1, V1_REAL_DATA_FILE)
    save_img(nii, v2, V2_REAL_DATA_FILE)
    save_img(nii, v3, V3_REAL_DATA_FILE)
    save_img(nii, l, EIG_REAL_DATA_FILE)

    if not IS_COSMOS:
        print('Difference map with DTI PEV')
        v1_diff_dti_name = os.path.join(RESULTS_FOLDER, 'diff_v1_dti.nii.gz')
        tmp = get_angle_images(pev_dti, v1)
        save_img(nii, tmp * white_matter, v1_diff_dti_name)


if __name__ == '__main__':
    main()
