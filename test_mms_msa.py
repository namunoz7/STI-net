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
import sys
import nibabel as nib
import torch.nn.functional

IMG_ROOT = '../../../researchers/cristian-tejos/datasets/Phantom_real_data/'
# IMG_ROOT = '../Phantom_real_data/'
# IMG_ROOT = '../../Imagenes/Phantom_real_data/'
MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')

TENSOR_NAMES = ['chi_sti.nii.gz', 'chi_cosmos_sti.nii.gz', 'chi_suite.nii.gz']
MMS_NAMES = ['mms_model.nii.gz', 'mms_cosmos_sti.nii.gz', 'mms_suite.nii.gz']
MSA_NAMES = ['msa_model.nii.gz', 'msa_cosmos_sti.nii.gz', 'msa_suite.nii.gz']
V1_NAMES = ['v1_model.nii.gz', 'v1_cosmos_sti.nii.gz', 'v1_suite.nii.gz']
V2_NAMES = ['v2_model.nii.gz', 'v2_cosmos_sti.nii.gz', 'v2_suite.nii.gz']
V3_NAMES = ['v3_model.nii.gz', 'v3_cosmos_sti.nii.gz', 'v3_suite.nii.gz']
EIG_NAMES = ['eig_model.nii.gz', 'eig_cosmos_sti.nii.gz', 'eig_suite.nii.gz']

# Parameters
MAX_THETA = [15, 25, 35, 40]
MAT_SIZE = [224, 224, 224]

# STI-net weights
CHECKPOINT_ROOT = os.path.join('checkpoints', 'Pytorch')
CHECKPOINT_FOLDER = 'STI_real_data_1_11.3'
CHECKPOINT_NAME = 'state_dicts_sti.pt'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FOLDER, CHECKPOINT_NAME)

# Reconstructed files
SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Results_phantom/')
DATE_FOLDER = 'September_12'
RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, CHECKPOINT_FOLDER, DATE_FOLDER)
ANGLES_FOLDER_BASE_NAME = 'Angles_'
ANGLES_MAT_NAME = 'angles.mat'

# PEV files
CHI_PEV_FILE = os.path.join(IMG_ROOT, 'Susceptibility_data', 'V1_filtered.nii.gz')
DTI_PEV_FILE = os.path.join(IMG_ROOT, 'Diffusion_data', 'DTI_reg_files', 'V1_filtered.nii.gz')
ATLAS_NAME = os.path.join(IMG_ROOT, 'Masks', 'atlas_mask.nii.gz')

DEVICE = torch.device('cpu')


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


def eig_decomposition(tensor, mask):
    """
    Generates the eigen decomposition of te given tensor in pytorch
    :param tensor: Susceptibility tensor in vector form
    :param mask: Mask of the brain
    :return: eigen values and each one of the eigen vectors.
    """
    def gen_mms_msa(eigenvalues):
        """
        Calculates the mean magnetic susceptibility (MMS) and magnetic susceptibility anisotropy (MSA)
        :param eigenvalues: Eigenvalues calculated from the reconstructed susceptibility tensor.
        :return: mms, msa
        """
        mean_susceptibility = torch.mean(eigenvalues, dim=-1)
        anisotropy = eigenvalues[:, :, :, 0] - torch.mean(eigenvalues[:, :, :, 1:3], -1)
        return mean_susceptibility, anisotropy

    def sort_eigenvectors(eigenvectors, indices, mat_shape):
        """
        Sort the elements of the eigenvectors with the resulting mat_size
        :param eigenvectors:
        :param mat_shape:
        :param indices:
        :return:
        """
        tmp = mat_size[0] * mat_size[1] * mat_shape[2]
        indices = indices.view(tmp, 3)
        eigenvectors = eigenvectors.view(tmp, 3, 3).real
        for n_tmp in range(tmp):
            tmp_1 = eigenvectors[n_tmp, :, :]
            tmp_idx = indices[n_tmp, :]
            eigenvectors[n_tmp, :, :] = tmp_1[:, tmp_idx]

        eigenvectors = eigenvectors.view(mat_shape[0], mat_shape[1], mat_shape[2], 3, 3)
        return eigenvectors

    mask = mask.unsqueeze(-1).repeat([1, 1, 1, 3])
    print('Preparing susceptibility tensor to do Eigen-decomposition')
    tensor = torch.stack([tensor[:, :, :, 0], tensor[:, :, :, 1], tensor[:, :, :, 2],
                          tensor[:, :, :, 1], tensor[:, :, :, 3], tensor[:, :, :, 4],
                          tensor[:, :, :, 2], tensor[:, :, :, 4], tensor[:, :, :, 5]], dim=-1)
    mat_size = tensor.size()
    tensor = torch.reshape(tensor, shape=(mat_size[0], mat_size[1], mat_size[2], 3, 3))
    print('Performing Eigen-decomposition')
    eigenvalue, vec = torch.linalg.eig(tensor)
    print('Sorting eigenvalues in descending order')
    eigenvalue, idx = torch.sort(eigenvalue.real, descending=True)
    print('Calculating MMS and MSA')
    mms, msa = gen_mms_msa(eigenvalue)
    print('Sorting eigenvectors')
    vec = sort_eigenvectors(vec.real, idx, mat_size)
    v1 = torch.mul(vec[:, :, :, :, 0], mask)
    v2 = torch.mul(vec[:, :, :, :, 1], mask)
    v3 = torch.mul(vec[:, :, :, :, 2], mask)

    return eigenvalue, v1, v2, v3, mms, msa


def get_angle_images(pev_1, pev_2):
    """
    Calculates the cosine of both principal eigenvectors.
    :param pev_1:
    :param pev_2:
    :return:
    """
    pev_1 = torch.unsqueeze(pev_1, dim=-1)
    pev_2 = torch.unsqueeze(pev_2, dim=-2)
    dot_prod = torch.matmul(pev_2, pev_1)
    dot_prod = torch.abs(torch.squeeze(dot_prod))
    return torch.squeeze(dot_prod)


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


def main():
    mask, _ = read_img(MASK_FILE)
    pev_sti, _ = read_img(CHI_PEV_FILE)
    pev_dti, _ = read_img(DTI_PEV_FILE)
    white_matter, _ = read_img(ATLAS_NAME)
    white_matter = white_matter == 3

    mask = torch.nn.functional.pad(mask, (48, 48), mode='constant', value=0)
    white_matter = torch.nn.functional.pad(white_matter, (48, 48), mode='constant', value=0)
    pev_dti = torch.nn.functional.pad(white_matter, (0, 0, 48, 48), mode='constant', value=0)
    pev_sti = torch.nn.functional.pad(white_matter, (0, 0, 48, 48), mode='constant', value=0)

    for n_angle in range(4):
        for n_solver in range(3):
            actual_theta = MAX_THETA[n_angle]
            angles_folder_name = ANGLES_FOLDER_BASE_NAME + str(actual_theta)
            angles_folder = os.path.join(RESULTS_FOLDER, angles_folder_name)

            # Files names
            chi_model_path = os.path.join(angles_folder, TENSOR_NAMES[n_solver])
            mms_model_path = os.path.join(angles_folder, MMS_NAMES[n_solver])
            msa_model_path = os.path.join(angles_folder, MSA_NAMES[n_solver])
            v1_model_path = os.path.join(angles_folder, V1_NAMES[n_solver])
            v2_model_path = os.path.join(angles_folder, V2_NAMES[n_solver])
            v3_model_path = os.path.join(angles_folder, V3_NAMES[n_solver])
            eig_model_path = os.path.join(angles_folder, EIG_NAMES[n_solver])
            v1_diff_dti_name = os.path.join(angles_folder, 'diff_v1.nii.gz')
            v1_diff_sti_name = os.path.join(angles_folder, 'diff_v1.nii.gz')

            print('Reading reconstructed susceptibility tensor')
            chi_model, nii = read_img(chi_model_path)
            print('Starting Eigen-decomposition')
            print('...')
            l, v1, v2, v3, mms, msa = eig_decomposition(chi_model, mask)

            print('... PEV difference')
            tmp = get_angle_images(pev_dti, v1)
            save_img(nii, tmp * white_matter, v1_diff_sti_name)
            tmp = get_angle_images(pev_sti, v1)
            save_img(nii, tmp * white_matter, v1_diff_dti_name)

            print('Saving images')
            save_img(nii, mms, mms_model_path)
            save_img(nii, msa, msa_model_path)
            save_img(nii, v1, v1_model_path)
            save_img(nii, v2, v2_model_path)
            save_img(nii, v3, v3_model_path)
            save_img(nii, l, eig_model_path)


if __name__ == '__main__':
    main()
