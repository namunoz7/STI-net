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
import nibabel as nib
import torch.nn.functional

# IMG_ROOT = '../Phantom_real_data/'
# DATASET_ROOT = '../../Imagenes/'
DATASET_ROOT = '/mnt/researchers/cristian-tejos/datasets/'
# IMG_ROOT = os.path.join(DATASET_ROOT, 'Phantom_real_data')

IS_COSMOS = False

if IS_COSMOS:
    DATA_FOLDER = os.path.join(DATASET_ROOT, 'RC1_tensor_data')
    MASK_FILE = os.path.join(DATA_FOLDER, 'mask_rot.nii.gz')
    PHI_FILE = os.path.join(DATA_FOLDER, 'phase_rot.nii.gz')
else:
    IMG_ROOT = os.path.join(DATASET_ROOT, 'Phantom_real_data')
    MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')
    PHI_FILE = os.path.join(IMG_ROOT, 'Susceptibility_data', 'bulk_phase.nii.gz')
    SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Results_phantom/')
    DATA_FOLDER = os.path.join(SIMULATED_FOLDER, 'Real_data_10')

# Parameters
CHECKPOINT_ROOT = os.path.join('checkpoints', 'Pytorch')
CHECKPOINT_FOLDER = 'STI_real_data_1_11.3'
ALGORITHMS_FOLDERS = ['STI_net']  # , 'COSMOS_STI', 'STI_suite']

# Files names
CHI_NAME = 'chi_real_data.nii.gz'
MMS_NAME = 'mms_model.nii.gz'
MSA_NAME = 'msa_model.nii.gz'
V1_NAME = 'v1_model.nii.gz'
V2_NAME = 'v2_model.nii.gz'
V3_NAME = 'v3_model.nii.gz'
EIG_NAME = 'eig_model.nii.gz'

DEVICE = torch.device('cpu')


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
    print('... Sorting eigenvectors')
    vec = sort_eigenvectors(vec.real, idx, mat_size)
    v1 = torch.mul(vec[:, :, :, :, 0], mask)
    v2 = torch.mul(vec[:, :, :, :, 1], mask)
    v3 = torch.mul(vec[:, :, :, :, 2], mask)

    return eigenvalue, v1, v2, v3, mms, msa


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
    mask, _, = read_img(MASK_FILE)
    if not IS_COSMOS:
        mask = torch.nn.functional.pad(mask, (48, 48), mode='constant', value=0)
    for n_algorithm in ALGORITHMS_FOLDERS:
        print('Load actual image')
        actual_folder = os.path.join(DATA_FOLDER, n_algorithm)
        chi_file = os.path.join(actual_folder, CHI_NAME)
        mms_file = os.path.join(actual_folder, MMS_NAME)
        msa_file = os.path.join(actual_folder, MSA_NAME)
        v1_file = os.path.join(actual_folder, V1_NAME)
        v2_file = os.path.join(actual_folder, V2_NAME)
        v3_file = os.path.join(actual_folder, V3_NAME)
        eig_file = os.path.join(actual_folder, EIG_NAME)

        check_root(actual_folder)

        chi_model, nii = read_img(chi_file)
        print(chi_model.shape)

        print('Starting Eigen-decomposition')
        print('...')
        l, v1, v2, v3, mms, msa = eig_decomposition(chi_model, mask)

        print('Saving images')
        save_img(nii, mms, mms_file)
        save_img(nii, msa, msa_file)
        save_img(nii, v1, v1_file)
        save_img(nii, v2, v2_file)
        save_img(nii, v3, v3_file)
        save_img(nii, l, eig_file)


if __name__ == '__main__':
    main()
