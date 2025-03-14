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

# IMG_ROOT = '../../../researchers/cristian-tejos/datasets/Phantom_real_data/'
IMG_ROOT = '../Phantom_real_data/'
# IMG_ROOT = '../../Imagenes/Phantom_real_data/'
MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')
GT_ROOT = os.path.join(IMG_ROOT, 'Phantom_tensor_2/Susceptibility_sti/Phantom_microstructure')
CHI_GT_FILE = os.path.join(GT_ROOT, 'chi.nii.gz')
MMS_GT_FILE = os.path.join(GT_ROOT, 'mms.nii.gz')
MSA_GT_FILE = os.path.join(GT_ROOT, 'msa.nii.gz')

# Reconstructed files
SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Exp_phantom_1/')
# SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Noised_data/')
ALGORITHMS_FOLDERS = ['MMSR']  # 'STI_pytorch', 'STI-net', 'DRSTI', 'COSMOS_STI', 'STI_suite',
ANGLES = [10, 20, 30, 40]

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
    gt_chi, _ = read_img(CHI_GT_FILE)
    gt_mms, _ = read_img(MMS_GT_FILE)
    gt_msa, _ = read_img(MSA_GT_FILE)
    white_matter, _ = read_img(ATLAS_NAME)
    white_matter = white_matter == 3

    for n_algorithm in ALGORITHMS_FOLDERS:
        for n_theta in range(len(ANGLES)):
            actual_theta_folder = 'Theta_' + str(ANGLES[n_theta])
            for n_psi in range(len(ANGLES)):
                actual_psi_folder = 'Psi_' + str(ANGLES[n_psi])
                actual_folder = os.path.join(SIMULATED_FOLDER, actual_theta_folder, actual_psi_folder, n_algorithm)
                actual_chi = os.path.join(actual_folder, 'chi.nii.gz')
                mms_model_path = os.path.join(actual_folder, 'mms.nii.gz')
                msa_model_path = os.path.join(actual_folder, 'msa.nii.gz')
                v1_model_path = os.path.join(actual_folder, 'eig_v1.nii.gz')
                v2_model_path = os.path.join(actual_folder, 'eig_v2.nii.gz')
                v3_model_path = os.path.join(actual_folder, 'eig_v3.nii.gz')
                eig_model_path = os.path.join(actual_folder, 'eig_val.nii.gz')
                v1_diff_dti_name = os.path.join(actual_folder, 'diff_v1_dti.nii.gz')
                v1_diff_sti_name = os.path.join(actual_folder, 'diff_v1_sti.nii.gz')
                diff_mms_file = os.path.join(actual_folder, 'diff_mms.nii.gz')
                diff_msa_file = os.path.join(actual_folder, 'diff_msa.nii.gz')

                print('Reading reconstructed susceptibility tensor')
                chi_model, nii = read_img(actual_chi)
                print('Starting Eigen-decomposition')
                print('...')
                l, v1, v2, v3, mms, msa = eig_decomposition(chi_model, mask)

                print('... PEV difference')
                tmp = get_angle_images(pev_dti, v1)
                save_img(nii, tmp * white_matter, v1_diff_dti_name)
                tmp = get_angle_images(pev_sti, v1)
                save_img(nii, tmp * white_matter, v1_diff_sti_name)

                print('Saving images')
                save_img(nii, mms, mms_model_path)
                save_img(nii, msa, msa_model_path)
                save_img(nii, v1, v1_model_path)
                save_img(nii, v2, v2_model_path)
                save_img(nii, v3, v3_model_path)
                save_img(nii, l, eig_model_path)
                save_img(nii, gt_mms - mms, diff_mms_file)
                save_img(nii, gt_msa - msa, diff_msa_file)


if __name__ == '__main__':
    main()
