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
from scipy.io import savemat
from Models.Pytorch.STIResNetSplit_2 import STIResNetSplit

# IMG_ROOT = '../Phantom_real_data/'
# IMG_ROOT = '../../Imagenes/Phantom_real_data/'
IMG_ROOT = '../../../researchers/cristian-tejos/datasets/Phantom_real_data/'

MASK_FILE = os.path.join(IMG_ROOT, 'Masks/brain_mask.nii.gz')
CHI_NAME = 'chi_sti_filt.nii.gz'
CHI_FILE = os.path.join(IMG_ROOT, 'Phantom_tensor_2/', CHI_NAME)

# Parameters
MAX_THETA = [15, 25, 35, 40]
MAX_PSI = [15, 25, 35, 40]
N_ORIENTATIONS = 6
MAT_SIZE = [224, 224, 224]

CHECKPOINT_ROOT = os.path.join('checkpoints', 'Pytorch')
CHECKPOINT_FOLDER = 'STI_real_data_1_11.3'
CHECKPOINT_NAME = 'state_dicts_sti.pt'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, CHECKPOINT_FOLDER, CHECKPOINT_NAME)

SIMULATED_FOLDER = os.path.join(IMG_ROOT, 'Results_phantom/')
RESULTS_FOLDER = os.path.join(SIMULATED_FOLDER, CHECKPOINT_FOLDER)

ANGLES_FOLDER_BASE_NAME = 'Angles_'
RECONSTRUCTED_NAMES = 'chi_sti.nii.gz'
PHI_NAME = 'phi_brain.nii.gz'
ANGLES_MAT_NAME = 'angles.mat'

DEVICE = torch.device('cpu')
GAMMA = 42.58
B0 = 3  # T
STD_NOISE = 8e-3
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
    header = nii_img.header
    voxel_size = np.array(header.get_zooms()[0:3])

    return img, voxel_size, nii_img


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


def angles_cylinders(num_rotations, theta, psi):
    """
    Rotation angles that are used to rotate the object in the scanner. It only generates 6 angles
    :param num_rotations:
    :param theta:
    :param psi:
    return vec_theta, vec_psi
    """
    # Parameters of the angles to model a cylinder

    min_rand = -5.0
    max_rand = 5.0

    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float32) + min_rand
    # Tilt angle of main field
    vec_theta = torch.tensor([0.0, 0.0, 0.0, theta/2, theta/2, theta], dtype=torch.float32)
    vec_theta = vec_theta.reshape(1, num_rotations)
    vec_theta = vec_theta + tmp
    vec_theta = torch.deg2rad(vec_theta)

    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float32) + min_rand
    # Rotation angle with the z axis
    vec_psi = torch.tensor([0.0, -psi, psi, -psi/2, psi/2, 0.0], dtype=torch.float32)
    vec_psi = vec_psi.reshape(1, num_rotations)
    vec_psi += tmp
    vec_psi = torch.deg2rad(vec_psi)

    return vec_theta, vec_psi


def get_direction_field(vec_theta, vec_psi, n_orientations):
    """
    Gets the direction field vector of the multiple orientations, made by the cylinders. All the angles are in radians
    :param vec_theta: Rotation angle in the x-z plane
    :param vec_psi: Rotation angle in the y-z plane
    :param n_orientations:
    :return:
    """
    direction_field = torch.zeros(n_orientations, 3, dtype=torch.float32)
    direction_field[:, 0] = torch.sin(vec_theta) * torch.cos(vec_psi)  # Hx
    direction_field[:, 1] = -torch.sin(vec_psi)  # Hy
    direction_field[:, 2] = torch.cos(vec_theta) * torch.cos(vec_psi)  # Hz
    direction_field = direction_field.unsqueeze(-1)

    return direction_field


def gen_k_space(fov, n_orientations, mat_size):
    """
    Defines the K space
    :param mat_size:
    :param fov:
    :param n_orientations:
    :return:
    """
    kx = torch.arange(1, mat_size[0] + 1, dtype=torch.float32)
    ky = torch.arange(1, mat_size[1] + 1, dtype=torch.float32)
    kz = torch.arange(1, mat_size[2] + 1, dtype=torch.float32)

    center_x = mat_size[0] // 2 + 1
    center_y = mat_size[1] // 2 + 1
    center_z = mat_size[2] // 2 + 1
    kx = kx - center_x
    ky = ky - center_y
    kz = kz - center_z

    delta_kx = 1 / fov[0]
    delta_ky = 1 / fov[1]
    delta_kz = 1 / fov[2]

    #  Generation of k space

    kx = kx * delta_kx
    ky = ky * delta_ky
    kz = kz * delta_kz

    kxx, kyy, kzz = torch.meshgrid(kx, ky, kz)

    kxx = kxx.unsqueeze(3).repeat(1, 1, 1, n_orientations)
    kyy = kyy.unsqueeze(3).repeat(1, 1, 1, n_orientations)
    kzz = kzz.unsqueeze(3).repeat(1, 1, 1, n_orientations)

    k = torch.zeros(mat_size[0], mat_size[1], mat_size[2], n_orientations, 3, dtype=torch.float32)
    k[:, :, :, :, 0] = kxx
    k[:, :, :, :, 1] = kyy
    k[:, :, :, :, 2] = kzz
    k = k.unsqueeze(-1)

    return k, kxx, kyy, kzz


def projection_variables(vec_theta, vec_psi, n_orientations, fov, mat_size):
    """
    Generates the projection variables of the STI model (a_ii and a_ij) and construct the projection matrix
    These are only made with 12 different orientation
    :param mat_size:
    :param fov:
    :param vec_theta: vector containing the angle of deviation with respect to the main field axis
    :param vec_psi: vector containing the angle of rotation of the x-y plane.
    :param n_orientations:
    :return: A: matrix containing each one of the projection angles.
    """
    direction_field = get_direction_field(vec_theta, vec_psi, n_orientations)
    print('...... Generating k space')
    k, kxx, kyy, kzz = gen_k_space(fov, n_orientations, mat_size)
    print('...... Done')
    k2 = (kxx * kxx) + (kyy * kyy) + (kzz * kzz)
    k2[k2 == 0] = eps
    kt_h = torch.matmul(k.transpose(-2, -1), direction_field).squeeze()
    direction_field = direction_field.squeeze()
    print('...... Calculating auxiliary variables')
    # Aux variables are defined
    a_11 = ((direction_field[:, 0] * direction_field[:, 0]) / 3) - ((kt_h / k2) * kxx * direction_field[:, 0])
    a_22 = ((direction_field[:, 1] * direction_field[:, 1]) / 3) - ((kt_h / k2) * kyy * direction_field[:, 1])
    a_33 = ((direction_field[:, 2] * direction_field[:, 2]) / 3) - ((kt_h / k2) * kzz * direction_field[:, 2])
    a_12 = ((2 / 3) * direction_field[:, 0] * direction_field[:, 1]) - \
           ((kt_h / k2) * (kxx * direction_field[:, 1] + kyy * direction_field[:, 0]))
    a_13 = ((2 / 3) * direction_field[:, 0] * direction_field[:, 2]) - \
           ((kt_h / k2) * (kxx * direction_field[:, 2] + kzz * direction_field[:, 0]))
    a_23 = ((2 / 3) * direction_field[:, 2] * direction_field[:, 1]) - \
           ((kt_h / k2) * (kzz * direction_field[:, 1] + kyy * direction_field[:, 2]))
    print('...... Done')
    matrix_projection = torch.zeros(mat_size[0], mat_size[1], mat_size[2], n_orientations, 6, dtype=torch.float32)
    matrix_projection[:, :, :, :, 0] = a_11
    matrix_projection[:, :, :, :, 1] = a_12
    matrix_projection[:, :, :, :, 2] = a_13
    matrix_projection[:, :, :, :, 3] = a_22
    matrix_projection[:, :, :, :, 4] = a_23
    matrix_projection[:, :, :, :, 5] = a_33

    matrix_projection = shift_fft(matrix_projection, (0, 1, 2), mat_size)

    return matrix_projection


def get_total_phase(chi, vec_theta, vec_psi, fov, mat_size, n_orientations):
    """
    Calculates the total phase of the tensor using the linear form of the Tensor model
    :param n_orientations: Number of orientations in the
    :param mat_size:
    :param fov:
    :param chi: Vectorized form of the tensor
    :param vec_theta: Angle of deviation of the image with the main magnetic field direction
    :param vec_psi: Angle of rotation in the x-y plane
    :return: phase: the local phase of the image
    """
    def add_gauss_noise(local_field, std_noise=STD_NOISE):
        """
        Generates noise in the
        :param local_field: phase image of the tensor
        :param std_noise: Standard deviation of the gaussian noise
        :return:
        """
        local_field += std_noise * torch.rand(local_field.size()).to(DEVICE)
        return local_field

    matrix_projection = projection_variables(vec_theta, vec_psi, n_orientations, fov, mat_size)
    fft_chi = fft_phase(chi)
    tmp_real = torch.matmul(matrix_projection, torch.real(fft_chi).unsqueeze(-1))
    tmp_img = torch.matmul(matrix_projection, torch.imag(fft_chi).unsqueeze(-1))
    tmp_phi = torch.cat((tmp_real, tmp_img), dim=-1)
    phi = inv_fft_phase(torch.view_as_complex(tmp_phi))
    phi = add_gauss_noise(phi.real, STD_NOISE)
    return torch.real(phi), matrix_projection


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
    chi, voxel_size, nii = read_img(CHI_FILE)
    mask, _, _, = read_img(MASK_FILE)
    fov = MAT_SIZE * voxel_size
    chi = torch.nn.functional.pad(chi, (0, 0, 48, 48), mode='constant', value=0)
    mask = torch.nn.functional.pad(mask, (48, 48), mode='constant', value=0)
    check_root(SIMULATED_FOLDER)
    check_root(RESULTS_FOLDER)
    sti_model = load_model(CHECKPOINT_PATH)
    for n_angle in range(4):
        actual_theta = MAX_THETA[n_angle]
        actual_psi = MAX_PSI[n_angle]
        angles_folder_name = ANGLES_FOLDER_BASE_NAME + str(actual_theta)
        angles_folder = os.path.join(RESULTS_FOLDER, angles_folder_name)
        check_root(angles_folder)

        phi_file = os.path.join(angles_folder, PHI_NAME)
        reconstructed_file = os.path.join(angles_folder, RECONSTRUCTED_NAMES)
        angles_file = os.path.join(angles_folder, ANGLES_MAT_NAME)

        print('Simulating field map')
        vec_theta, vec_psi = angles_cylinders(N_ORIENTATIONS, theta=actual_theta, psi=actual_psi)
        dic = {'vec_theta': vec_theta.numpy(), 'vec_psi': vec_psi.numpy()}
        savemat(angles_file, dic)
        print(angles_file + ' saved')
        phi, mat_projection = get_total_phase(chi, vec_theta, vec_psi, fov, MAT_SIZE, N_ORIENTATIONS)
        phi = torch.mul(phi, mask.unsqueeze(-1).repeat([1, 1, 1, N_ORIENTATIONS]))

        save_img(nii, phi * phase_scale, phi_file)

        chi_model = sti_model(torch.unsqueeze(phi, dim=0))
        chi_model = torch.squeeze(chi_model)
        chi_model = torch.mul(chi_model, mask.unsqueeze(-1).repeat([1, 1, 1, 6]))
        save_img(nii, chi_model, reconstructed_file)


if __name__ == '__main__':
    main()
