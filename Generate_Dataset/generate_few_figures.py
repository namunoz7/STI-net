#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:47:45 2019

@author: Nestor Munoz
"""

import numpy as np
import torch
import sys
import h5py
import x_space_torch as space
import geometric_transformation as gt
import rand_parameters as rp
import sti_functions as sti
import fourier_torch as ft

N_CYLINDERS = 55
N_SPHERES = 60
NUM_CYLINDERS = 30
NUM_SPHERES = 30
MAT_SIZE = np.array([64, 64, 64])
N_ORIENTATIONS = 6
START = 0
NUM_FIGURES = 10
RESOLUTION = np.array([0.1, 0.1, 0.1])
FOV = MAT_SIZE * RESOLUTION
MAT_SIZE = tuple(MAT_SIZE)
WINDOW = 32
DELTA_FOV = ((np.array(MAT_SIZE) - WINDOW)//2)
# RESULTS_FOLDER = '../../Dataset/'
RESULTS_FOLDER = '../../../Imagenes/Dataset/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = sys.float_info.epsilon


def gen_cylinders_v2(n_cylinders):
    """
    Generates the off-resonance field of multiple cylinders at different rotation angles with the magnetic field
    :param n_cylinders: Maximum number of cylinders in the figure
    :return: [chi, phase, n_figures] = Tensor containing the STI. Tensor containing the off-resonance field. Number of
    figures in the image
    """

    vec_radius, vec_alpha, vec_beta = rp.rand_parameters_cylinders(n_cylinders)
    xx, yy, mat_rot = gt.transform_cylinders(n_cylinders, vec_alpha, vec_beta, RESOLUTION)

    xx2 = xx * xx
    yy2 = yy * yy
    rho = torch.sqrt(xx2 + yy2)

    mask = (rho < vec_radius).unsqueeze(-1)
    mask_cylinder = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2],
                                n_cylinders, 1, dtype=torch.float64, device=DEVICE)
    mask_cylinder[mask] = 1.0

    chi = sti.tilt_scan_image(n_cylinders, mask_cylinder, mat_rot, is_cylinder=True)
    return chi, n_cylinders


def gen_spheres_v2(n_spheres):
    """
    Generates the magnetic off-resonance field of multiple spheres
    :param n_spheres: Number of spheres in available in the FOV
    :return:
    """
    vec_radius, vec_alpha, vec_beta = rp.rand_parameters_cylinders(n_spheres)
    xx, yy, zz = space.define_space(n_spheres, RESOLUTION, MAT_SIZE)
    _, _, mat_rot = gt.transform_cylinders(n_spheres, vec_alpha, vec_beta, RESOLUTION)
    center_nx, center_ny, center_nz = space.get_center(n_spheres, RESOLUTION, MAT_SIZE)
    xx = xx.permute(1, 2, 3, 0) - center_nx
    yy = yy.permute(1, 2, 3, 0) - center_ny
    zz = zz.permute(1, 2, 3, 0) - center_nz

    xx2 = (xx * xx)
    yy2 = (yy * yy)
    zz2 = (zz * zz)

    rho_2 = xx2 + yy2 + zz2
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1
    tmp = rho <= vec_radius
    mask_sphere = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], n_spheres, 1, dtype=torch.float64, device=DEVICE)
    mask_sphere[tmp] = 1

    chi = sti.tilt_scan_image(n_spheres, mask_sphere, mat_rot, is_cylinder=False)
    return chi, n_spheres


def get_total_phase(chi, vec_theta, vec_psi):
    """
    Calculates the total phase of the tensor using the linear form of the Tensor model
    :param chi: Vectorized form of the tensor
    :param vec_theta: Angle of deviation of the image with the main magnetic field direction
    :param vec_psi: Angle of rotation in the x-y plane
    :return: phase: the local phase of the image
    """
    matrix_projection = sti.projection_variables(vec_theta, vec_psi, N_ORIENTATIONS)
    fft_chi = ft.fft_phase(chi)
    tmp_real = torch.matmul(matrix_projection, torch.real(fft_chi).unsqueeze(-1))
    tmp_img = torch.matmul(matrix_projection, torch.imag(fft_chi).unsqueeze(-1))

    tmp_phi = torch.cat((tmp_real, tmp_img), dim=-1)
    phi = ft.inv_fft_phase(torch.view_as_complex(tmp_phi))
    return torch.real(phi), matrix_projection


class Cylinders:
    global N_CYLINDERS
    global N_ORIENTATIONS

    def __init__(self, mat_size):
        self.vec_theta, self.vec_psi = rp.angles_cylinders(N_ORIENTATIONS)
        self.chi = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        # Iterate to generate more figures
        chi_cylinders, n_figures = gen_cylinders_v2(NUM_CYLINDERS)
        self.chi = self.chi + chi_cylinders


class Spheres:
    global N_SPHERES
    global N_ORIENTATIONS

    def __init__(self, mat_size):
        self.chi = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        tmp, n_spheres = gen_spheres_v2(NUM_SPHERES)
        self.chi = self.chi + tmp


def main():
    for n_img in range(START, NUM_FIGURES):
        file = 'data_' + str(n_img) + '.h5'
        file_name = RESULTS_FOLDER + file

        cylinders = Cylinders(MAT_SIZE)
        spheres = Spheres(MAT_SIZE)

        chi = cylinders.chi + spheres.chi
        bulk_phase, matrix_projection = get_total_phase(chi, cylinders.vec_theta, cylinders.vec_psi)

        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset(name='phase', data=bulk_phase[DELTA_FOV[0] - 1:MAT_SIZE[0] - DELTA_FOV[0] - 1,
                                                            DELTA_FOV[1] - 1:MAT_SIZE[1] - DELTA_FOV[1] - 1,
                                                            DELTA_FOV[2] - 1:MAT_SIZE[2] - DELTA_FOV[2] - 1, :])
            hf.create_dataset(name='chi', data=chi[DELTA_FOV[0] - 1:MAT_SIZE[0] - DELTA_FOV[0] - 1,
                                                   DELTA_FOV[1] - 1:MAT_SIZE[1] - DELTA_FOV[1] - 1,
                                                   DELTA_FOV[2] - 1:MAT_SIZE[2] - DELTA_FOV[2] - 1, :])
            hf.create_dataset(name='A', data=matrix_projection[DELTA_FOV[0] - 1:MAT_SIZE[0] - DELTA_FOV[0] - 1,
                                                               DELTA_FOV[1] - 1:MAT_SIZE[1] - DELTA_FOV[1] - 1,
                                                               DELTA_FOV[2] - 1:MAT_SIZE[2] - DELTA_FOV[2] - 1, :, :])
        print(file + ' saved')


if __name__ == '__main__':
    main()
