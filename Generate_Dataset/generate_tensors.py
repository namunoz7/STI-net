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

NUM_CYLINDERS = 3
NUM_SPHERES = 3
NUM_ELLIPSOIDS_I = 3
NUM_ELLIPSOIDS_A = 3
MAT_SIZE = np.array([72, 72, 72])
N_ORIENTATIONS = 6
START = 0
NUM_FIGURES = 74000
RESOLUTION = np.array([0.1, 0.1, 0.1])
FOV = MAT_SIZE * RESOLUTION
MAT_SIZE = tuple(MAT_SIZE)
WINDOW = 64
DELTA_FOV = ((np.array(MAT_SIZE) - WINDOW)//2)
RESULTS_FOLDER = '../../../../researchers/cristian-tejos/datasets/STI_dataset/Chi/'
# RESULTS_FOLDER = '../../../Imagenes/Dataset/Chi/'
# RESULTS_FOLDER = '../../Dataset/Chi/'
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
    Generates the magnetic susceptibility eigen values of multiple spheres
    :param n_spheres: Number of spheres in available in the FOV
    :return:
    """
    vec_radius, vec_alpha, vec_beta = rp.rand_parameters_cylinders(n_spheres)
    xx, yy, zz = space.define_space(n_spheres, RESOLUTION, MAT_SIZE)
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

    _, _, mat_rot = gt.transform_cylinders(n_spheres, vec_alpha, vec_beta, RESOLUTION)
    chi = sti.tilt_scan_image(n_spheres, mask_sphere, mat_rot, is_cylinder=False)
    return chi, n_spheres


def gen_ellipsoid(n_ellipsoids, is_anisotropic):
    """
    Generates the magnetic off-resonance field of multiple spheres
    :param is_anisotropic: Tells if the susceptibility values are anisotropic (cylinder)
    :param n_ellipsoids: Number of spheres in available in the FOV
    :return:
    """
    vec_radius, vec_alpha, vec_beta = rp.rand_parameters_ellipsoid(n_ellipsoids)
    xx, yy, zz = space.define_space(n_ellipsoids, RESOLUTION, MAT_SIZE)
    center_nx, center_ny, center_nz = space.get_center(n_ellipsoids, RESOLUTION, MAT_SIZE)
    xx = xx.permute(1, 2, 3, 0) - center_nx
    yy = yy.permute(1, 2, 3, 0) - center_ny
    zz = zz.permute(1, 2, 3, 0) - center_nz

    xx2 = (xx * xx)
    yy2 = (yy * yy)
    zz2 = (zz * zz)

    rho_2 = (xx2 / vec_radius[:, 0]) + (yy2 / vec_radius[:, 1]) + (zz2 / vec_radius[:, 2])
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1
    ellipsoid = torch.le(rho, 1)
    mask_ellipsoid = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], n_ellipsoids, 1,
                                 dtype=torch.float64, device=DEVICE)
    mask_ellipsoid[ellipsoid] = 1

    _, _, mat_rot = gt.transform_cylinders(n_ellipsoids, vec_alpha, vec_beta, RESOLUTION)
    chi = sti.tilt_scan_image(n_ellipsoids, mask_ellipsoid, mat_rot, is_cylinder=is_anisotropic)
    return chi, n_ellipsoids


def spheres_border(num_figures):
    """
    Generates the spheres of the border mask
    :param num_figures:
    :return:
    """
    radius = 8
    centers = rp.center_borders(num_figures, WINDOW)
    num_figures = centers.shape[0]
    xx, yy, zz = space.define_space(centers.shape[0], [1, 1, 1], (WINDOW, WINDOW, WINDOW))
    xx = xx.permute(1, 2, 3, 0) - centers[:, 0] + torch.randint(low=-3, high=3, size=(num_figures,))
    yy = yy.permute(1, 2, 3, 0) - centers[:, 1] + torch.randint(low=-3, high=3, size=(num_figures,))
    zz = zz.permute(1, 2, 3, 0) - centers[:, 2] + torch.randint(low=-3, high=3, size=(num_figures,))
    xx2 = (xx * xx)
    yy2 = (yy * yy)
    zz2 = (zz * zz)

    rho_2 = xx2 + yy2 + zz2
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1
    mask_sphere = rho <= radius
    mask_sphere = torch.sum(mask_sphere, dim=-1)
    mask_sphere = mask_sphere == 0
    return mask_sphere


def bite_borders(tensor, border_mask):
    """
    Bite the borders of the tensor with the mask passed as argument
    :param tensor:
    :param border_mask:
    :return:
    """
    border_mask = border_mask.repeat(tensor.shape[-1], 1, 1, 1).permute(1, 2, 3, 0)
    tensor = tensor * border_mask
    return tensor


def read_h5_file(filename):
    """
    Reads the h5 file
    :param filename:
    :return:
    """
    with h5py.File(filename, 'r') as hf:
        return torch.tensor(np.array(hf.get('A')))


class Cylinders:
    global NUM_CYLINDERS
    global N_ORIENTATIONS

    def __init__(self, mat_size):
        self.vec_theta, self.vec_psi = rp.angles_cylinders(N_ORIENTATIONS)
        self.chi = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        # Iterate to generate more figures
        chi_cylinders, n_figures = gen_cylinders_v2(NUM_CYLINDERS)
        self.chi = self.chi + chi_cylinders


class Spheres:
    global NUM_SPHERES
    global N_ORIENTATIONS

    def __init__(self, mat_size):
        self.chi = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        tmp, n_spheres = gen_spheres_v2(NUM_SPHERES)
        self.chi = self.chi + tmp


class Ellipsoids:
    global NUM_ELLIPSOIDS_I
    global NUM_ELLIPSOIDS_A
    global N_ORIENTATIONS

    def __init__(self, mat_size):
        self.chi_i = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        tmp, n_spheres = gen_ellipsoid(NUM_ELLIPSOIDS_I, is_anisotropic=False)
        self.chi_i = self.chi_i + tmp

        self.chi_a = torch.zeros(mat_size[0], mat_size[1], mat_size[2], 6, dtype=torch.float64, device=DEVICE)
        tmp, n_spheres = gen_ellipsoid(NUM_ELLIPSOIDS_A, is_anisotropic=True)
        self.chi_a = self.chi_a + tmp


def main():
    for n_img in range(START, NUM_FIGURES):
        file = 'data_' + str(n_img) + '.h5'
        file_name = RESULTS_FOLDER + file
        cylinders = Cylinders(MAT_SIZE)
        spheres = Spheres(MAT_SIZE)
        ellipsoids = Ellipsoids(MAT_SIZE)
        chi = cylinders.chi + spheres.chi + ellipsoids.chi_i + ellipsoids.chi_a
        chi = chi[DELTA_FOV[0] - 1:MAT_SIZE[0] - DELTA_FOV[0] - 1,
                  DELTA_FOV[1] - 1:MAT_SIZE[1] - DELTA_FOV[1] - 1,
                  DELTA_FOV[2] - 1:MAT_SIZE[2] - DELTA_FOV[2] - 1, :]
        mask_borders = spheres_border(7)
        chi = bite_borders(chi, mask_borders)

        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset(name='chi', data=chi)
        print(file + ' saved')


if __name__ == '__main__':
    main()
