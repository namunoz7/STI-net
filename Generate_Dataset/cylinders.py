# -*- coding: utf-8 -*-
"""
Generates the cylinders to train the Neural Network
"""

import numpy as np
from numpy.random import randint
from numpy.random import random_sample
from matplotlib import pyplot as plt
import torch
import sys

N_CYLINDERS = 128
N = [128, 128, 128]
N_ORIENTATIONS = 12
FOV = [30, 20, 20]
eps = sys.float_info.epsilon


def define_space(vec_radius, matrix_size, num_figures):
    """
    Create the tensors necessary to simulate the cylinders and spheres
    :param vec_radius: Vector containing the radius of the figures
    :param matrix_size: Size of the space in x, y and z
    :param num_figures: number of figures to simulate
    :return: xx, yy, zz: the space
    :return: center_nx, center_ny, center_nz: center of each figure
    """
    x = torch.arange(1, matrix_size[0] + 1)
    y = torch.arange(1, matrix_size[1] + 1)
    z = torch.arange(1, matrix_size[2] + 1)

    xx, yy, zz = torch.meshgrid([x, y, z])

    new_shape = tuple(matrix_size) + (num_figures,)

    xx = xx.repeat(1, 1, 1, num_figures)
    yy = yy.repeat(1, 1, 1, num_figures)
    zz = zz.repeat(1, 1, 1, num_figures)

    xx = xx.reshape(new_shape)
    yy = yy.reshape(new_shape)
    zz = zz.reshape(new_shape)

    # Center of the cylinder
    min_center = torch.ones(num_figures) + vec_radius
    max_center_x = matrix_size[0] * torch.ones(num_figures) - vec_radius
    max_center_y = matrix_size[1] * torch.ones(num_figures) - vec_radius
    max_center_z = matrix_size[2] * torch.ones(num_figures) - vec_radius
    center_nx = min_center + (max_center_x - min_center) * torch.rand(num_figures)
    center_ny = min_center + (max_center_y - min_center) * torch.rand(num_figures)
    center_nz = min_center + (max_center_z - min_center) * torch.rand(num_figures)

    center_nx = center_nx.round()
    center_ny = center_ny.round()
    center_nz = center_nz.round()

    xx = xx - center_nx
    yy = yy - center_ny
    zz = zz - center_nz

    return [xx, yy, zz, center_nx, center_ny, center_nz]


def rand_parameters_cylinders(n_max_cylinders, matrix_size):
    """
    Generates necessary parameters to simulate the cylinders
    """
    n_max_theta = 180.0
    n_max_psi = 360.0
    num_cylinders = randint(1, n_max_cylinders)

    max_r_cylinders = np.min(matrix_size) / 4
    max_r_cylinders = np.round(max_r_cylinders)
    vec_radius_cylinders = randint(5, max_r_cylinders, size=num_cylinders)

    v_alpha = n_max_theta * random_sample(size=num_cylinders) - 0.5 * n_max_theta
    v_alpha = v_alpha * np.pi / 180
    v_alpha = torch.Tensor(v_alpha)

    v_beta = n_max_psi * random_sample(size=num_cylinders) - 0.5 * n_max_psi
    v_beta = v_beta * np.pi / 180
    v_beta = torch.Tensor(v_beta)

    return [num_cylinders, vec_radius_cylinders, v_alpha, v_beta]


def angles_cylinders(num_rotations):
    """
    Rotation angles that are used to rotate the object in the scanner
    """
    # Parameters of the angles to model the cylinders
    theta_max = 180.0
    psi_max = 360.0
    big = 90.0
    medium = 45.0
    small = 0.0

    # Tilt angle of main field
    tmp_theta = np.array([big, big, medium, big, medium, small])
    vec_theta = theta_max * random_sample((num_rotations - 6))
    vec_theta = np.concatenate((tmp_theta, vec_theta))
    vec_theta = vec_theta * np.pi / 180.0
    vec_theta = torch.Tensor(vec_theta)
    vec_theta = vec_theta.reshape(1, num_rotations)

    # Rotation angle with the z axis
    tmp_psi = np.array([small, medium, medium, big, big, small])
    vec_psi = psi_max * random_sample((num_rotations - 6))
    vec_psi = np.concatenate((tmp_psi, vec_psi))
    vec_psi = vec_psi * np.pi / 180.0
    vec_psi = torch.Tensor(vec_psi)
    vec_psi = vec_psi.reshape(1, num_rotations)

    return [vec_theta, vec_psi]


def geometric_transformation(xx, yy, zz,
                             num_cylinders,
                             vec_alpha, vec_beta,
                             center_nx, center_ny, center_nz,
                             matrix_size):
    """
    Transform the space to define the cylinders
    :param xx:
    :param yy:
    :param zz:
    :param num_cylinders:
    :param vec_alpha:
    :param vec_beta:
    :param center_nx:
    :param center_ny:
    :param center_nz:
    :param matrix_size:
    :return:
    """
    tensor_1 = torch.ones(num_cylinders)
    tensor_0 = torch.zeros(num_cylinders)

    # Rotation matrix to rotate theta with respect to the X axis
    rotation_x = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0, tensor_0]),
                              torch.stack([tensor_0, torch.cos(vec_alpha), -torch.sin(vec_alpha), tensor_0]),
                              torch.stack([tensor_0, torch.sin(vec_alpha), torch.cos(vec_alpha), tensor_0]),
                              torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)
    # Rotation matrix of psi with respect to the Z axis
    rotation_z = torch.stack([torch.stack([torch.cos(vec_beta), -torch.sin(vec_beta), tensor_0, tensor_0]),
                              torch.stack([torch.sin(vec_beta), torch.cos(vec_beta), tensor_0, tensor_0]),
                              torch.stack([tensor_0, tensor_0, tensor_1, tensor_0]),
                              torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)

    # Translation matrices
    translation_1 = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0, center_nx]),
                                 torch.stack([tensor_0, tensor_1, tensor_0, center_ny]),
                                 torch.stack([tensor_0, tensor_0, tensor_1, center_nz]),
                                 torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)

    translation_2 = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0, -center_nx]),
                                 torch.stack([tensor_0, tensor_1, tensor_0, -center_ny]),
                                 torch.stack([tensor_0, tensor_0, tensor_1, -center_nz]),
                                 torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)

    new_shape = (num_cylinders, -1)

    print("Preparing data to make geometric transform")
    xx = xx.permute(-1, 0, 1, 2).reshape(new_shape)
    yy = yy.permute(-1, 0, 1, 2).reshape(new_shape)
    zz = zz.permute(-1, 0, 1, 2).reshape(new_shape)

    tensor_1 = torch.ones(xx.size())
    original_positions = torch.stack([xx, yy, zz, tensor_1]).permute(1, 0, 2)

    # Rotation of the cylinder
    print('     Moving the cylinders to the center of the space')
    tmp = torch.matmul(translation_1, original_positions)
    print('     Rotating them outside of the field')
    tmp = torch.matmul(rotation_z, tmp.float())
    print('     Rotating them in the x,y plane')
    tmp = torch.matmul(rotation_x, tmp.float())
    print('     Moving them back to their center')
    new_positions = torch.matmul(translation_2.float(), tmp.float())

    print('Generating the new xx, yy variables')
    xx = new_positions[:, 0, :].reshape(num_cylinders, matrix_size[0], matrix_size[1], matrix_size[2])
    yy = new_positions[:, 1, :].reshape(num_cylinders, matrix_size[0], matrix_size[1], matrix_size[2])

    xx = xx.permute(1, 2, 3, 0)
    yy = yy.permute(1, 2, 3, 0)
    return xx, yy


def gen_cylinders_v2(vec_radius, vec_alpha, vec_beta, matrix_size, vec_theta, vec_psi, num_cylinders):
    """
    Generates the off-resonance field of multiple cylinders at different rotation angles with the magnetic field
    :param vec_radius: Vector containing the radius of the multiple cylinders
    :param vec_alpha: Inclination angles of each cylinder in the space
    :param vec_beta: Rotation angle of x,y plane
    :param matrix_size: Matrix size of the image
    :param vec_theta: Rotation angle with respect to the field
    :param vec_psi: Rotation angle with respect to the axial plane
    :param num_cylinders: Number of cylinders in the image
    :return: Tensor of size (N[0], N[1], N[2], n_orientations) with the off-resonance field of the cylinders in every
             orientation
    """

    print('Creating ' + str(num_cylinders) + ' cylinders')

    vec_radius = torch.Tensor(vec_radius)
    vec_alpha = torch.Tensor(vec_alpha)
    vec_beta = torch.Tensor(vec_beta)

    xx, yy, zz, center_nx, center_ny, center_nz = define_space(vec_radius,
                                                               matrix_size,
                                                               num_cylinders)

    print('Rotating all the cylinders')
    xx, yy = geometric_transformation(xx, yy, zz,
                                      num_cylinders,
                                      vec_alpha, vec_beta,
                                      center_nx, center_ny, center_nz,
                                      matrix_size)

    xx2 = xx * xx
    yy2 = yy * yy
    rho = torch.sqrt(xx2 + yy2)

    # Define susceptibility values
    chi_i, chi_e = get_susceptibility_values(num_cylinders)
    delta_x = (chi_e - chi_i)
    tmp_cylinder = rho < vec_radius

    print('Defining phase values')
    # Phase inside the cylinder
    cylinder = torch.zeros(matrix_size[0], matrix_size[1], matrix_size[2], num_cylinders)
    cylinder[tmp_cylinder] = 1.0
    delta_x_i = torch.matmul(cylinder, delta_x)
    delta_x_i = delta_x_i.reshape(matrix_size[0], matrix_size[1], matrix_size[2], 1)
    tmp_sin = torch.sin(vec_theta)
    tmp = ((0.5 * tmp_sin * tmp_sin) - (1 / 3))
    phase_in = torch.matmul(delta_x_i, tmp)

    # Phase outside the cylinder
    cylinder = torch.ones(matrix_size[0], matrix_size[1], matrix_size[2], num_cylinders)
    cylinder[tmp_cylinder] = 0.0
    rho[rho == 0] = 1.0
    r2_rho2 = (vec_radius * vec_radius) / (rho * rho)
    cylinder = cylinder * r2_rho2
    delta_x_o = torch.matmul(cylinder, delta_x)
    # delta_x_o = delta_x_o * r2_rho2
    delta_x_o = delta_x_o.reshape(matrix_size[0], matrix_size[1], matrix_size[2], 1)
    cos_2psi = torch.cos(2 * vec_psi)
    sin2_theta = torch.sin(vec_theta)
    tmp = cos_2psi * sin2_theta * sin2_theta
    phase_out = 0.5 * torch.matmul(delta_x_o, tmp)

    bulk_phase_cylinder = phase_in + phase_out

    return bulk_phase_cylinder


def get_susceptibility_values(n_figures):
    # Susceptibility outside the cylinder
    a = 0.36e-6
    b = -0.45e-6
    chi_o = (b - a) * torch.rand(n_figures) + a

    # Susceptibility inside the cylinder
    a = 0.72e-6
    b = -0.81e-6
    chi_i = (b - a) * torch.rand(n_figures) + a

    return [chi_i, chi_o]


class Cylinders:
    global N_CYLINDERS
    global N_ORIENTATIONS

    def __init__(self, matrix_size, field_of_view):
        self.matrix_size = matrix_size
        self.FOV = field_of_view
        self.vec_theta, self.vec_psi = angles_cylinders(N_ORIENTATIONS)
        self.num_cylinders, vec_radius, vec_alpha, vec_beta = rand_parameters_cylinders(N_CYLINDERS, matrix_size)
        print("Generating " + str(self.num_cylinders) + " cylinders")
        self.local_phase = gen_cylinders_v2(vec_radius,
                                            vec_alpha, vec_beta,
                                            self.matrix_size,
                                            self.vec_theta, self.vec_psi,
                                            self.num_cylinders)


def main():
    cylinders = Cylinders(N, FOV)
    print()
    print('Printing image')
    img = cylinders.local_phase[:, :, 64, 0]
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.colorbar(shrink=1)
    plt.show()

    img = cylinders.local_phase[:, :, 64, 1]
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.colorbar(shrink=1)
    plt.show()


if __name__ == '__main__':
    main()
