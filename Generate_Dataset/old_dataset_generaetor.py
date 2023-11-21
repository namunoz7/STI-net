# -*- coding: utf-8 -*-
"""
Generates the geometric figures of the train and validation data

n_maxCylinders = Maximum number of cylinders used in every image
n_maxSpheres = Maximum number of spheres used in every image
N = size of the image
n_orientations = Number of orientations of the geometric phantoms used to 
                 simulate the phase tensor
"""

import numpy as np
from numpy.random import randint
from numpy.random import random_sample
from matplotlib import pyplot as plt
import torch
import sys

N_CYLINDERS = 128
n_maxSpheres = 128
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

    xx, yy, zz = torch.meshgrid(x, y, z)

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

    v_beta = n_max_psi * random_sample(size=num_cylinders) - 0.5 * n_max_psi
    v_beta = v_beta * np.pi / 180

    return [vec_radius_cylinders, num_cylinders, v_alpha, v_beta]


def rotate_cylinders(xx, yy, zz, num_cylinders, vec_alpha, vec_beta, center_nx, center_ny, center_nz, matrix_size):
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

    xx = xx.permute(-1, 0, 1, 2).reshape(new_shape)
    yy = yy.permute(-1, 0, 1, 2).reshape(new_shape)
    zz = zz.permute(-1, 0, 1, 2).reshape(new_shape)

    tensor_1 = torch.ones(xx.size())
    original_positions = torch.stack([xx, yy, zz, tensor_1]).permute(1, 0, 2)

    # Rotation of the cylinder
    print('     Moving the cylinders to the center of the space')
    tmp = torch.matmul(translation_1, original_positions)
    print('     Rotating them outside of the field')
    tmp = torch.matmul(rotation_z, tmp)
    print('     Rotating them in the x,y plane')
    tmp = torch.matmul(rotation_x, tmp)
    print('     Moving them back to their center')
    new_positions = torch.matmul(translation_2, tmp)

    print('Generating the new xx, yy variables')
    xx = new_positions[:, 0, :].reshape(num_cylinders, matrix_size[0], matrix_size[1], matrix_size[2])
    yy = new_positions[:, 1, :].reshape(num_cylinders, matrix_size[0], matrix_size[1], matrix_size[2])

    xx = xx.permute(1, 2, 3, 0)
    yy = yy.permute(1, 2, 3, 0)
    return xx, yy


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


def gen_cylinders(s_radius, s_theta, s_psi, N, v_theta, v_psi, n_orientations):
    """
    Generates one cylinder reoriented with the angles as input.
    Inputs: s_radius = radius size of the cylinder
            N = field of view in the cylinder
            s_theta = angle of rotation of the cylinder with respect to the Z
                      axis
            [v_theta, v_psi] = angle of rotation with respect to the main magnetic field
    """

    # Center of the cylinder
    center_Nx = randint(1 + s_radius, N[0] - s_radius)
    center_Ny = randint(1 + s_radius, N[1] - s_radius)
    center_Nz = randint(1 + s_radius, N[2] - s_radius)

    x = np.arange(1, N[0] + 1)
    y = np.arange(1, N[1] + 1)
    z = np.arange(1, N[2] + 1)

    x = x - center_Nx
    y = y - center_Ny
    z = z - center_Nz

    xx, yy, zz = np.meshgrid(x, y, z)

    # Rotation matrix to rotate theta with respect to the X axis
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(s_theta), -np.sin(s_theta), 0],
                   [0, -np.sin(s_theta), np.cos(s_theta), 0],
                   [0, 0, 0, 1]])
    # Rotation matrix of psi with respect to the Z axis
    Rz = np.array([[np.cos(s_psi), -np.sin(s_psi), 0, 0],
                   [np.sin(s_psi), np.cos(s_psi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Translation matrices
    T1 = np.array([[1, 0, 0, center_Nx],
                   [0, 1, 0, center_Ny],
                   [0, 0, 1, center_Nz],
                   [0, 0, 0, 1]])
    T2 = np.array([[1, 0, 0, -center_Nx],
                   [0, 1, 0, -center_Ny],
                   [0, 0, 1, -center_Nz],
                   [0, 0, 0, 1]])

    xx = np.reshape(xx, (np.size(xx)))
    yy = np.reshape(yy, (np.size(yy)))
    zz = np.reshape(zz, (np.size(zz)))
    unos = np.ones(np.size(xx))

    original_positions = np.concatenate([[xx],
                                         [yy],
                                         [zz],
                                         [unos]])
    # Rotation of the cylinder
    new_positions = T2 @ Rx @ Rz @ T1 @ original_positions

    del original_positions

    xx = np.reshape(new_positions[0, :], N)
    yy = np.reshape(new_positions[1, :], N)

    rho = np.sqrt(xx * xx + yy * yy)
    cylinder = torch.zeros(N[0], N[1], N[2], 1)
    cylinder[rho < s_radius] = 1
    Xi, Xe = get_susceptibility_values()
    DeltaX = (Xi - Xe)
    tmp = torch.sin(v_theta)
    phase_in = DeltaX * ((0.5 * tmp * tmp) - (1 / 3))
    phase_in = cylinder * phase_in

    cylinder = torch.logical_not(cylinder)
    r2_rho2 = (s_radius * s_radius) / (rho * rho)
    r2_rho2 = torch.Tensor(r2_rho2)
    r2_rho2 = r2_rho2.reshape(N[0], N[1], N[2], 1)
    tmp_1 = torch.cos(2 * v_psi)
    tmp_1 = tmp_1.reshape(1, n_orientations)
    tmp = tmp_1 * tmp * tmp
    phase_out = 0.5 * DeltaX * torch.matmul(r2_rho2, tmp)
    phase_out = cylinder * phase_out

    bulk_phase_cylinder = phase_in + phase_out

    return bulk_phase_cylinder


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
    xx, yy = rotate_cylinders(xx, yy, zz,
                              num_cylinders,
                              vec_alpha, vec_beta,
                              center_nx, center_ny, center_nz,
                              matrix_size)

    xx2 = xx * xx
    yy2 = yy * yy
    rho = torch.sqrt(xx2 + yy2)

    # Define susceptibility values
    chi_i, chi_e = get_susceptibility_values(num_cylinders)
    delta_x = (chi_i - chi_e)
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


def rand_radius_spheres(n_max_spheres, matrix_size):
    num_spheres = randint(1, n_max_spheres)

    max_r_spheres = np.min(matrix_size) / 4
    max_r_spheres = np.round(max_r_spheres)
    v_radius = randint(15, max_r_spheres, size=num_spheres)

    return [v_radius, num_spheres]


def gen_spheres(s_radius, N):
    """
    Generates one sphere reoriented with the angles as input.
    Inputs: s_radius = radius size of the sphere
            N = field of view in the sphere
            s_theta = angle of rotation of the sphere with respect to the Z
                      axis
    """
    # Center of the sphere
    center_Nx = randint(1 + s_radius, N[0] - s_radius)
    center_Ny = randint(1 + s_radius, N[1] - s_radius)
    center_Nz = randint(1 + s_radius, N[2] - s_radius)

    x = np.arange(1, N[0] + 1)
    y = np.arange(1, N[1] + 1)
    z = np.arange(1, N[2] + 1)

    x = x - center_Nx
    y = y - center_Ny
    z = z - center_Nz

    xx, yy, zz = np.meshgrid(x, y, z)

    xx2 = xx * xx
    yy2 = yy * yy
    zz2 = zz * zz

    R = np.sqrt(np.square(xx) +
                np.square(yy) +
                np.square(zz))

    R[center_Nx, center_Ny, center_Nz] = 1

    # tmp_radius = (xx * xx) + (yy * yy) + (zz * zz)

    [Xi, Xe] = get_susceptibility_values()
    DeltaX = Xi - Xe

    sphere = np.zeros(N)
    sphere[R <= s_radius] = Xi
    sphere[R > s_radius] = Xe

    chi = (1 - (2 / 3) * sphere) * ((s_radius ** 3) * (DeltaX / (3 + DeltaX))) * (
            (2 * zz2 - xx2 - yy2) / ((R ** 2) ** (5 / 2)))
    chi[R < s_radius] = 0
    sphere_mask = np.zeros(N)
    sphere_mask[R <= s_radius] = 1
    return [chi, sphere_mask]


def gen_spheres_v2(matrix_size, num_max_spheres):
    """
    Generates the magnetic off-resonance field of multiple spheres
    :param num_max_spheres: Maximum number of spheres in available in the FOV
    :param matrix_size: Size of the image
    :return:
    """

    vec_radius, num_spheres = rand_radius_spheres(num_max_spheres, N)
    vec_radius = torch.Tensor(vec_radius)

    print('Creating ' + str(num_spheres) + ' spheres')

    print('Defining space for spheres')
    xx, yy, zz, center_nx, center_ny, center_nz = define_space(vec_radius, matrix_size, num_spheres)

    xx2 = xx * xx
    yy2 = yy * yy
    zz2 = zz * zz

    rho_2 = xx2 + yy2 + zz2
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1

    [chi_i, chi_o] = get_susceptibility_values(num_spheres)
    delta_chi = chi_o - chi_i

    print('Defining spheres susceptibility')
    tmp = rho <= vec_radius
    chi_sphere = torch.zeros(N[0], N[1], N[2], num_spheres)
    chi_sphere[tmp] = 1
    # chi_sphere = torch.matmul(chi_sphere, chi_i)
    chi_sphere = chi_sphere * chi_i

    tmp = rho > vec_radius
    chi_sphere_out = torch.zeros(N[0], N[1], N[2], num_spheres)
    chi_sphere_out[tmp] = 1
    # chi_sphere_out = torch.matmul(chi_sphere_out, chi_o)
    chi_sphere_out = chi_sphere_out * chi_o
    chi_sphere = chi_sphere + chi_sphere_out

    print('Define phase of sphere')
    chi = (1 - (2 / 3) * chi_sphere) * ((vec_radius ** 3) * (delta_chi / (3 + delta_chi))) * (
            (2 * zz2 - xx2 - yy2) / (rho_2 ** (5 / 2)))
    chi[rho < vec_radius] = 0
    chi = chi.sum(-1)
    return [chi]


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


def projection_variables(vec_theta, vec_psi, matrix_size, fov):
    """
    Generates the projection variables of the STI model (a_ii and a_ij) and construct the projection matrix
    :param fov: Field Of View of the entire space
    :param matrix_size: size of the space
    :param vec_theta: vector containing the angle of deviation with respect to the main field axis
    :param vec_psi: vector containing the angle of rotation of the x-y plane.
    :return: A: matrix containing each one of the projection angles.
    """

    sn_theta = torch.sin(vec_theta)
    cs_theta = torch.cos(vec_theta)
    sn_psi = torch.sin(vec_psi)
    cs_psi = torch.cos(vec_psi)

    direction_field_x = sn_theta * cs_psi
    direction_field_y = sn_theta * sn_psi
    direction_field_z = cs_theta

    direction_field = torch.zeros(3, N_ORIENTATIONS)
    direction_field[0, :] = direction_field_x
    direction_field[1, :] = direction_field_y
    direction_field[2, :] = direction_field_z

    direction_field = direction_field.reshape(N_ORIENTATIONS, 3, 1)

    x = np.arange(1, matrix_size[0] + 1)
    y = np.arange(1, matrix_size[1] + 1)
    z = np.arange(1, matrix_size[2] + 1)

    delta_kx = fov[0] / matrix_size[0]
    delta_ky = fov[1] / matrix_size[1]
    delta_kz = fov[2] / matrix_size[2]

    kx = x * delta_kx
    ky = y * delta_ky
    kz = z * delta_kz

    kxx, kyy, kzz = np.meshgrid(kx, ky, kz)
    kxx = torch.Tensor(kxx)
    kyy = torch.Tensor(kyy)
    kzz = torch.Tensor(kzz)

    kxx = kxx.unsqueeze(3).repeat(1, 1, 1, N_ORIENTATIONS)
    kyy = kyy.unsqueeze(3).repeat(1, 1, 1, N_ORIENTATIONS)
    kzz = kzz.unsqueeze(3).repeat(1, 1, 1, N_ORIENTATIONS)

    k = torch.zeros(matrix_size[0], matrix_size[1], matrix_size[2], N_ORIENTATIONS, 3)
    k[:, :, :, :, 0] = kxx
    k[:, :, :, :, 1] = kyy
    k[:, :, :, :, 2] = kzz
    k = k.reshape(matrix_size[0], matrix_size[1], matrix_size[2], N_ORIENTATIONS, 3, 1)
    k_transpose = k.transpose(4, 5)
    k2 = torch.matmul(k_transpose, k).squeeze()
    k_th = torch.matmul(k_transpose, direction_field).squeeze()

    # Aux variables are defined
    a_11 = ((direction_field_x * direction_field_x) / 3) - ((k_th / k2) * kxx * direction_field_x)
    a_22 = ((direction_field_y * direction_field_y) / 3) - ((k_th / k2) * kyy * direction_field_y)
    a_33 = ((direction_field_z * direction_field_z) / 3) - ((k_th / k2) * kzz * direction_field_z)
    a_12 = ((2 / 3) * direction_field_x * direction_field_y) - \
           ((k_th / k2) * (kxx * direction_field_y + kyy * direction_field_x))
    a_13 = ((2 / 3) * direction_field_x * direction_field_z) - \
           ((k_th / k2) * (kxx * direction_field_z + kzz * direction_field_x))
    a_23 = ((2 / 3) * direction_field_z * direction_field_y) - \
           ((k_th / k2) * (kzz * direction_field_y + kyy * direction_field_z))

    matrix_projection = torch.zeros(matrix_size[0], matrix_size[1], matrix_size[2], N_ORIENTATIONS, 6)
    matrix_projection[:, :, :, :, 0] = a_11
    matrix_projection[:, :, :, :, 1] = a_12
    matrix_projection[:, :, :, :, 2] = a_13
    matrix_projection[:, :, :, :, 3] = a_22
    matrix_projection[:, :, :, :, 4] = a_23
    matrix_projection[:, :, :, :, 5] = a_33

    return matrix_projection


r_cylinders, n_cylinders, v_thetaCylinders, v_psiCylinders = rand_parameters_cylinders(N_CYLINDERS, N)
[v_theta, v_psi] = angles_cylinders(N_ORIENTATIONS)

print('Generating matrix projection')
A = projection_variables(v_theta, v_psi, N, FOV)
print()
print('Generating cylinders')
cylinder_v2 = gen_cylinders_v2(r_cylinders,
                               v_thetaCylinders,
                               v_psiCylinders, N,
                               v_theta, v_psi, n_cylinders)

print()
print('Generating spheres')
spheres_v2 = gen_spheres_v2(N, n_maxSpheres)
spheres_v2 = spheres_v2[0]
spheres_v2 = spheres_v2.repeat(N_ORIENTATIONS, 1, 1, 1).permute(1, 2, 3, 0)

bulk_phase = cylinder_v2 + spheres_v2

print()
print('Printing image')
img = bulk_phase[:, :, 64, 0]
plt.figure()
plt.imshow(img, cmap='gray')
plt.colorbar(shrink=1)
plt.show()

img = bulk_phase[:, :, 64, 1]
plt.figure()
plt.imshow(img, cmap='gray')
plt.colorbar(shrink=1)
plt.show()
