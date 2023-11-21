import torch
import sys
from generate_tensors import MAT_SIZE
eps = sys.float_info.epsilon
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_direction_field(vec_theta, vec_psi, n_orientations):
    """
    Gets the direction field vector of the multiple orientations, made by the cylinders.
    :param vec_theta: tilt angle (inclination angle with respect to the z axis)
    :param vec_psi: azimuth angle (rotation angle made in the x-y plane)
    :param n_orientations:
    :return:
    """
    direction_field = torch.zeros(n_orientations, 3, dtype=torch.float64, device=DEVICE)
    direction_field[:, 0] = torch.sin(vec_theta) * torch.cos(vec_psi)  # Hx
    direction_field[:, 1] = -torch.sin(vec_psi)  # Hz
    direction_field[:, 2] = torch.cos(vec_theta) * torch.cos(vec_psi)  # Hy
    direction_field = direction_field.unsqueeze(-1)

    return direction_field


def gen_k_space(fov, n_orientations):
    """
    Defines the K space
    :param fov:
    :param n_orientations:
    :return:
    """
    kx = torch.arange(1, MAT_SIZE[0] + 1, dtype=torch.float64, device=DEVICE)
    ky = torch.arange(1, MAT_SIZE[1] + 1, dtype=torch.float64, device=DEVICE)
    kz = torch.arange(1, MAT_SIZE[2] + 1, dtype=torch.float64, device=DEVICE)

    center_x = MAT_SIZE[0] // 2 + 1
    center_y = MAT_SIZE[1] // 2 + 1
    center_z = MAT_SIZE[2] // 2 + 1
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

    k = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], n_orientations, 3, dtype=torch.float64, device=DEVICE)
    k[:, :, :, :, 0] = kxx
    k[:, :, :, :, 1] = kyy
    k[:, :, :, :, 2] = kzz
    k = k.unsqueeze(-1)

    return k, kxx, kyy, kzz
