import torch
import sys
from generate_tensors import MAT_SIZE
eps = sys.float_info.epsilon
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_direction_field(vec_theta, vec_psi, n_orientations):
    """
    Gets the direction field vector of the multiple orientations, made by the cylinders. All the angles are in radians
    :param vec_theta: Rotation angle in the left-right axis
    :param vec_psi: Rotation angle in the antero-posterior axis
    :param n_orientations:
    :return:
    """
    direction_field_1 = torch.zeros(n_orientations, 3, dtype=torch.float64)
    direction_field_2 = torch.zeros(n_orientations, 3, dtype=torch.float64)

    # Orientation of the main magnetic field
    direction_field_1[:, 0] = torch.sin(vec_theta)  # Hx
    direction_field_1[:, 1] = -torch.sin(vec_psi) * torch.cos(vec_theta)  # Hy
    direction_field_1[:, 2] = torch.cos(vec_psi) * torch.cos(vec_theta)  # Hz
    direction_field_1 = direction_field_1.unsqueeze(-1)

    # Opposed direction
    direction_field_2[:, 0] = -torch.sin(vec_theta) * torch.cos(vec_psi)  # Hx
    direction_field_2[:, 1] = torch.sin(vec_psi)  # Hy
    direction_field_2[:, 2] = torch.cos(vec_psi) * torch.cos(vec_theta)  # Hz
    direction_field_2 = direction_field_2.unsqueeze(-1)

    return direction_field_1, direction_field_2


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
