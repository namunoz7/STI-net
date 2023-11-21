import torch
import numpy as np
import x_space_torch as space
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rotate_x(num_figures, vec_omega):
    """
    Rotates the space with the x axis
    :param num_figures: Number of geometric figures inside the window
    :param vec_omega: Rotation angles of the y-z plane
    :return: Rotation matrix of the x axis
    """
    tensor_1 = torch.ones(num_figures, dtype=torch.float64, device=DEVICE)
    tensor_0 = torch.zeros(num_figures, dtype=torch.float64, device=DEVICE)
    rotation_x = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0]),
                              torch.stack([tensor_0, torch.cos(vec_omega), -torch.sin(vec_omega)]),
                              torch.stack([tensor_0, torch.sin(vec_omega), torch.cos(vec_omega)])]).permute(2, 0, 1)
    return rotation_x


def rotate_y(num_cylinders, vec_alpha):
    """
    Rotates the space with the y axis
    :param num_cylinders:
    :param vec_alpha: Rotation angles in the x-z plane
    :return: Rotation matrix of the y axis
    """
    tensor_1 = torch.ones(num_cylinders, dtype=torch.float64, device=DEVICE)
    tensor_0 = torch.zeros(num_cylinders, dtype=torch.float64, device=DEVICE)
    rotation_y = torch.stack([torch.stack([torch.cos(vec_alpha), tensor_0, torch.sin(vec_alpha)]),
                              torch.stack([tensor_0, tensor_1, tensor_0]),
                              torch.stack([-torch.sin(vec_alpha), tensor_0, torch.cos(vec_alpha)])]).permute(2, 0, 1)
    return rotation_y


def rotate_z(num_cylinders, vec_beta):
    """
    Rotates the cylinders with the z axis
    :param num_cylinders:
    :param vec_beta: Rotation angles with in the x-y plane
    :return:
    """
    tensor_1 = torch.ones(num_cylinders, dtype=torch.float64, device=DEVICE)
    tensor_0 = torch.zeros(num_cylinders, dtype=torch.float64, device=DEVICE)

    rotation_z = torch.stack([torch.stack([torch.cos(vec_beta), -torch.sin(vec_beta), tensor_0]),
                              torch.stack([torch.sin(vec_beta), torch.cos(vec_beta), tensor_0]),
                              torch.stack([tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)

    return rotation_z


def translate_1(num_cylinders, center_nx, center_ny, center_nz):
    """
    Translate the center to the position (0, 0, 0)
    :param num_cylinders: number of cylinders
    :param center_nx: center of the cylinders in the x axis
    :param center_ny: center of the cylinders in the y axis
    :param center_nz: center of the cylinders in the z axis
    :return:
    """
    torch.cuda.empty_cache()
    tensor_1 = torch.ones(num_cylinders, dtype=torch.float64, device=DEVICE)
    tensor_0 = torch.zeros(num_cylinders, dtype=torch.float64, device=DEVICE)
    translation_1 = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0, center_nx]),
                                 torch.stack([tensor_0, tensor_1, tensor_0, center_ny]),
                                 torch.stack([tensor_0, tensor_0, tensor_1, center_nz]),
                                 torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)  # .to(device)

    return translation_1


def translate_2(num_cylinders, center_nx, center_ny, center_nz):
    """

    :param num_cylinders:
    :param center_nx:
    :param center_ny:
    :param center_nz:
    :return:
    """
    tensor_1 = torch.ones(num_cylinders, dtype=torch.float64, device=DEVICE)
    tensor_0 = torch.zeros(num_cylinders, dtype=torch.float64, device=DEVICE)
    translation_2 = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0, -center_nx]),
                                 torch.stack([tensor_0, tensor_1, tensor_0, -center_ny]),
                                 torch.stack([tensor_0, tensor_0, tensor_1, -center_nz]),
                                 torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).permute(2, 0, 1)  # .to(device)
    return translation_2


def transform_cylinders(num_cylinders, vec_alpha, vec_beta, spatial_res):
    """
    Transform the space to define the cylinders
    :param spatial_res: Spatial resolution of the image
    :param num_cylinders: Number of cylinders to simulate
    :param vec_alpha: Rotation angle in the x-z plane
    :param vec_beta: Rotation angle in the x-y plane
    :return: [xx, yy] work space of the simulation
    """
    from generate_tensors import MAT_SIZE
    new_shape = (num_cylinders, np.prod(MAT_SIZE))

    xx, yy, zz = space.define_space(num_cylinders, spatial_res, MAT_SIZE)
    original_shape = xx.shape
    center_nx, center_ny, center_nz = space.get_center(num_cylinders, spatial_res, MAT_SIZE)
    original_center = torch.stack([center_nx, center_ny, center_nz], dim=1)
    original_center = original_center.unsqueeze(-1).to(torch.float64)

    xx = xx.reshape(new_shape)
    yy = yy.reshape(new_shape)
    zz = zz.reshape(new_shape)

    original_positions = torch.stack([xx, yy, zz], dim=1)

    # Rotation of the cylinder
    rot_z = rotate_z(num_cylinders, vec_beta)
    rot_y = rotate_y(num_cylinders, vec_alpha)
    mat_rot = torch.matmul(rot_y, rot_z)
    new_positions = torch.matmul(mat_rot, original_positions)
    new_center = torch.matmul(mat_rot, original_center)
    xx = new_positions[:, 0, :].reshape(original_shape)
    yy = new_positions[:, 1, :].reshape(original_shape)

    center_nx = new_center[:, 0, 0]
    center_ny = new_center[:, 1, 0]

    xx = xx.permute(1, 2, 3, 0) - center_nx
    yy = yy.permute(1, 2, 3, 0) - center_ny
    return xx, yy, mat_rot
