import torch
import numpy as np
import x_space_torch as space
from numpy.random import randint
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rand_parameters_cylinders(num_figures):
    """
    Generates necessary parameters to simulate the cylinders
    :param num_figures:
    """
    from generate_tensors import RESOLUTION
    delta_radius = np.linalg.norm(RESOLUTION)

    n_max_alpha = 60.0
    n_max_beta = 60.0
    min_angle = -60.0

    max_r_cylinders = 15
    vec_radius_cylinders = delta_radius * torch.randint(5, max_r_cylinders, (num_figures,),
                                                        dtype=torch.float64, device=DEVICE)

    v_alpha = (n_max_alpha - min_angle) * torch.rand(num_figures, dtype=torch.float64, device=DEVICE) + min_angle
    v_beta = (n_max_beta - min_angle) * torch.rand(num_figures, dtype=torch.float64, device=DEVICE) + min_angle

    v_beta = torch.deg2rad(v_beta)
    v_alpha = torch.deg2rad(v_alpha)

    return vec_radius_cylinders, v_alpha, v_beta  # , v_omega


def rand_parameters_ellipsoid(num_figures):
    """
    Generates necessary parameters to simulate the ellipsoids
    :param num_figures:
    """
    from generate_tensors import RESOLUTION
    delta_radius = np.linalg.norm(RESOLUTION)

    n_max_alpha = 60.0
    n_max_beta = 60.0
    min_angle = -60.0

    max_r_ellipsoids = 20
    vec_radius_ellipsoids = delta_radius * torch.randint(8, max_r_ellipsoids, (num_figures, 3),
                                                         dtype=torch.float64, device=DEVICE)

    v_alpha = (n_max_alpha - min_angle) * torch.rand(num_figures, dtype=torch.float64, device=DEVICE) + min_angle
    v_beta = (n_max_beta - min_angle) * torch.rand(num_figures, dtype=torch.float64, device=DEVICE) + min_angle

    v_beta = v_beta.deg2rad()
    v_alpha = v_alpha.deg2rad()

    return vec_radius_ellipsoids, v_alpha, v_beta


def center_borders(num_figures, mat_size):
    """
    Get the center of the spheres for each image
    :param num_figures: The maximum number of figures in the complete volume
    :param mat_size: Maximum size of the shape of the volume
    :return:
    """
    centers = torch.linspace(0, mat_size, num_figures)
    tmp = centers
    centers = torch.stack([tmp, torch.zeros(num_figures)], dim=1)
    for n in range(1, num_figures):
        tmp2 = torch.stack([tmp, torch.full(size=(num_figures,), fill_value=tmp[n])], dim=1)
        centers = torch.cat([centers, tmp2], dim=0)
    x_0 = torch.zeros(centers.shape[0])
    x_1 = torch.full(size=(centers.shape[0],), fill_value=mat_size)
    center_1 = torch.stack([x_0, centers[:, 0], centers[:, 1]], dim=1)
    center_2 = torch.stack([x_1, centers[:, 0], centers[:, 1]], dim=1)
    center_3 = torch.stack([centers[:, 0], x_0, centers[:, 1]], dim=1)
    center_4 = torch.stack([centers[:, 0], x_1, centers[:, 1]], dim=1)
    center_5 = torch.stack([centers[:, 0], centers[:, 1], x_0], dim=1)
    center_6 = torch.stack([centers[:, 0], centers[:, 1], x_1], dim=1)
    center = torch.cat([center_1, center_2, center_3, center_4, center_5, center_6], dim=0)
    return center


def spheres_border(num_figures, window):
    """
    Generates the spheres of the border mask
    :param window:
    :param num_figures:
    :return:
    """
    radius = 8
    centers = center_borders(num_figures, window)
    num_figures = centers.shape[0]
    xx, yy, zz = space.define_space(centers.shape[0], [1, 1, 1], (window, window, window))
    xx = xx.permute(1, 2, 3, 0) - centers[:, 0] + torch.randint(low=-3, high=3, size=(num_figures,))
    yy = yy.permute(1, 2, 3, 0) - centers[:, 1] + torch.randint(low=-3, high=3, size=(num_figures,))
    zz = zz.permute(1, 2, 3, 0) - centers[:, 2] + torch.randint(low=-3, high=3, size=(num_figures,))
    xx2 = (xx * xx)
    yy2 = (yy * yy)
    zz2 = (zz * zz)

    rho_2 = xx2 + yy2 + zz2
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1
    mask_sphere = torch.sum(rho <= radius, dim=-1) == 0
    return mask_sphere


def angles_cylinders(num_rotations, max_angle):
    """
    Rotation angles that are used to rotate the object in the scanner. It only generates 6 angles
    :param max_angle: Maximum angle of tilting orientation with respect to the
    :param num_rotations:
    """
    # Parameters of the angles to model the cylinders
    # theta_min = 10.0
    # theta2 = 40.0
    # psi2 = 40.0

    min_rand = -5.0
    max_rand = 5.0
    # max_angle = 25.0

    # Tilt angle of main field
    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float64) + min_rand
    vec_theta = torch.tensor([0.0, 0.0, 0.0, max_angle/2, max_angle/2, max_angle], dtype=torch.float64)
    vec_theta = vec_theta.reshape(1, num_rotations)
    vec_theta += tmp
    vec_theta = torch.deg2rad(vec_theta)

    # Rotation angle with the z axis
    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float64) + min_rand
    vec_psi = torch.tensor([0.0, -max_angle, max_angle, -max_angle/2, max_angle/2, 0.0], dtype=torch.float64)
    vec_psi = vec_psi.reshape(1, num_rotations)
    vec_psi += tmp
    vec_psi = torch.deg2rad(vec_psi)

    return vec_theta, vec_psi


def rand_radius_spheres(n_max_spheres):
    from generate_tensors import WINDOW
    from generate_tensors import RESOLUTION
    delta_radius = np.linalg.norm(RESOLUTION)

    n_spheres = randint(1, n_max_spheres)
    max_r_spheres = WINDOW // 4
    max_r_spheres = np.round(max_r_spheres).item()
    v_radius = delta_radius * torch.randint(low=10, high=max_r_spheres, size=(n_spheres,), dtype=torch.float64)

    return v_radius, n_spheres
